"""
Pipeline unificado: limpieza -> análisis (IQR) -> clustering (KMeans/DBSCAN/GMM)
- Configuración embebida (sin CLI) editable en la sección CONFIGURACIÓN.
- Selección de k en KMeans con Silhouette, Calinski‑Harabasz y Gap Statistic (gráfico guardado).
- DBSCAN: eps estimado con la curva de distancias k‑NN (método de la rodilla/knee).
- GMM: n_components seleccionado por BIC (y se grafica BIC/AIC).
- Estabilidad: ARI promedio entre múltiples semillas en TRAIN para KMeans y GMM.
- Feature selection automática (sin forzar columnas específicas):
  1) Detección de columnas numéricas (excluye IDs y categóricas),
  2) Umbral de varianza para descartar constantes,
  3) Pruning de correlación (tipo prune_correlated_features),
  4) Selección greedy hacia adelante maximizando Silhouette con KMeans.
- Outputs organizados en carpetas bajo ./outputs (limpieza, análisis y clustering).
"""

import os
os.environ["MPLBACKEND"] = "Agg"

import csv
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Para visualizaciones 3D interactivas
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly no disponible. Se generarán solo gráficos 2D.")

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, Birch
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, calinski_harabasz_score, adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=FutureWarning)


# ===================== CONFIGURACIÓN =====================
RAW_FILE = "afiliados_mas_usos.csv"  # NOMBRE DEL CSV DE ENTRADA
SEP = ";"  # SEPARADOR DEL CSV: generalmente ';' en datos regionales; cambiar si el CSV usa ',' o '\t'

OUTPUT_DIR = Path("outputs")  # CARPETA RAIZ DE SALIDAS: todos los resultados se guardan bajo esta carpeta
DIR_CLEAN = OUTPUT_DIR / "01_clean"  # SUBCARPETA: archivos de limpieza (CSV limpio)
DIR_ANALYSIS = OUTPUT_DIR / "02_analysis"  # SUBCARPETA: análisis estadístico y sin outliers
DIR_CLUSTER = OUTPUT_DIR / "03_clustering"  # SUBCARPETA: resultados de clustering, perfiles y gráficos

RANDOM_STATE = 42  # SEMILLA GLOBAL: cambia la reproducibilidad de splits y modelos
TEST_SIZE = 0.2  # PORCENTAJE DE TEST: fracción de datos reservada para prueba (0.2 = 20%)
K_RANGE = range(25, 50) # RANGO DE K: valores candidatos para nº de clusters KMeans y GMM (reducido para velocidad)
GAP_B = 4  # N REFERENCIAS GAP: Nº de muestras de referencia uniformes usadas en Gap Statistic (reducido para velocidad)
DBSCAN_MIN_SAMPLES = 10  # MIN_SAMPLES DBSCAN: vecinos mínimos para formar núcleo
MIN_SELECTED_FEATURES = 7  # FS (MÍNIMO): número mínimo de variables a seleccionar por feature selection
MAX_SELECTED_FEATURES = 15  # FS (MÁXIMO): tope superior de variables a seleccionar por greedy forward
CORR_THRESHOLD = 0.97  # FS (CORR): umbral de |rho| Spearman para eliminar columnas muy correlacionadas

# ===================== CONFIGURACIÓN DE COLUMNAS =====================
# COLUMNAS OBLIGATORIAS: estas columnas SIEMPRE se incluyen en el modelo (vacío = ninguna obligatoria)
FORCE_INCLUDE_COLUMNS = []  # Ejemplo: ["PorcentajeFemenino", "TotalAfiliados"]

# COLUMNAS EXCLUIDAS: estas columnas NUNCA se incluyen en el modelo
FORCE_EXCLUDE_COLUMNS = ["DocEmp","TipoDocEmp","NomEmp","CodclaseEconomicaEmp","CodSeccionEconomicaEmp","NitSinDV","Femenino","Masculino","Desconocido","TarifaA","TarifaB","TarifaC","Jovenes","AdultoJoven","Adulto","AdultoMayor"]  # Ejemplo: ["DocEmp", "NitSinDV", "NomEmp"]

# NOTA: Si ambas listas están vacías, el algoritmo selecciona automáticamente todas las columnas numéricas

FULL_ORDER = [
    "DocEmp","TipoDocEmp","NomEmp","CodclaseEconomicaEmp","CodSeccionEconomicaEmp","NitSinDV",
    "Femenino","Masculino","Desconocido",
    "TarifaA","TarifaB","TarifaC",
    "Jovenes","AdultoJoven","Adulto","AdultoMayor",
    "TotalAfiliados","AportePC","SubsidioPC",
    "UsosCultura","UsosEducacion","UsosSalud","UsosEmpleo",
    "PorcentajeFemenino","PorcentajeMasculino","PorcentajeDesconocido",
    "PorcentajeTarifaA","PorcentajeTarifaB","PorcentajeTarifaC",
    "PorcentajeJovenes","PorcentajeAdultoJoven","PorcentajeAdulto","PorcentajeAdultoMayor"
]

NUMERIC_ALL = [
    "Femenino","Masculino","Desconocido",
    "TarifaA","TarifaB","TarifaC",
    "Jovenes","AdultoJoven","Adulto","AdultoMayor",
    "TotalAfiliados","AportePC","SubsidioPC",
    "UsosCultura","UsosEducacion","UsosSalud","UsosEmpleo",
    "PorcentajeFemenino","PorcentajeMasculino","PorcentajeDesconocido",
    "PorcentajeTarifaA","PorcentajeTarifaB","PorcentajeTarifaC",
    "PorcentajeJovenes","PorcentajeAdultoJoven","PorcentajeAdulto","PorcentajeAdultoMayor"
]

CAT_EXPECTED = ["TipoDocEmp","CodclaseEconomicaEmp","CodSeccionEconomicaEmp"]  # Editar si cambian categóricas
ID_LIKE = ["DocEmp","NitSinDV","NomEmp"]                                       # Editar si cambian identificadores


# ===================== UTILIDADES =====================
def ensure_dirs():
    # Crea las carpetas de salida si no existen; seguro para múltiples ejecuciones
    for d in [OUTPUT_DIR, DIR_CLEAN, DIR_ANALYSIS, DIR_CLUSTER]:
        d.mkdir(parents=True, exist_ok=True)


# ----------------- 1) Limpieza CSV -----------------
def clean_csv_file(input_file: str, output_file: str) -> None:
    """
    Limpia el CSV de entrada con reglas simples de formato y define encabezados estándar.
    Parámetros:
    - input_file: ruta al CSV crudo (probablemente exportado con errores de formato)
    - output_file: ruta donde se escribirá el CSV limpio con separador ';' y headers consistentes
    """
    headers = [
        "DocEmp", "TipoDocEmp", "NomEmp", "CodclaseEconomicaEmp", "CodSeccionEconomicaEmp",
        "NitSinDV", "Femenino", "Masculino", "Desconocido", "TarifaA", "TarifaB", "TarifaC",
        "Jovenes", "AdultoJoven", "Adulto", "AdultoMayor", "TotalAfiliados", "AportePC",
        "SubsidioPC", "UsosCultura", "UsosEducacion", "UsosSalud", "UsosEmpleo",
        "PorcentajeFemenino", "PorcentajeMasculino", "PorcentajeDesconocido",
        "PorcentajeTarifaA", "PorcentajeTarifaB", "PorcentajeTarifaC", "PorcentajeJovenes",
        "PorcentajeAdultoJoven", "PorcentajeAdulto", "PorcentajeAdultoMayor"
    ]
    with open(input_file, 'r', encoding='utf-8-sig') as infile, \
         open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile, delimiter=';')
        writer.writerow(headers)
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
            if line_num == 1 and line.startswith('\ufeff'):
                line = line[1:]
            if line.startswith('"') and line.endswith('"'):
                line = line[1:-1]
            fields = line.split(';')
            cleaned = []
            for field in fields:
                if field.startswith('"') and field.endswith('"'):
                    field = field[1:-1]
                field = re.sub(r"\s+", " ", field).strip()
                cleaned.append(field)
            writer.writerow(cleaned)


# ----------------- 2) Análisis + Outliers -----------------
def cargar_y_tipear(path_csv: str, sep: str, numeric_all: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Lee un CSV con separador dado y convierte un conjunto objetivo de columnas numéricas a float.
    Devuelve el DataFrame y la lista de columnas numéricas encontradas.
    Nota: Se usa en la etapa de análisis para estadísticas y outliers.
    """
    df = pd.read_csv(path_csv, sep=sep, dtype=str)
    numeric_cols = [c for c in numeric_all if c in df.columns]
    for c in numeric_cols:
        s = (
            df[c]
            .str.replace(",", ".", regex=False)
            .str.replace(" ", "", regex=False)
        )
        df[c] = pd.to_numeric(s, errors="coerce")
    return df, numeric_cols


def resumen_numericas_en_columnas(df_num: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula un resumen estadístico por columna (métricas en filas) incluyendo conteos y % faltantes.
    """
    desc = df_num.describe(percentiles=[0.25, 0.5, 0.75]).rename(index={"25%":"p25","50%":"p50","75%":"p75"})
    missing = df_num.isna().sum().to_frame().T; missing.index = ["missing"]
    missing_pct = (df_num.isna().mean()*100).round(2).to_frame().T; missing_pct.index=["missing_pct"]
    zeros = (df_num==0).sum().to_frame().T; zeros.index=["zeros"]
    zeros_pct = ((df_num==0).mean()*100).round(2).to_frame().T; zeros_pct.index=["zeros_pct"]
    unique_vals = df_num.nunique().to_frame().T; unique_vals.index=["unique_vals"]
    orden=["count","mean","std","min","p25","p50","p75","max","missing","missing_pct","zeros","zeros_pct","unique_vals"]
    return pd.concat([desc,missing,missing_pct,zeros,zeros_pct,unique_vals], axis=0).reindex(orden)


def eliminar_outliers_iqr(df: pd.DataFrame, numeric_cols: List[str], k: float=1.5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Elimina filas con outliers por el criterio IQR por variable.
    Parámetros:
    - df: DataFrame original
    - numeric_cols: columnas numéricas a evaluar
    - k: multiplicador IQR (1.5 por defecto)
    Devuelve (df_filtrado, tabla_limites_por_variable)
    """
    if not numeric_cols:
        return df.copy(), pd.DataFrame()
    q1 = df[numeric_cols].quantile(0.25, interpolation="linear")
    q3 = df[numeric_cols].quantile(0.75, interpolation="linear")
    iqr = q3 - q1
    lower = q1 - k*iqr
    upper = q3 + k*iqr
    limites = pd.DataFrame({"Q1":q1,"Q3":q3,"IQR":iqr,"lower":lower,"upper":upper})
    keep = pd.Series(True, index=df.index)
    for c in numeric_cols:
        keep &= df[c].isna() | ((df[c] >= lower[c]) & (df[c] <= upper[c]))
    return df.loc[keep].copy(), limites


def boxplots_zscore(df_num: pd.DataFrame, out_png: Path, titulo: str) -> None:
    """
    Genera boxplots verticales tras estandarizar cada variable a z-score.
    - df_num: DataFrame numérico
    - out_png: ruta del PNG de salida
    - titulo: título del gráfico
    """
    if df_num.shape[1] == 0:
        return
    z = (df_num - df_num.mean()) / df_num.std(ddof=0).replace(0, np.nan)
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(max(10, df_num.shape[1]*0.6), 6))
    z.boxplot(column=z.columns.tolist(), vert=True, showmeans=True, meanline=True, ax=ax)
    ax.set_title(titulo + " (z-score)")
    ax.set_ylabel("z")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout(); plt.savefig(out_png, dpi=220); plt.close(fig)


# ----------------- 3) Clustering utils -----------------
def safe_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Calcula Silhouette de forma robusta (requiere al menos 2 clusters distintos, ignora ruido -1).
    Devuelve NaN si no es aplicable.
    """
    labs = set(np.asarray(labels).tolist())
    if -1 in labs: labs.remove(-1)
    if len(labs) < 2: return np.nan
    try:
        return silhouette_score(X, labels)
    except Exception:
        return np.nan


def prune_correlated_features(X_df: pd.DataFrame, threshold: float = CORR_THRESHOLD) -> Tuple[List[str], List[str]]:
    """
    Elimina columnas altamente correlacionadas (|rho| >= threshold) conservando la de mayor varianza.
    Devuelve (keep, dropped).
    """
    corr = X_df.corr(numeric_only=True, method="spearman").abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = set()
    for col in upper.columns:
        highs = [row for row in upper.index if upper.loc[row, col] >= threshold]
        for row in highs:
            drop = row if X_df[row].var() < X_df[col].var() else col
            to_drop.add(drop)
    keep = [c for c in X_df.columns if c not in to_drop]
    return keep, sorted(list(to_drop))


def greedy_forward_selection(X_df: pd.DataFrame,
                             min_feats: int = MIN_SELECTED_FEATURES,
                             max_feats: int = MAX_SELECTED_FEATURES,
                             random_state: int = RANDOM_STATE) -> Tuple[List[str], pd.DataFrame]:
    """
    Selección greedy de variables maximizando Silhouette con KMeans en cada paso.
    - Parte de vacío y agrega la variable que más mejora el Silhouette.
    - Recalcula k óptimo (kmeans_select_k) en cada evaluación para el subconjunto actual.
    - Se detiene al alcanzar max_feats o no haber mejora.
    - Garantiza al menos min_feats (rellenando con las mejores siguientes por ganancia marginal histórica).
    Devuelve (lista de columnas seleccionadas, DataFrame con scores de todas las columnas).
    """
    cols = X_df.columns.tolist()
    selected: List[str] = []
    gains: List[Tuple[str, float]] = []
    remaining = cols.copy()
    current_score = -np.inf
    all_scores = []  # Para guardar scores de todas las columnas evaluadas

    while remaining and len(selected) < max_feats:
        best_feat, best_score = None, current_score
        for feat in remaining:
            trial_cols = selected + [feat]
            X_trial = X_df[trial_cols].values
            # Selección de k para el subconjunto actual (optimizado: menos iteraciones)
            k_star = kmeans_select_k(X_trial, k_range=K_RANGE, rs=random_state)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                km = KMeans(n_clusters=k_star, n_init=5, random_state=random_state).fit(X_trial)  # Reducido n_init
            score = safe_silhouette(X_trial, km.labels_)
            all_scores.append({"feature": feat, "step": len(selected) + 1, "score": score, "selected": False})
            if np.isnan(score):
                continue
            if score > best_score + 1e-6:
                best_feat, best_score = feat, score
        if best_feat is None:
            break
        selected.append(best_feat)
        gains.append((best_feat, best_score - current_score if current_score != -np.inf else best_score))
        current_score = best_score
        remaining.remove(best_feat)
        # Marcar como seleccionada
        for score_record in all_scores:
            if score_record["feature"] == best_feat and score_record["step"] == len(selected):
                score_record["selected"] = True

    if len(selected) < min_feats:
        # Completar con las variables restantes de mayor ganancia marginal (recomputada en un pase)
        extra_scores = []
        for feat in remaining:
            trial_cols = selected + [feat]
            X_trial = X_df[trial_cols].values
            k_star = kmeans_select_k(X_trial, k_range=K_RANGE, rs=random_state)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                km = KMeans(n_clusters=k_star, n_init=5, random_state=random_state).fit(X_trial)
            score = safe_silhouette(X_trial, km.labels_)
            extra_scores.append((feat, score))
            all_scores.append({"feature": feat, "step": "extra", "score": score, "selected": False})
        extra_scores.sort(key=lambda t: (-(t[1] if not np.isnan(t[1]) else -np.inf)))
        for feat, _ in extra_scores:
            if len(selected) >= min_feats:
                break
            selected.append(feat)
            # Marcar como seleccionada
            for score_record in all_scores:
                if score_record["feature"] == feat and score_record["step"] == "extra":
                    score_record["selected"] = True

    # Crear DataFrame con todos los scores
    scores_df = pd.DataFrame(all_scores)
    return selected, scores_df


def kmeans_select_k(X: np.ndarray, k_range=K_RANGE, out_dir: Optional[Path]=None, rs: int=RANDOM_STATE) -> int:
    """
    Selecciona k óptimo para KMeans usando solo Silhouette (más rápido que el método anterior).
    - X: matriz de características (escaladas)
    - k_range: rango de k a explorar
    - out_dir: carpeta donde guardar el gráfico de selección (si se provee)
    - rs: semilla
    Devuelve el k recomendado (int).
    """
    # Limitar k al nº de filas únicas para evitar clusters vacíos por duplicados
    unique_count = np.unique(X, axis=0).shape[0]
    k_list = [k for k in k_range if 2 <= k <= unique_count]
    if not k_list:
        return 2
    
    # Optimización: usar solo Silhouette para ser más rápido
    sils = []
    for k in k_list:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            km = KMeans(n_clusters=k, n_init=3, random_state=rs).fit(X)  # Reducido n_init a 3
        labels = km.labels_
        sil = safe_silhouette(X, labels)
        sils.append(sil if not np.isnan(sil) else 0)
    
    # Seleccionar k con mejor Silhouette
    best_idx = int(np.argmax(sils))
    best_k = int(list(k_list)[best_idx])
    
    if out_dir:
        plt.figure(figsize=(9,4))
        plt.plot(list(k_list), sils, "-o", label="silhouette")
        plt.axvline(best_k, color='k', ls='--', label=f"k*={best_k}")
        plt.xlabel("Número de clusters (k)")
        plt.ylabel("Silhouette Score")
        plt.title("Selección de k para KMeans")
        plt.legend(); plt.tight_layout(); plt.savefig(out_dir/"kmeans_selection.png", dpi=150, bbox_inches='tight'); plt.close()
    return int(best_k)


def dbscan_estimate_eps(X: np.ndarray, min_samples: int=DBSCAN_MIN_SAMPLES, out_dir: Optional[Path]=None) -> float:
    """
    Estima eps para DBSCAN usando la curva de distancias del k‑ésimo vecino (k=min_samples).
    Devuelve el valor de eps aproximado y guarda una gráfica si out_dir no es None.
    """
    # Optimización: usar solo una muestra de los datos para estimar eps más rápido
    if len(X) > 5000:
        # Fix: usar np.random.RandomState en lugar de random_state en choice
        rng = np.random.RandomState(RANDOM_STATE)
        sample_idx = rng.choice(len(X), size=5000, replace=False)
        X_sample = X[sample_idx]
    else:
        X_sample = X
    
    nn = NearestNeighbors(n_neighbors=min_samples, n_jobs=-1)
    nn.fit(X_sample)
    dists, _ = nn.kneighbors(X_sample)
    kth = np.sort(dists[:, -1])
    
    # Knee por máxima distancia a la recta entre extremos
    x = np.arange(len(kth))
    if len(x) < 2:
        eps = 0.5  # Valor por defecto si hay muy pocos puntos
    else:
        p1 = np.array([x[0], kth[0]]); p2 = np.array([x[-1], kth[-1]])
        line = p2 - p1
        if np.linalg.norm(line) == 0:
            eps = float(np.percentile(kth, 75))  # Usar percentil 75 si no hay variación
        else:
            norm = np.array([-line[1], line[0]])
            dist = np.abs((np.stack([x, kth], axis=1) - p1) @ norm) / np.linalg.norm(line)
            idx = int(np.argmax(dist))
            eps = float(kth[idx])
    
    # Asegurar que eps sea válido (mayor que 0)
    if eps <= 0:
        eps = float(np.percentile(kth, 75))  # Fallback al percentil 75
    if eps <= 0:
        eps = 0.5  # Último recurso
    
    if out_dir:
        plt.figure(figsize=(8,4))
        plt.plot(kth)
        plt.axvline(idx if 'idx' in locals() else len(kth)//2, color='r', ls='--')
        plt.axhline(eps, color='r', ls='--')
        plt.title(f"kNN distance curve (k={min_samples}) eps≈{eps:.3f}")
        plt.tight_layout()
        plt.savefig(out_dir/"dbscan_knn_curve.png", dpi=150, bbox_inches='tight')
        plt.close()
    return eps


def gmm_select_components(X: np.ndarray, k_range=K_RANGE, rs: int=RANDOM_STATE, out_dir: Optional[Path]=None) -> int:
    """
    Selecciona el nº de componentes para GMM minimizando BIC (optimizado para velocidad).
    """
    # Limitar k al nº de filas únicas
    unique_count = np.unique(X, axis=0).shape[0]
    k_list = [k for k in k_range if 2 <= k <= unique_count]
    if not k_list:
        return 2
    
    bics, aics = [], []
    for k in k_list:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            gm = GaussianMixture(n_components=k, covariance_type="full", random_state=rs, max_iter=50).fit(X)  # Reducido max_iter
        bics.append(gm.bic(X)); aics.append(gm.aic(X))
    
    best_idx = int(np.argmin(bics))
    best_k = int(list(k_list)[best_idx])
    
    if out_dir:
        plt.figure(figsize=(9,4))
        plt.plot(list(k_list), bics, "-o", label="BIC")
        plt.plot(list(k_list), aics, "-o", label="AIC")
        plt.axvline(best_k, color='k', ls='--', label=f"k*={best_k}")
        plt.xlabel("Número de componentes")
        plt.ylabel("BIC / AIC")
        plt.title("Selección de componentes para GMM")
        plt.legend(); plt.tight_layout(); plt.savefig(out_dir/"gmm_bic_aic.png", dpi=150, bbox_inches='tight'); plt.close()
    return best_k


def ari_stability(model_ctor, X: np.ndarray, n_runs: int=5, rs: int=RANDOM_STATE) -> float:
    """
    Mide estabilidad de clusters como ARI promedio entre múltiples ejecuciones (semillas distintas).
    - model_ctor: función que recibe random_state y devuelve un modelo con fit/fit_predict
    - X: matriz de entrenamiento
    - n_runs: número de repeticiones
    - rs: semilla base
    """
    labels = []
    for i in range(n_runs):
        model = model_ctor(random_state=rs+i)
        if hasattr(model, 'fit_predict'):
            lab = model.fit_predict(X)
        else:
            lab = model.fit(X).predict(X)
        labels.append(lab)
    # ARI promedio entre pares
    pairs = []
    for i in range(n_runs):
        for j in range(i+1, n_runs):
            pairs.append(adjusted_rand_score(labels[i], labels[j]))
    return float(np.mean(pairs)) if pairs else np.nan


# ----------------- 4) Orquestación -----------------
def create_3d_plots(coords_2d: np.ndarray, coords_3d: np.ndarray, labels: np.ndarray, 
                   title: str, out_dir: Path, model_name: str) -> None:
    """
    Crea visualizaciones 3D interactivas con plotly y guarda también versiones estáticas.
    """
    if not PLOTLY_AVAILABLE:
        print(f"Plotly no disponible. Saltando visualización 3D para {model_name}")
        return
    
    # Crear DataFrame para plotly
    df_plot = pd.DataFrame({
        'PC1': coords_2d[:, 0],
        'PC2': coords_2d[:, 1], 
        'PC3': coords_3d[:, 2] if coords_3d.shape[1] > 2 else np.zeros(len(coords_2d)),
        'Cluster': labels,
        'Model': model_name
    })
    
    # Gráfico 3D interactivo
    fig_3d = px.scatter_3d(df_plot, x='PC1', y='PC2', z='PC3', 
                          color='Cluster', title=f"{title} - Vista 3D Interactiva",
                          color_discrete_sequence=px.colors.qualitative.Set1)
    fig_3d.update_layout(scene=dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3'))
    fig_3d.write_html(str(out_dir / f"{model_name.lower()}_3d_interactive.html"))
    
    # Gráfico 2D con plotly (más interactivo que matplotlib)
    fig_2d = px.scatter(df_plot, x='PC1', y='PC2', color='Cluster', 
                       title=f"{title} - Vista 2D Interactiva",
                       color_discrete_sequence=px.colors.qualitative.Set1)
    fig_2d.write_html(str(out_dir / f"{model_name.lower()}_2d_interactive.html"))


def run_pipeline():
    """
    Función principal que ejecuta todo el pipeline modularizado.
    """
    print("=== INICIANDO PIPELINE DE CLUSTERING ===")
    ensure_dirs()  # Asegura estructura de carpetas de salida

    # 1) Limpieza
    clean_csv_path = DIR_CLEAN / "datos_clustering.csv"  # Ruta destino del CSV limpio
    if not clean_csv_path.exists():  # Limpia solo si aún no existe (evita reprocesar innecesariamente)
        clean_csv_file(RAW_FILE, str(clean_csv_path))  # Ejecuta limpieza de CSV crudo a limpio

    # 2) Análisis + outliers
    df, numeric_cols = cargar_y_tipear(str(clean_csv_path), SEP, NUMERIC_ALL)  # Carga CSV limpio y tipa numéricas esperadas
    df_num = df[numeric_cols].copy()  # Subconjunto numérico para estadísticas
    resumen_original = resumen_numericas_en_columnas(df_num)  # Resumen con outliers
    df_clean, _ = eliminar_outliers_iqr(df, numeric_cols, k=1.5)  # Remueve outliers por IQR
    df_num_clean = df_clean[numeric_cols].copy()  # Numéricas sin outliers
    resumen_sin_out = resumen_numericas_en_columnas(df_num_clean)  # Resumen sin outliers
    # Guardados
    with pd.ExcelWriter(DIR_ANALYSIS/"resumen_estadistico_numericas.xlsx") as writer:  # Exporta ambos resúmenes a Excel
        resumen_original.to_excel(writer, sheet_name="Numericas_Original")
        resumen_sin_out.to_excel(writer, sheet_name="Numericas_SinOutliers")
    cols_present = [c for c in FULL_ORDER if c in df_clean.columns]  # Asegura orden consistente si existen todas
    sin_out_csv = DIR_ANALYSIS/"dataset_sin_outliers.csv"  # Dataset base para clustering (filas sin outliers)
    df_clean[cols_present].to_csv(sin_out_csv, sep=SEP, index=False, encoding="utf-8-sig")  # Guarda CSV filtrado
    boxplots_zscore(df_num, DIR_ANALYSIS/"boxplots_original_z.png", "Boxplots numericas (ORIGINAL)")  # Gráfico z-score original
    boxplots_zscore(df_num_clean, DIR_ANALYSIS/"boxplots_sin_outliers_z.png", "Boxplots numericas (SIN outliers)")  # Gráfico z-score sin outliers

    # 3) Clustering sobre dataset_sin_outliers (detección automática de numéricas)
    work = pd.read_csv(sin_out_csv, sep=SEP, dtype=str)  # Carga dataset filtrado (todo como string para tipificar luego)
    # Convertir a numérico todas las columnas posibles, dejando categóricas/ID intactas
    numeric_candidates = []
    for col in work.columns:  # Recorre todas las columnas para detectar cuáles son numéricas
        s_try = (
            work[col].astype(str)
            .str.replace(",", ".", regex=False)
            .str.replace(" ", "", regex=False)
        )
        num = pd.to_numeric(s_try, errors="coerce")
        # Si >=80% de filas se pueden convertir a número y no es ID/Categórica, se considera numérica para el modelo
        if num.notna().mean() >= 0.8 and col not in ID_LIKE + CAT_EXPECTED:
            work[col] = num
            numeric_candidates.append(col)
    
    # Aplicar configuración de columnas obligatorias y excluidas
    # 1. Agregar columnas obligatorias si existen y están en el dataset
    for col in FORCE_INCLUDE_COLUMNS:
        if col in work.columns and col not in numeric_candidates:
            # Intentar convertir a numérico si no está ya
            s_try = (
                work[col].astype(str)
                .str.replace(",", ".", regex=False)
                .str.replace(" ", "", regex=False)
            )
            num = pd.to_numeric(s_try, errors="coerce")
            if num.notna().mean() >= 0.8:
                work[col] = num
                if col not in numeric_candidates:
                    numeric_candidates.append(col)
    
    # 2. Remover columnas excluidas
    numeric_candidates = [col for col in numeric_candidates if col not in FORCE_EXCLUDE_COLUMNS]
    
    # 3. Si no hay columnas obligatorias, usar todas las detectadas; si las hay, asegurar que estén incluidas
    if FORCE_INCLUDE_COLUMNS:
        final_candidates = []
        for col in FORCE_INCLUDE_COLUMNS:
            if col in numeric_candidates:
                final_candidates.append(col)
        # Agregar el resto de columnas no obligatorias
        for col in numeric_candidates:
            if col not in final_candidates:
                final_candidates.append(col)
        numeric_candidates = final_candidates
    
    # Matriz de trabajo: solo columnas numéricas detectadas y filas completas (sin NaN)
    work_df = work[numeric_candidates].dropna().copy()
    aux_cols = [c for c in (ID_LIKE + CAT_EXPECTED) if c in work.columns]  # Columnas auxiliares para perfiles/resultados
    aux_df = work.loc[work_df.index, aux_cols].copy()  # Se alinean con las filas válidas de work_df

    # Estandarización
    scaler = StandardScaler()  # Escalador estándar (media 0, var 1) para mejorar métricas y PCA
    X = scaler.fit_transform(work_df.values)  # Ajusta en todo el conjunto filtrado y transforma

    # Feature selection automática: varianza -> correlación -> greedy forward
    X_df = pd.DataFrame(X, columns=numeric_candidates, index=work_df.index)  # DataFrame escalado con nombres de columnas
    # 1) Variance threshold (descartar columnas con var≈0 tras escalar; std≈0 implica var≈0)
    variances = X_df.var(axis=0)  # Varianzas por columna después de escalar
    keep_var = variances[variances > 1e-12].index.tolist()  # Umbral muy pequeño para eliminar constantes
    X_df = X_df[keep_var]  # Mantener columnas con varianza
    # 2) Correlation pruning
    keep_corr, _ = prune_correlated_features(X_df, threshold=CORR_THRESHOLD)  # Elimina variables redundantes por alta correlación
    X_df = X_df[keep_corr]  # Mantiene solo las columnas poco correlacionadas
    # 3) Greedy forward selection
    selected_columns, feature_scores_df = greedy_forward_selection(  # Selección final de variables por greedy
        X_df,
        min_feats=MIN_SELECTED_FEATURES,
        max_feats=MAX_SELECTED_FEATURES,
        random_state=RANDOM_STATE,
    )
    X_sel = X_df[selected_columns].values  # Matriz final con columnas seleccionadas
    
    # Guardar scores de feature selection
    feature_scores_df.to_csv(DIR_CLUSTER/"feature_selection_scores.csv", index=False, encoding="utf-8")
    print(f"Feature selection completada. {len(selected_columns)} columnas seleccionadas de {len(numeric_candidates)} candidatas.")

    # 4) Análisis de correlación y pairplot con variables seleccionadas
    X_df_selected = X_df[selected_columns].copy()
    
    # Matriz de correlación
    corr_matrix = X_df_selected.corr(numeric_only=True, method="spearman")
    corr_matrix.to_csv(DIR_CLUSTER/"correlation_selected.csv", encoding="utf-8", index=True)
    
    # Heatmap de correlación
    plt.figure(figsize=(max(10, 0.6*len(corr_matrix.columns)), max(8, 0.6*len(corr_matrix.columns))))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', center=0, 
                square=False, linewidths=0.2, cbar_kws={"shrink": 0.7})
    plt.title("Matriz de Correlación - Variables Seleccionadas", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(DIR_CLUSTER/"correlation_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Pairplot (limitado a 8 variables para evitar figuras enormes)
    pairplot_vars = selected_columns[:8] if len(selected_columns) > 8 else selected_columns
    pair_df = X_df_selected[pairplot_vars].copy()
    if len(pair_df) > 2000:  # Muestra aleatoria para aligerar PNG
        pair_df = pair_df.sample(2000, random_state=RANDOM_STATE)
    
    g = sns.pairplot(pair_df, diag_kind="kde", corner=True, plot_kws={'alpha': 0.6})
    g.fig.suptitle("Pairplot - Variables Seleccionadas (sin outliers)", y=1.02, fontsize=14)
    g.savefig(DIR_CLUSTER/"pairplot_selected.png", dpi=150, bbox_inches='tight')
    plt.close('all')

    # Selección de k y modelos sobre X_sel
    print("Iniciando selección de hiperparámetros...")
    model_dir = DIR_CLUSTER
    print("  - Seleccionando k óptimo para KMeans...")
    best_k_kmeans = kmeans_select_k(X_sel, k_range=K_RANGE, out_dir=model_dir, rs=RANDOM_STATE)  # k óptimo KMeans
    print(f"  - KMeans: k={best_k_kmeans}")
    print("  - Estimando eps para DBSCAN...")
    eps_db = dbscan_estimate_eps(X_sel, min_samples=DBSCAN_MIN_SAMPLES, out_dir=model_dir)  # eps óptimo DBSCAN
    print(f"  - DBSCAN: eps={eps_db:.3f}")
    print("  - Seleccionando componentes para GMM...")
    best_k_gmm = gmm_select_components(X_sel, k_range=K_RANGE, rs=RANDOM_STATE, out_dir=model_dir)  # componentes óptimos GMM
    print(f"  - GMM: n_components={best_k_gmm}")
    # BIRCH: selección rápida de n_clusters por silhouette en grid pequeño
    print("  - Seleccionando n_clusters para BIRCH...")
    birch_candidates = [None] + list(range(3, min(20, X_sel.shape[1] + 5)))
    best_b_sil, best_b_nc = -1.0, None
    for nc in birch_candidates:
        try:
            b = Birch(n_clusters=nc, threshold=0.5).fit(X_sel)
            lab = b.labels_
            sil = safe_silhouette(X_sel, lab)
            if not np.isnan(sil) and sil > best_b_sil:
                best_b_sil, best_b_nc = sil, nc
        except Exception:
            continue
    print(f"  - BIRCH: n_clusters={best_b_nc} (silhouette={best_b_sil:.3f})")

    # Holdout
    idx_all = np.arange(X_sel.shape[0])
    idx_train, idx_test = train_test_split(idx_all, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True)
    Xtr, Xte = X_sel[idx_train], X_sel[idx_test]

    # Entrenar y predecir
    print("Entrenando modelos finales...")
    # Seguridad adicional: capear k al nº de puntos únicos de TRAIN
    unique_train = np.unique(Xtr, axis=0).shape[0]
    k_final = min(best_k_kmeans, max(2, unique_train))
    print(f"  - Entrenando KMeans (k={k_final})...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        km = KMeans(n_clusters=k_final, n_init=10, random_state=RANDOM_STATE).fit(Xtr)  # Entrena KMeans final
    print(f"  - Entrenando DBSCAN (eps={eps_db:.3f})...")
    db = DBSCAN(eps=eps_db, min_samples=DBSCAN_MIN_SAMPLES, n_jobs=-1).fit(Xtr)  # Entrena DBSCAN final
    print(f"  - Entrenando GMM (n_components={best_k_gmm})...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        gm = GaussianMixture(n_components=best_k_gmm, covariance_type="full", random_state=RANDOM_STATE, max_iter=100).fit(Xtr)  # Entrena GMM final
    print(f"  - Entrenando BIRCH (n_clusters={best_b_nc})...")
    birch_final = Birch(n_clusters=best_b_nc, threshold=0.5).fit(Xtr)

    km_all = np.concatenate([km.labels_, km.predict(Xte)])
    # DBSCAN predict a test usando radios a núcleos (asignación por vecinos de core)
    def dbscan_predict(dbscan_model: DBSCAN, X_train: np.ndarray, X_new: np.ndarray, eps: float) -> np.ndarray:
        core_idx = getattr(dbscan_model, "core_sample_indices_", None)
        if core_idx is None or len(core_idx)==0:
            return np.full(X_new.shape[0], -1, dtype=int)
        core_samples = X_train[core_idx]
        core_labels = dbscan_model.labels_[core_idx]
        nn = NearestNeighbors(radius=eps, n_jobs=-1).fit(core_samples)
        neighs = nn.radius_neighbors(X_new, return_distance=False)
        out = []
        for idxs in neighs:
            labs = core_labels[idxs]
            labs = [l for l in labs if l!=-1]
            out.append(pd.Series(labs).mode().iloc[0] if len(labs) else -1)
        return np.array(out, dtype=int)

    db_all = np.concatenate([db.labels_, dbscan_predict(db, Xtr, Xte, eps_db)])
    gm_all = np.concatenate([gm.predict(Xtr), gm.predict(Xte)])
    birch_all = np.concatenate([birch_final.labels_, Birch(n_clusters=best_b_nc, threshold=0.5).fit(Xte).labels_])

    # Estabilidad (ARI promedio entre seeds) en TRAIN
    ari_km = ari_stability(lambda random_state: KMeans(n_clusters=best_k_kmeans, n_init=1, random_state=random_state), Xtr)  # Estabilidad KMeans
    ari_gm = ari_stability(lambda random_state: GaussianMixture(n_components=best_k_gmm, covariance_type="full", random_state=random_state), Xtr)  # Estabilidad GMM

    # PCA para visualización
    pca = PCA(n_components=2, random_state=RANDOM_STATE).fit(Xtr)
    coords_all = pca.transform(np.vstack([Xtr, Xte]))

    # Salida consolidada
    out_idx = np.concatenate([idx_train, idx_test])
    out_original = work.loc[work_df.index].iloc[out_idx].copy()  # Recompone DataFrame original en el orden (TRAIN+TEST)
    keep_cols = []
    for lst in [ID_LIKE, CAT_EXPECTED, selected_columns]:
        for c in lst:
            if c in out_original.columns and c not in keep_cols:
                keep_cols.append(c)
    out = out_original[keep_cols].copy()  # Mantiene IDs/categóricas y variables seleccionadas
    out[["PC1","PC2"]] = coords_all
    out["cluster_kmeans"] = km_all
    out["cluster_dbscan"] = db_all
    out["cluster_gmm"] = gm_all
    out_path = DIR_CLUSTER/"Clustering_resultados.csv"
    out.to_csv(out_path, index=False, encoding="utf-8")

    # Gráficos PCA 2D
    def plot_clusters(coords: np.ndarray, labels: np.ndarray, title: str, out_path: Path) -> None:
        plt.figure(figsize=(9,7))
        for lab in sorted(set(labels)):
            m = labels==lab
            lbl = f"cluster {lab}" if lab!=-1 else "ruido (-1)"
            plt.scatter(coords[m,0], coords[m,1], s=14, label=lbl)
        plt.title(title); plt.xlabel("PC1"); plt.ylabel("PC2")
        plt.legend(markerscale=1.5, fontsize=9)
        plt.tight_layout()
        plt.savefig(out_path, dpi=140, bbox_inches='tight')  # bbox_inches para evitar el warning de tight_layout
        plt.close()

    plot_clusters(coords_all, km_all, f"KMeans (k={best_k_kmeans}) en PCA 2D", DIR_CLUSTER/"kmeans_pca.png")
    plot_clusters(coords_all, db_all, f"DBSCAN (eps={eps_db:.3f}, min_samples={DBSCAN_MIN_SAMPLES}) en PCA 2D", DIR_CLUSTER/"dbscan_pca.png")
    plot_clusters(coords_all, gm_all, f"GMM (k={best_k_gmm}) en PCA 2D", DIR_CLUSTER/"gmm_pca.png")
    plot_clusters(coords_all, birch_all, f"BIRCH (n_clusters={best_b_nc}) en PCA 2D", DIR_CLUSTER/"birch_pca.png")
    
    # Gráficos PCA 3D interactivos
    if PLOTLY_AVAILABLE:
        print("Generando visualizaciones 3D interactivas...")
        # PCA 3D para visualización
        pca_3d = PCA(n_components=3, random_state=RANDOM_STATE).fit(Xtr)
        coords_3d = pca_3d.transform(np.vstack([Xtr, Xte]))
        
        create_3d_plots(coords_all, coords_3d, km_all, f"KMeans (k={best_k_kmeans})", DIR_CLUSTER, "kmeans")
        create_3d_plots(coords_all, coords_3d, db_all, f"DBSCAN (eps={eps_db:.3f})", DIR_CLUSTER, "dbscan")
        create_3d_plots(coords_all, coords_3d, gm_all, f"GMM (k={best_k_gmm})", DIR_CLUSTER, "gmm")
        create_3d_plots(coords_all, coords_3d, birch_all, f"BIRCH (n_clusters={best_b_nc})", DIR_CLUSTER, "birch")
        print("Gráficos 3D interactivos guardados en formato HTML.")

    # Perfiles por cluster (numéricos y categóricos)
    def profiles_numeric(df_full: pd.DataFrame, labels: np.ndarray, numeric_cols: List[str], out_prefix: Path) -> None:
        """
        Calcula perfiles numéricos por cluster y guarda CSV aplanado + Excel (multiindex/flat/long).
        - df_full: DataFrame con columnas numéricas originales (no escaladas) + categóricas
        - labels: etiquetas de cluster (mismo orden que df_full)
        - numeric_cols: columnas numéricas a perfilar
        - out_prefix: prefijo/ruta base de salida
        """
        ser = pd.Series(labels, name="cluster")
        prof = df_full.groupby(ser)[numeric_cols].agg(["median","mean","std","min","max"]).copy()
        flat = prof.copy(); flat.columns = [f"{c}__{s}" for c,s in flat.columns]
        flat.index.name = "cluster"
        flat.to_csv(str(out_prefix.with_name(out_prefix.name+"_flat.csv")), encoding="utf-8")
        with pd.ExcelWriter(str(out_prefix.with_suffix('.xlsx'))) as writer:
            prof.to_excel(writer, sheet_name="multiindex")
            flat.to_excel(writer, sheet_name="flat")
            long = prof.stack(level=0).rename_axis(["cluster","variable"]).reset_index()
            long.to_excel(writer, sheet_name="long_tidy", index=False)

    def profiles_categorical(df_full: pd.DataFrame, labels: np.ndarray, cat_cols: List[str], out_prefix: Path) -> None:
        """
        Calcula distribuciones relativas por cluster para columnas categóricas y guarda un CSV por variable.
        """
        ser = pd.Series(labels, name="cluster")
        for c in cat_cols:
            if c in df_full.columns:
                ct = pd.crosstab(ser, df_full[c], normalize="index")
                ct.to_csv(str(out_prefix.with_name(out_prefix.name+f"_catdist_{c}.csv")), encoding="utf-8")

    # Perfiles de cluster (solo numéricos - todas las columnas numéricas, no solo las seleccionadas)
    all_numeric_cols = [c for c in numeric_candidates if c in work.columns]
    full_for_profiles = work.loc[work_df.index].iloc[out_idx][all_numeric_cols].copy()
    profiles_numeric(full_for_profiles, km_all, all_numeric_cols, DIR_CLUSTER/"cluster_profiles_kmeans")
    profiles_numeric(full_for_profiles, db_all, all_numeric_cols, DIR_CLUSTER/"cluster_profiles_dbscan")
    profiles_numeric(full_for_profiles, gm_all, all_numeric_cols, DIR_CLUSTER/"cluster_profiles_gmm")
    profiles_numeric(full_for_profiles, birch_all, all_numeric_cols, DIR_CLUSTER/"cluster_profiles_birch")

    # Métricas
    metrics = pd.DataFrame([
        {"modelo":"KMeans","param":f"k={best_k_kmeans}","silhouette":safe_silhouette(Xte, km.predict(Xte)),"ARI_stability_train":ari_km},
        {"modelo":"DBSCAN","param":f"eps={eps_db:.3f}, min_samples={DBSCAN_MIN_SAMPLES}","silhouette":safe_silhouette(Xte, dbscan_predict(db, Xtr, Xte, eps_db)),"ARI_stability_train":np.nan},
        {"modelo":"GMM","param":f"n_components={best_k_gmm}","silhouette":safe_silhouette(Xte, gm.predict(Xte)),"ARI_stability_train":ari_gm},
        {"modelo":"BIRCH","param":f"n_clusters={best_b_nc}","silhouette":safe_silhouette(X_sel, Birch(n_clusters=best_b_nc, threshold=0.5).fit(X_sel).labels_),"ARI_stability_train":np.nan},
    ])
    metrics.to_csv(DIR_CLUSTER/"model_scores.csv", index=False, encoding="utf-8")

    print(f"\n=== PIPELINE COMPLETADO ===")
    print(f"Resultados guardados en: {OUTPUT_DIR.resolve()}")
    if PLOTLY_AVAILABLE:
        print("Gráficos 3D interactivos disponibles en formato HTML en la carpeta de clustering.")
    else:
        print("Para gráficos 3D interactivos, instala plotly: pip install plotly")


if __name__ == "__main__":
    run_pipeline()


