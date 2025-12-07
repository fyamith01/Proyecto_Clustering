"""
Script de Clustering con Reducción de Dimensionalidad
=====================================================

Este script realiza clustering aplicando primero reducción de dimensionalidad,
luego clustering en el espacio reducido, y finalmente mapea los resultados
de vuelta al espacio original.

Modelos utilizados:
- K-means
- Gaussian Mixture Model (GMM)
- DBSCAN
- BIRCH

Autor: Pipeline de Clustering
Fecha: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict

# Scikit-learn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, Birch
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.model_selection import train_test_split

# Configuración
warnings.filterwarnings('ignore')
plt.style.use('default')
sns.set_palette("husl")

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    px = None
    go = None

# ===================== CONFIGURACIÓN =====================
RAW_FILE = "afiliados_mas_usos.csv"
SEP = ";"
RANDOM_STATE = 42

OUTPUT_DIR = Path("outputs_reduccion_dimensional")
DIR_CLEAN = OUTPUT_DIR / "01_clean"
DIR_ANALYSIS = OUTPUT_DIR / "02_analysis"
DIR_CLUSTERING = OUTPUT_DIR / "03_clustering"

# Logging global
logger = logging.getLogger("clustering_reduccion_dimensional")


def setup_logging() -> Path:
    ensure_dirs()
    log_file = OUTPUT_DIR / f"reduccion_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False
    return log_file


def log_info(msg: str) -> None:
    logger.info(msg)


def log_warning(msg: str) -> None:
    logger.warning(msg)

# Parámetros de clustering
K_RANGE = range(2, 12)
N_COMPONENTS_REDUCTION = 3  # Dimensiones para reducción (2 o 3 para clustering)
MIN_CLUSTER_SIZE = 50  # Tamaño mínimo de cluster

# Columnas a excluir del análisis
EXCLUDE_COLS = [
    "DocEmp", "TipoDocEmp", "NomEmp", "CodclaseEconomicaEmp", 
    "CodSeccionEconomicaEmp", "SeccionEconomicaEmp", "NitSinDV",
    "Femenino", "Masculino", "Desconocido", "TarifaA", "TarifaB", "TarifaC",
    "Jovenes", "AdultoJoven", "Adulto", "AdultoMayor","UsosCultura", 
    "UsosEducacion", "UsosSalud", "UsosEmpleo"
]

# ===================== UTILIDADES =====================
def ensure_dirs():
    """Crea las carpetas de salida si no existen"""
    for d in [OUTPUT_DIR, DIR_CLEAN, DIR_ANALYSIS, DIR_CLUSTERING]:
        d.mkdir(parents=True, exist_ok=True)


def detect_csv_separator(file_path: str) -> str:
    """Detecta automáticamente el separador del CSV"""
    log_info(f"=== Detectando separador CSV: {file_path} ===")
    
    separators = [',', ';']
    separator_scores = {}
    
    try:
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            sample_lines = []
            for i, line in enumerate(f):
                if i >= 5:
                    break
                line = line.strip()
                if line:
                    sample_lines.append(line)
        
        for sep in separators:
            field_counts = []
            consistent_parsing = True
            
            for line in sample_lines:
                import csv
                from io import StringIO
                
                try:
                    reader = csv.reader(StringIO(line), delimiter=sep, quotechar='"')
                    fields = next(reader)
                    field_counts.append(len(fields))
                except:
                    consistent_parsing = False
                    break
            
            if consistent_parsing and field_counts:
                consistency = len(set(field_counts)) == 1
                avg_fields = sum(field_counts) / len(field_counts)
                
                separator_scores[sep] = {
                    'consistency': consistency,
                    'avg_fields': avg_fields,
                    'field_counts': field_counts
                }
                
                log_info(f"Separador '{sep}': {field_counts} campos por línea, consistente: {consistency}")
        
        best_sep = ';'  # Default
        if separator_scores:
            best_score = -1
            for sep, score in separator_scores.items():
                current_score = 0
                if score['consistency']:
                    current_score += 100
                if 30 <= score['avg_fields'] <= 40:
                    current_score += 50
                current_score += score['avg_fields']
                
                if current_score > best_score:
                    best_score = current_score
                    best_sep = sep
        
        log_info(f"Separador detectado: '{best_sep}'")
        log_info("=== Fin detección ===")
        return best_sep
        
    except Exception as e:
        log_warning(f"Error en detección de separador: {e}")
        log_warning("Usando separador por defecto: ';'")
        return ';'


def clean_csv_file(input_file: str, output_file: str) -> None:
    """Limpia el CSV con detección automática de separador y headers"""
    import csv
    import re
    
    expected_headers = [
        "DocEmp", "TipoDocEmp", "NomEmp", "CodclaseEconomicaEmp", "CodSeccionEconomicaEmp", "SeccionEconomicaEmp",
        "NitSinDV", "Femenino", "Masculino", "Desconocido", "TarifaA", "TarifaB", "TarifaC",
        "Jovenes", "AdultoJoven", "Adulto", "AdultoMayor", "TotalAfiliados", "AportePC", "TotalUsos",
        "UsosCultura", "UsosEducacion", "UsosSalud", "UsosEmpleo",
        "PorcentajeFemenino", "PorcentajeMasculino", "PorcentajeDesconocido",
        "PorcentajeTarifaA", "PorcentajeTarifaB", "PorcentajeTarifaC", "PorcentajeJovenes",
        "PorcentajeAdultoJoven", "PorcentajeAdulto", "PorcentajeAdultoMayor"
    ]
    
    # Detectar separador automáticamente
    detected_separator = detect_csv_separator(input_file)
    
    # Detectar si el archivo ya tiene headers
    has_headers = False
    with open(input_file, 'r', encoding='utf-8-sig') as infile:
        reader = csv.reader(infile, delimiter=detected_separator, quotechar='"')
        try:
            first_row = next(reader)
            clean_first_row = [field.strip() for field in first_row]
            
            matches = sum(1 for header in expected_headers if header in clean_first_row)
            if matches >= len(expected_headers) * 0.7:
                has_headers = True
                log_info(f"Headers detectados automáticamente ({matches}/{len(expected_headers)} coincidencias)")
            else:
                log_info("No se detectaron headers válidos, se agregarán automáticamente")
        except:
            log_warning("Error al leer primera línea, se agregarán headers automáticamente")
    
    # Procesar archivo completo
    with open(input_file, 'r', encoding='utf-8-sig') as infile, \
         open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        
        reader = csv.reader(infile, delimiter=detected_separator, quotechar='"')
        writer = csv.writer(outfile, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        
        if not has_headers:
            writer.writerow(expected_headers)
        
        for line_num, row in enumerate(reader, 1):
            cleaned_row = []
            for field in row:
                if field.startswith('\ufeff'):
                    field = field[1:]
                field = re.sub(r"\s+", " ", field).strip()
                cleaned_row.append(field)
            
            if has_headers and line_num == 1:
                writer.writerow(cleaned_row)
                continue
            
            if len(cleaned_row) > len(expected_headers):
                cleaned_row = cleaned_row[:len(expected_headers)]
                log_warning(f"Advertencia: Línea {line_num} tenía {len(row)} campos, truncada a {len(expected_headers)}")
            elif len(cleaned_row) < len(expected_headers):
                while len(cleaned_row) < len(expected_headers):
                    cleaned_row.append("")
                log_warning(f"Advertencia: Línea {line_num} tenía {len(row)} campos, rellenada a {len(expected_headers)}")
            
            writer.writerow(cleaned_row)
    
    log_info(f"Archivo limpiado exitosamente. Separador original: '{detected_separator}' → Separador final: ';'")


def load_and_prepare_data(file_path: str) -> Tuple[pd.DataFrame, List[str]]:
    """Carga y prepara los datos para clustering"""
    df = pd.read_csv(file_path, sep=SEP, encoding='utf-8', dtype=str)
    
    # Identificar columnas numéricas
    numeric_cols = []
    for col in df.columns:
        if col not in EXCLUDE_COLS:
            # Intentar convertir a numérico
            try:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
                if df[col].notna().sum() > len(df) * 0.8:  # Al menos 80% de valores válidos
                    numeric_cols.append(col)
            except:
                continue
    
    # Filtrar solo filas con datos completos en columnas numéricas
    df_clean = df[numeric_cols].dropna()
    
    print(f"Datos cargados: {len(df)} filas originales")
    print(f"Datos limpios: {len(df_clean)} filas, {len(numeric_cols)} columnas numéricas")
    
    return df, df_clean, numeric_cols


# ===================== REDUCCIÓN DE DIMENSIONALIDAD =====================
def apply_dimensionality_reduction(X: np.ndarray, method: str = 'pca', n_components: int = 10) -> Tuple[np.ndarray, object]:
    """Aplica reducción de dimensionalidad con supresión de warnings"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        if method == 'pca':
            reducer = PCA(n_components=n_components, random_state=RANDOM_STATE)
        elif method == 'svd':
            reducer = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
        elif method == 'tsne':
            # t-SNE para visualización (máximo 3 componentes)
            n_comp = min(3, n_components)
            reducer = TSNE(n_components=n_comp, random_state=RANDOM_STATE, perplexity=30)
        else:
            raise ValueError(f"Método no soportado: {method}")
        
        X_reduced = reducer.fit_transform(X)
    
    print(f"Reducción {method.upper()}: {X.shape[1]} → {X_reduced.shape[1]} dimensiones")
    if hasattr(reducer, 'explained_variance_ratio_'):
        var_exp = reducer.explained_variance_ratio_.sum()
        print(f"Varianza explicada: {var_exp:.3f}")
    
    return X_reduced, reducer


def plot_variance_explained(pca_model, output_dir: Path):
    """Gráfico de varianza explicada por PCA"""
    if not hasattr(pca_model, 'explained_variance_ratio_'):
        return
    
    var_ratio = pca_model.explained_variance_ratio_
    cum_var = np.cumsum(var_ratio)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Varianza por componente
    ax1.bar(range(1, len(var_ratio) + 1), var_ratio)
    ax1.set_xlabel('Componente Principal')
    ax1.set_ylabel('Varianza Explicada')
    ax1.set_title('Varianza Explicada por Componente')
    
    # Varianza acumulada
    ax2.plot(range(1, len(cum_var) + 1), cum_var, 'bo-')
    ax2.axhline(y=0.8, color='r', linestyle='--', label='80%')
    ax2.axhline(y=0.9, color='g', linestyle='--', label='90%')
    ax2.set_xlabel('Número de Componentes')
    ax2.set_ylabel('Varianza Explicada Acumulada')
    ax2.set_title('Varianza Explicada Acumulada')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "pca_variance_explained.png", dpi=150, bbox_inches='tight')
    plt.close()


# ===================== CLUSTERING =====================
def find_optimal_k(X: np.ndarray, k_range: range) -> int:
    """Encuentra el k óptimo usando silhouette score"""
    scores = []
    for k in k_range:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
            labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        scores.append(score)
    
    optimal_k = k_range[np.argmax(scores)]
    print(f"K óptimo: {optimal_k} (Silhouette: {max(scores):.3f})")
    
    return optimal_k


def calculate_clustering_metrics(X: np.ndarray, labels: np.ndarray) -> Dict:
    """Calcula múltiples métricas de clustering"""
    metrics = {}
    
    # Silhouette Score (mejor: cercano a 1)
    metrics['silhouette'] = silhouette_score(X, labels)
    
    # Calinski-Harabasz Score (mejor: más alto)
    metrics['calinski_harabasz'] = calinski_harabasz_score(X, labels)
    
    # Davies-Bouldin Score (mejor: más bajo)
    metrics['davies_bouldin'] = davies_bouldin_score(X, labels)
    
    # Número de clusters efectivos (excluyendo ruido si existe)
    unique_labels = np.unique(labels)
    metrics['n_clusters'] = len(unique_labels[unique_labels >= 0])  # Excluir -1 (ruido)
    
    return metrics


def apply_clustering_models(X_reduced: np.ndarray, optimal_k: int) -> Dict:
    """Aplica múltiples modelos de clustering en el espacio reducido con métricas múltiples"""
    results = {}
    
    # K-means
    print("Aplicando K-means...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kmeans = KMeans(n_clusters=optimal_k, random_state=RANDOM_STATE, n_init=10)
        kmeans_labels = kmeans.fit_predict(X_reduced)
    kmeans_metrics = calculate_clustering_metrics(X_reduced, kmeans_labels)
    results['kmeans'] = {
        'model': kmeans,
        'labels': kmeans_labels,
        **kmeans_metrics
    }
    
    # Gaussian Mixture Model
    print("Aplicando GMM...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gmm = GaussianMixture(n_components=optimal_k, random_state=RANDOM_STATE)
        gmm_labels = gmm.fit_predict(X_reduced)
    gmm_metrics = calculate_clustering_metrics(X_reduced, gmm_labels)
    results['gmm'] = {
        'model': gmm,
        'labels': gmm_labels,
        **gmm_metrics
    }
    
    # DBSCAN
    print("Aplicando DBSCAN...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Calcular eps automáticamente
        from sklearn.neighbors import NearestNeighbors
        neighbors = NearestNeighbors(n_neighbors=5)
        neighbors_fit = neighbors.fit(X_reduced)
        distances, indices = neighbors_fit.kneighbors(X_reduced)
        distances = np.sort(distances[:, -1])
        eps = np.percentile(distances, 90)
        
        dbscan = DBSCAN(eps=eps, min_samples=5)
        dbscan_labels = dbscan.fit_predict(X_reduced)
    
    # Solo calcular métricas si hay al menos 2 clusters (excluir ruido)
    unique_labels = np.unique(dbscan_labels)
    if len(unique_labels[unique_labels >= 0]) >= 2:
        dbscan_metrics = calculate_clustering_metrics(X_reduced, dbscan_labels)
        results['dbscan'] = {
            'model': dbscan,
            'labels': dbscan_labels,
            **dbscan_metrics
        }
    else:
        print("DBSCAN no generó >=2 clusters válidos; se omiten métricas.")
    
    # BIRCH
    print("Aplicando BIRCH...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        birch = Birch(n_clusters=optimal_k, threshold=0.5)
        birch_labels = birch.fit_predict(X_reduced)
    birch_metrics = calculate_clustering_metrics(X_reduced, birch_labels)
    results['birch'] = {
        'model': birch,
        'labels': birch_labels,
        **birch_metrics
    }
    
    # Mostrar resultados con múltiples métricas
    print("\n=== RESULTADOS DE CLUSTERING ===")
    for name, result in results.items():
        print(f"{name.upper()}:")
        print(f"  Clusters: {result['n_clusters']}")
        print(f"  Silhouette: {result['silhouette']:.3f}")
        print(f"  Calinski-Harabasz: {result['calinski_harabasz']:.3f}")
        print(f"  Davies-Bouldin: {result['davies_bouldin']:.3f}")
        print()
    
    return results


# ===================== VISUALIZACIÓN =====================
def create_2d_plots(X_reduced: np.ndarray, clustering_results: Dict, output_dir: Path):
    """Crea gráficos 2D de los clusters de manera dinámica"""
    if X_reduced.shape[1] < 2:
        print("No hay suficientes dimensiones para gráficos 2D")
        return
    
    n_models = len(clustering_results)
    if n_models == 0:
        print("No hay resultados de clustering para graficar.")
        return
    
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 6))
    if n_models == 1:
        axes = [axes]
    
    for i, (name, result) in enumerate(clustering_results.items()):
        ax = axes[i]
        scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], 
                             c=result['labels'], cmap='tab10', alpha=0.7)
        ax.set_title(f"{name.upper()}\nSilhouette: {result['silhouette']:.3f}")
        ax.set_xlabel('Componente 1')
        ax.set_ylabel('Componente 2')
        plt.colorbar(scatter, ax=ax)
    
    plt.tight_layout()
    plt.savefig(output_dir / "clustering_2d_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()


def create_3d_interactive_plots(X_reduced: np.ndarray, clustering_results: Dict, output_dir: Path):
    """Crea gráficos 3D interactivos con Plotly"""
    if not PLOTLY_AVAILABLE or X_reduced.shape[1] < 3:
        print("No se pueden crear gráficos 3D (Plotly no disponible o < 3 dimensiones)")
        return
    
    for name, result in clustering_results.items():
        fig = go.Figure(data=[go.Scatter3d(
            x=X_reduced[:, 0],
            y=X_reduced[:, 1], 
            z=X_reduced[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=result['labels'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Cluster")
            ),
            text=[f"Cluster: {label}" for label in result['labels']],
            hovertemplate="<b>Cluster: %{text}</b><br>" +
                         "PC1: %{x:.2f}<br>" +
                         "PC2: %{y:.2f}<br>" +
                         "PC3: %{z:.2f}<extra></extra>"
        )])
        
        fig.update_layout(
            title=f"Clustering 3D - {name.upper()} (Silhouette: {result['silhouette']:.3f})",
            scene=dict(
                xaxis_title="Componente 1",
                yaxis_title="Componente 2", 
                zaxis_title="Componente 3"
            ),
            width=800,
            height=600
        )
        
        fig.write_html(output_dir / f"clustering_3d_{name}.html")


# ===================== ANÁLISIS DE RESULTADOS =====================
def analyze_clusters(df_original: pd.DataFrame, clustering_results: Dict, 
                    numeric_cols: List[str], output_dir: Path):
    """Analiza los perfiles de clusters en el espacio original"""
    
    for name, result in clustering_results.items():
        print(f"\n=== ANÁLISIS DE CLUSTERS - {name.upper()} ===")
        
        # Agregar etiquetas al DataFrame original
        df_analysis = df_original.copy()
        df_analysis[f'cluster_{name}'] = result['labels']
        
        # Estadísticas por cluster
        cluster_stats = df_analysis.groupby(f'cluster_{name}')[numeric_cols].agg(['mean', 'median', 'std', 'count'])
        
        # Guardar en Excel
        with pd.ExcelWriter(output_dir / f"cluster_profiles_{name}.xlsx") as writer:
            cluster_stats.to_excel(writer, sheet_name="Estadisticas")
            
            # Distribución de clusters
            cluster_dist = df_analysis[f'cluster_{name}'].value_counts().sort_index()
            cluster_dist.to_excel(writer, sheet_name="Distribucion")
        
        print(f"Distribución de clusters: {cluster_dist.to_dict()}")


def save_final_results(df_original: pd.DataFrame, clustering_results: Dict, 
                      reducer, output_dir: Path):
    """Guarda los resultados finales con todas las columnas originales"""
    
    # DataFrame final con todas las columnas originales
    df_final = df_original.copy()
    
    # Agregar etiquetas de clustering
    for name, result in clustering_results.items():
        df_final[f'cluster_{name}'] = result['labels']
    
    # Guardar archivo final
    df_final.to_csv(output_dir / "datos_con_clusters_finales.csv", index=False, encoding='utf-8')
    
    # Guardar métricas completas
    metrics = {
        'metodo_reduccion': type(reducer).__name__,
        'dimensiones_originales': df_original.select_dtypes(include=[np.number]).shape[1],
        'dimensiones_reducidas': reducer.n_components_ if hasattr(reducer, 'n_components_') else 'N/A',
        'varianza_explicada': reducer.explained_variance_ratio_.sum() if hasattr(reducer, 'explained_variance_ratio_') else 'N/A',
        'modelos_clustering': {
            name: {
                'n_clusters': result['n_clusters'],
                'silhouette_score': result['silhouette'],
                'calinski_harabasz_score': result['calinski_harabasz'],
                'davies_bouldin_score': result['davies_bouldin']
            }
            for name, result in clustering_results.items()
        },
        'mejor_modelo': {
            'criterio': 'silhouette_score',
            'modelo': max(clustering_results.items(), key=lambda x: x[1]['silhouette'])[0],
            'score': max(result['silhouette'] for result in clustering_results.values())
        }
    }
    
    with open(output_dir / "metricas_clustering.json", 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    print(f"\nResultados guardados en: {output_dir.resolve()}")


# ===================== PIPELINE PRINCIPAL =====================
def limpiar_datos():
    """Limpia el CSV de entrada con detección automática de separador"""
    clean_csv_path = DIR_CLEAN / "datos_clustering.csv"
    
    # Siempre regenerar para asegurar detección correcta de separador
    if clean_csv_path.exists():
        print("Eliminando archivo limpio anterior para regenerar con detección automática...")
        clean_csv_path.unlink()
    
    print("Iniciando limpieza con detección automática de separador...")
    clean_csv_file(RAW_FILE, str(clean_csv_path))
    print("Limpieza de CSV completada")
    
    return clean_csv_path


def preparar_datos(clean_csv_path):
    """Carga y prepara los datos"""
    df_original, df_numeric, numeric_cols = load_and_prepare_data(clean_csv_path)
    
    # Estandarización
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_numeric)
    
    print(f"Datos estandarizados: {X_scaled.shape}")
    return df_original, X_scaled, numeric_cols, scaler


def aplicar_reduccion_dimensional(X_scaled):
    """
    Aplica múltiples métodos de reducción de dimensionalidad a 3 dimensiones.
    El clustering se realizará en este espacio reducido (3D) y luego se mapeará
    de vuelta al espacio original.
    """
    print("\n=== REDUCCIÓN DE DIMENSIONALIDAD A 3 DIMENSIONES ===")
    print(f"Reduciendo de {X_scaled.shape[1]} dimensiones → {N_COMPONENTS_REDUCTION} dimensiones")
    
    # Diccionario para almacenar todos los resultados
    reduction_results = {}
    
    # 1. PCA a 3 dimensiones
    print("\nAplicando PCA a 3D...")
    X_pca, pca_reducer = apply_dimensionality_reduction(X_scaled, 'pca', N_COMPONENTS_REDUCTION)
    reduction_results['pca'] = {
        'data': X_pca, 
        'reducer': pca_reducer,
        'original_data': X_scaled  # Guardar datos originales para mapeo inverso
    }
    
    # Visualizar varianza explicada de PCA
    plot_variance_explained(pca_reducer, DIR_ANALYSIS)
    
    # 2. SVD (TruncatedSVD) a 3 dimensiones
    print("\nAplicando SVD a 3D...")
    X_svd, svd_reducer = apply_dimensionality_reduction(X_scaled, 'svd', N_COMPONENTS_REDUCTION)
    reduction_results['svd'] = {
        'data': X_svd, 
        'reducer': svd_reducer,
        'original_data': X_scaled
    }
    
    # 3. t-SNE a 3 dimensiones (solo si hay suficientes datos)
    if len(X_scaled) >= 50:
        print("\nAplicando t-SNE a 3D...")
        try:
            X_tsne, tsne_reducer = apply_dimensionality_reduction(X_scaled, 'tsne', N_COMPONENTS_REDUCTION)
            reduction_results['tsne'] = {
                'data': X_tsne, 
                'reducer': tsne_reducer,
                'original_data': X_scaled
            }
        except Exception as e:
            print(f"Error en t-SNE: {e}")
            print("Continuando sin t-SNE...")
    else:
        print("\nMuy pocos datos para t-SNE, omitiendo...")
    
    print(f"\n✓ Métodos de reducción aplicados: {list(reduction_results.keys())}")
    print(f"✓ Clustering se realizará en espacio de {N_COMPONENTS_REDUCTION} dimensiones")
    print(f"✓ Resultados se mapearán de vuelta al espacio original de {X_scaled.shape[1]} dimensiones")
    
    # Retornar PCA como método principal (compatibilidad)
    return X_pca, pca_reducer, reduction_results


def ejecutar_clustering_multiple(reduction_results):
    """Ejecuta clustering en múltiples espacios reducidos"""
    print("\n=== CLUSTERING EN MÚLTIPLES ESPACIOS REDUCIDOS ===")
    
    all_clustering_results = {}
    
    for method_name, reduction_data in reduction_results.items():
        print(f"\n--- Clustering en espacio {method_name.upper()} ---")
        X_reduced = reduction_data['data']
        
        # Encontrar k óptimo para este espacio
        optimal_k = find_optimal_k(X_reduced, K_RANGE)
        
        # Aplicar modelos de clustering
        clustering_results = apply_clustering_models(X_reduced, optimal_k)
        
        # Agregar información del método de reducción
        for model_name, result in clustering_results.items():
            result['reduction_method'] = method_name
            result['optimal_k'] = optimal_k
        
        all_clustering_results[method_name] = clustering_results
    
    return all_clustering_results


def ejecutar_clustering(X_reduced):
    """Ejecuta los modelos de clustering (función de compatibilidad)"""
    print("\n=== CLUSTERING EN ESPACIO REDUCIDO ===")
    
    # Encontrar k óptimo
    optimal_k = find_optimal_k(X_reduced, K_RANGE)
    
    # Aplicar modelos de clustering
    clustering_results = apply_clustering_models(X_reduced, optimal_k)
    
    return clustering_results


def generar_visualizaciones_multiple(reduction_results, all_clustering_results):
    """Genera visualizaciones para múltiples métodos de reducción"""
    print("\n=== GENERANDO VISUALIZACIONES MÚLTIPLES ===")
    
    for method_name, clustering_results in all_clustering_results.items():
        print(f"Generando visualizaciones para {method_name.upper()}...")
        X_reduced = reduction_results[method_name]['data']
        
        # Crear subdirectorio para este método
        method_dir = DIR_CLUSTERING / method_name
        method_dir.mkdir(exist_ok=True)
        
        # Gráficos 2D
        create_2d_plots(X_reduced, clustering_results, method_dir)
        
        # Gráficos 3D interactivos
        create_3d_interactive_plots(X_reduced, clustering_results, method_dir)


def generar_visualizaciones(X_reduced, clustering_results):
    """Genera visualizaciones de los resultados (función de compatibilidad)"""
    print("\n=== GENERANDO VISUALIZACIONES ===")
    
    # Gráficos 2D
    create_2d_plots(X_reduced, clustering_results, DIR_CLUSTERING)
    
    # Gráficos 3D interactivos
    create_3d_interactive_plots(X_reduced, clustering_results, DIR_CLUSTERING)


def analizar_resultados_multiple(df_original, reduction_results, all_clustering_results, numeric_cols):
    """
    Analiza múltiples resultados y selecciona el mejor.
    
    IMPORTANTE: El clustering se realizó en el espacio reducido (3D), pero las etiquetas
    se asignan directamente a los datos originales (mapeo automático por índice).
    Esto significa que cada fila del dataset original recibe la etiqueta del cluster
    al que pertenece según el análisis en el espacio reducido.
    """
    print("\n=== ANÁLISIS DE RESULTADOS MÚLTIPLES ===")
    print("Nota: Clustering realizado en 3D, etiquetas mapeadas al espacio original\n")
    
    # Recopilar todas las métricas para comparación
    all_metrics = []
    best_overall = {'score': -1, 'method': None, 'model': None, 'results': None}
    
    for method_name, clustering_results in all_clustering_results.items():
        print(f"\nAnalizando resultados de {method_name.upper()}:")
        
        # Análisis de perfiles de clusters para este método
        method_dir = DIR_CLUSTERING / method_name
        analyze_clusters(df_original, clustering_results, numeric_cols, method_dir)
        
        # Encontrar el mejor modelo para este método
        best_method = max(clustering_results.items(), key=lambda x: x[1]['silhouette'])
        method_best_score = best_method[1]['silhouette']
        
        print(f"  Mejor modelo en {method_name}: {best_method[0]} (Silhouette: {method_best_score:.3f})")
        
        # Actualizar mejor global
        if method_best_score > best_overall['score']:
            best_overall = {
                'score': method_best_score,
                'method': method_name,
                'model': best_method[0],
                'results': clustering_results
            }
        
        # Recopilar métricas para comparación
        for model_name, result in clustering_results.items():
            all_metrics.append({
                'reduction_method': method_name,
                'clustering_model': model_name,
                'silhouette_score': result['silhouette'],
                'calinski_harabasz_score': result['calinski_harabasz'],
                'davies_bouldin_score': result['davies_bouldin'],
                'n_clusters': result['n_clusters']
            })
    
    # Crear DataFrame de comparación
    comparison_df = pd.DataFrame(all_metrics)
    comparison_df = comparison_df.sort_values('silhouette_score', ascending=False)
    
    # Guardar comparación
    comparison_df.to_csv(DIR_CLUSTERING / "comparacion_metodos.csv", index=False, encoding='utf-8')
    comparison_df.to_excel(DIR_CLUSTERING / "comparacion_metodos.xlsx", index=False)
    
    print(f"\n🏆 MEJOR COMBINACIÓN GLOBAL:")
    print(f"   Método: {best_overall['method'].upper()}")
    print(f"   Modelo: {best_overall['model'].upper()}")
    print(f"   Silhouette Score: {best_overall['score']:.3f}")
    
    # Guardar resultados finales del mejor método
    best_reducer = reduction_results[best_overall['method']]['reducer']
    save_final_results(df_original, best_overall['results'], best_reducer, DIR_CLUSTERING)
    
    return best_overall, comparison_df


def analizar_resultados(df_original, clustering_results, numeric_cols, reducer):
    """Analiza y guarda los resultados finales (función de compatibilidad)"""
    print("\n=== ANÁLISIS DE RESULTADOS ===")
    
    # Análisis de perfiles de clusters
    analyze_clusters(df_original, clustering_results, numeric_cols, DIR_CLUSTERING)
    
    # Guardar resultados finales
    save_final_results(df_original, clustering_results, reducer, DIR_CLUSTERING)


def run_pipeline():
    """Pipeline principal con múltiples métodos de reducción de dimensionalidad"""
    print("=" * 80)
    print("CLUSTERING CON REDUCCIÓN DE DIMENSIONALIDAD")
    print("=" * 80)
    print("\n📋 METODOLOGÍA:")
    print("   1. Reducir datos a 3 dimensiones (PCA, SVD, t-SNE)")
    print("   2. Hacer clustering en espacio 3D")
    print("   3. Mapear etiquetas al espacio original")
    print("   4. Comparar y seleccionar mejor método")
    print("\n" + "=" * 80 + "\n")
    
    # Ejecutar pipeline modular
    ensure_dirs()
    clean_csv_path = limpiar_datos()
    df_original, X_scaled, numeric_cols, scaler = preparar_datos(clean_csv_path)
    
    # Aplicar múltiples métodos de reducción de dimensionalidad
    X_reduced, reducer, reduction_results = aplicar_reduccion_dimensional(X_scaled)
    
    # Ejecutar clustering en múltiples espacios reducidos
    all_clustering_results = ejecutar_clustering_multiple(reduction_results)
    
    # Generar visualizaciones para todos los métodos
    generar_visualizaciones_multiple(reduction_results, all_clustering_results)
    
    # Analizar resultados y seleccionar el mejor
    best_overall, comparison_df = analizar_resultados_multiple(df_original, reduction_results, all_clustering_results, numeric_cols)
    
    # Resumen final con métricas
    print(f"\n=== CLUSTERING MÚLTIPLE COMPLETADO ===")
    print(f"Resultados guardados en: {OUTPUT_DIR.resolve()}")
    print(f"- Dimensiones originales: {X_scaled.shape[1]}")
    print(f"- Reducción a: {N_COMPONENTS_REDUCTION} dimensiones (para clustering)")
    print(f"- Métodos de reducción probados: {list(reduction_results.keys())}")
    print(f"- Modelos de clustering: K-means, GMM, DBSCAN, Birch")
    print(f"- Métricas evaluadas: Silhouette, Calinski-Harabasz, Davies-Bouldin")
    print(f"- Datos finales: TODAS las columnas originales + etiquetas del mejor modelo")
    print(f"- Mapeo: Clustering en {N_COMPONENTS_REDUCTION}D → Etiquetas en espacio original")
    
    # Mostrar tabla de comparación (top 5)
    print(f"\n📊 TOP 5 COMBINACIONES:")
    top_5 = comparison_df.head(5)
    for idx, row in top_5.iterrows():
        print(f"   {row['reduction_method'].upper()}-{row['clustering_model'].upper()}: "
              f"Silhouette={row['silhouette_score']:.3f}, "
              f"CH={row['calinski_harabasz_score']:.1f}, "
              f"DB={row['davies_bouldin_score']:.3f}")
    
    # Mostrar mejor modelo global
    print(f"\n🏆 MEJOR COMBINACIÓN GLOBAL:")
    print(f"   {best_overall['method'].upper()} + {best_overall['model'].upper()}")
    print(f"   Silhouette Score: {best_overall['score']:.3f}")
    
    if PLOTLY_AVAILABLE:
        print(f"\n- Gráficos 3D interactivos disponibles por método en formato HTML")
        print(f"- Carpetas: {', '.join(reduction_results.keys())}")
    else:
        print("\n- Para gráficos 3D interactivos, instala plotly: pip install plotly")


if __name__ == "__main__":
    run_pipeline()
