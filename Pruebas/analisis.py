"""
Resumen estadístico de variables numéricas (en columnas) + eliminación de outliers por IQR
- Genera 2 resúmenes: Original (con outliers) y SinOutliers (filtrado por IQR)
- Exporta a Excel con 2 hojas: Numericas_Original, Numericas_SinOutliers
- Exporta CSV con TODAS las columnas en el MISMO ORDEN del archivo original, pero SIN filas outliers
- Crea boxplots horizontales estéticos para ambas versiones
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ====== CONSTANTES =======
FILE_NAME = "datos_clustering.csv"
SEP = ";"                                      

OUTPUT_XLSX = "resumen_estadistico_numericas.xlsx"      
CSV_SIN_OUT = "dataset_sin_outliers.csv"        

PNG_ORIGINAL = "boxplots_numericas_original.png"        
PNG_SIN_OUT = "boxplots_numericas_sin_outliers.png"      

IQR_K = 1.5  # Multiplicador de IQR (1.5 estándar)

# Orden para salida del csv sin outlayers
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

# Columnas numéricas (para resumen y detección de outlayers)
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

# ===== FUNCIONES =========
def cargar_y_tipear(path_csv: str, sep: str, numeric_all: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Lee el CSV como texto, convierte las columnas numéricas presentes a float (coma decimal -> punto)."""
    df = pd.read_csv(path_csv, sep=sep, dtype=str)
    numeric_cols = [c for c in numeric_all if c in df.columns]

    # Convertir numéricas: coma->punto y quitar espacios
    for c in numeric_cols:
        df[c] = (
            df[c]
            .str.replace(",", ".", regex=False)
            .str.replace(" ", "", regex=False)
        )
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df, numeric_cols


def resumen_numericas_en_columnas(df_num: pd.DataFrame) -> pd.DataFrame:
    """
    Devuelve un DataFrame con métricas en filas y VARIABLES en COLUMNAS.
    Métricas: count, mean, std, min, p25, p50, p75, max, missing, missing_pct, zeros, zeros_pct, unique_vals
    """
    desc = df_num.describe(percentiles=[0.25, 0.5, 0.75])
    desc = desc.rename(index={"25%": "p25", "50%": "p50", "75%": "p75"})

    missing = df_num.isna().sum().to_frame().T; missing.index = ["missing"]
    missing_pct = (df_num.isna().mean() * 100).round(2).to_frame().T; missing_pct.index = ["missing_pct"]
    zeros = (df_num == 0).sum().to_frame().T; zeros.index = ["zeros"]
    zeros_pct = ((df_num == 0).mean() * 100).round(2).to_frame().T; zeros_pct.index = ["zeros_pct"]
    unique_vals = df_num.nunique().to_frame().T; unique_vals.index = ["unique_vals"]

    resumen = pd.concat([desc, missing, missing_pct, zeros, zeros_pct, unique_vals], axis=0)
    orden = ["count","mean","std","min","p25","p50","p75","max","missing","missing_pct","zeros","zeros_pct","unique_vals"]
    resumen = resumen.reindex(orden)
    return resumen


def eliminar_outliers_iqr(df: pd.DataFrame, numeric_cols: list[str], k: float = 1.5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Elimina filas con outliers por IQR en cada columna numérica.
    - Mantiene filas donde (valor entre [Q1 - k*IQR, Q3 + k*IQR]) o NaN para CADA columna.
    Retorna df_filtrado + tabla de límites (Q1, Q3, IQR, lower, upper) por columna.
    """
    if not numeric_cols:
        return df.copy(), pd.DataFrame()

    q1 = df[numeric_cols].quantile(0.25, interpolation="linear")
    q3 = df[numeric_cols].quantile(0.75, interpolation="linear")
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr

    limites = pd.DataFrame({"Q1": q1, "Q3": q3, "IQR": iqr, "lower": lower, "upper": upper})

    mask = pd.DataFrame(True, index=df.index, columns=numeric_cols)
    for c in numeric_cols:
        col = df[c]
        mask[c] = col.isna() | ((col >= lower[c]) & (col <= upper[c]))
    keep = mask.all(axis=1)
    return df.loc[keep].copy(), limites


# ---------- 1) BOXPLOTS ESTANDARIZADOS (z-score) ----------
def boxplots_zscore(df_num: pd.DataFrame, path_png: str, titulo: str):
    """Boxplots verticales tras estandarizar cada variable: z = (x - mean) / std."""
    if df_num.shape[1] == 0:
        return
    # Evita división por cero
    z = (df_num - df_num.mean()) / df_num.std(ddof=0).replace(0, np.nan)
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(max(10, df_num.shape[1]*0.6), 6))
    z.boxplot(column=z.columns.tolist(), vert=True, showmeans=True, meanline=True, ax=ax)
    ax.set_title(titulo + " (z-score)", fontsize=14, weight="bold")
    ax.set_ylabel("z")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(path_png, dpi=220)
    plt.close(fig)



# ====== EJECUCIÓN ========
# 1) Cargar y tipar
df, numeric_cols = cargar_y_tipear(FILE_NAME, SEP, NUMERIC_ALL)
if not numeric_cols:
    raise ValueError("No se encontraron columnas numéricas definidas en el archivo.")

# 2) Frames numéricos (para resúmenes y gráficos)
df_num = df[numeric_cols].copy()

# 3) Resumen ORIGINAL (variables en columnas)
resumen_original = resumen_numericas_en_columnas(df_num)

# 4) Eliminar outliers por IQR y rehacer resumen
df_clean, _ = eliminar_outliers_iqr(df, numeric_cols, k=IQR_K)
df_num_clean = df_clean[numeric_cols].copy()
resumen_sin_out = resumen_numericas_en_columnas(df_num_clean)

# 5) Exportar Excel (solo 2 hojas de numéricas)
with pd.ExcelWriter(OUTPUT_XLSX, engine="xlsxwriter") as writer:
    resumen_original.to_excel(writer, sheet_name="Numericas_Original")
    writer.sheets["Numericas_Original"].set_column(0, 0, 20)
    writer.sheets["Numericas_Original"].set_column(1, 200, 16)

    resumen_sin_out.to_excel(writer, sheet_name="Numericas_SinOutliers")
    writer.sheets["Numericas_SinOutliers"].set_column(0, 0, 20)
    writer.sheets["Numericas_SinOutliers"].set_column(1, 200, 16)

# 6) Exportar CSV con TODAS las columnas en el MISMO ORDEN que FULL_ORDER, pero SIN filas outliers
cols_present = [c for c in FULL_ORDER if c in df_clean.columns]  # por si faltara alguna en origen
df_clean[cols_present].to_csv(CSV_SIN_OUT, sep=SEP, index=False, encoding="utf-8-sig")

# 7) Gráficos (boxplots estéticos)
boxplots_zscore(df_num, "boxplots_original_z.png",
            "Boxplots variables numéricas (ORIGINAL)")
boxplots_zscore(df_num_clean, "boxplots_sin_outliers_z.png",
                "Boxplots variables numéricas (SIN outliers)")

# 8) Mensajes de estado
print("==== PROCESO COMPLETADO ====")
print(f"Excel: {Path(OUTPUT_XLSX).resolve()}")
print(f"CSV (todas las columnas, sin outliers): {Path(CSV_SIN_OUT).resolve()}")
print(f"PNG original: {Path(PNG_ORIGINAL).resolve()}")
print(f"PNG sin outliers: {Path(PNG_SIN_OUT).resolve()}")