import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns


def analisis_anova_completo(ruta_archivo_excel):
    """
    Realiza un análisis ANOVA de un factor completo, incluyendo verificación de
    requisitos, generación de gráficos y pruebas post-hoc, siguiendo la
    estructura del documento de ejemplo.
    """
    try:
        # --- 0. CARGA Y PREPARACIÓN DE DATOS ---
        print("--- 0. CARGANDO Y PREPARANDO DATOS ---")
        xls = pd.ExcelFile(ruta_archivo_excel)
        all_sales_data = []
        for sheet_name in xls.sheet_names:
            df_temp = pd.read_excel(xls, sheet_name=sheet_name)
            df_temp = df_temp.iloc[:, [0, 1]]
            df_temp.columns = ['Fecha', 'Ventas']
            df_temp['Supermercado'] = sheet_name
            all_sales_data.append(df_temp)
        df_all_sales = pd.concat(
            all_sales_data, ignore_index=True).dropna(subset=['Ventas'])
        print("Datos cargados correctamente.\n")

    except FileNotFoundError:
        print(
            f"ERROR: No se encontró el archivo en la ruta '{ruta_archivo_excel}'.")
        print(
            "Por favor, asegúrate de que el archivo ha sido subido y la ruta es correcta.")
        return
    except Exception as e:
        print(f"Ocurrió un error inesperado al cargar los datos: {e}")
        return

    # --- 1. PLANTEAMIENTO DEL CONTRASTE ---
    print("--- 1. PLANTEAMIENTO DEL ANÁLISIS ANOVA ---")
    print("H_0: Las medias de ventas de todos los supermercados son iguales.")
    print("H_1: Al menos una media de ventas es diferente.")
    alpha = 0.05
    print(f"Nivel de significancia alpha = {alpha}\n")

    # --- 2. VERIFICACIÓN DE REQUISITOS ---
    print("--- 2. VERIFICACIÓN DE REQUISITOS DEL ANOVA ---")
    print("a) Independencia: Se asume que las observaciones son independientes por el diseño del estudio.")

    print("\nb) Prueba de Homogeneidad de Varianzas (Levene):")
    samples = [group["Ventas"].values for name,
               group in df_all_sales.groupby("Supermercado")]
    levene_stat, levene_p = stats.levene(*samples)
    print(f"   p-valor = {levene_p:.4f}")
    if levene_p > alpha:
        print("   Conclusión: Se cumple el requisito de homogeneidad de varianzas. ")
    else:
        print("   Conclusión: NO se cumple el requisito de homogeneidad de varianzas. ")

    print("\nc) Prueba de Normalidad de los Residuos (Shapiro-Wilk):")
    model = ols('Ventas ~ C(Supermercado)', data=df_all_sales).fit()
    shapiro_stat, shapiro_p = stats.shapiro(model.resid)
    print(f"   p-valor = {shapiro_p:.4f}")
    if shapiro_p > alpha:
        print("   Conclusión: Se cumple el requisito de normalidad. ")
    else:
        print("   Conclusión: NO se cumple el requisito de normalidad. ")

    # --- 3. ANÁLISIS GRÁFICO DE LAS DISTRIBUCIONES ---
    print("\n--- 3. GENERANDO GRÁFICOS DE LAS DISTRIBUCIONES ---")

    # Gráfico de Densidad
    plt.figure(figsize=(12, 7))
    sns.kdeplot(data=df_all_sales, x='Ventas', hue='Supermercado',
                fill=True, common_norm=False, alpha=0.3)
    plt.title('Distribución de Ventas por Supermercado (Gráfico de Densidad)')
    plt.savefig('grafico_densidad_ventas.png')
    plt.close()

    # Gráfico de Cajas (Box Plot)
    plt.figure(figsize=(12, 7))
    sns.boxplot(data=df_all_sales, x='Supermercado', y='Ventas')
    plt.title('Comparación de Ventas por Supermercado (Gráfico de Cajas)')
    plt.savefig('boxplot_ventas.png')
    plt.close()

    print("Gráficos 'grafico_densidad_ventas.png' y 'boxplot_ventas.png' guardados.")
    print("Estos gráficos ayudan a visualizar las diferencias y la forma de los datos,")
    print("especialmente útil si no se cumplió el supuesto de normalidad.\n")

    # --- 4. DESARROLLO DEL CONTRASTE ANOVA ---
    print("--- 4. TABLA ANOVA Y RESULTADO DEL CONTRASTE ---")
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)

    p_value_anova = anova_table.loc['C(Supermercado)', 'PR(>F)']

    print("\nConclusión del Contraste ANOVA:")
    if p_value_anova < alpha:
        print(
            f"   El p-valor ({p_value_anova:.4e}) es menor que alpha. SE RECHAZA H_0. ")
        print("   Existen diferencias estadísticamente significativas en las ventas medias de al menos un supermercado. ")
        run_post_hoc = True
    else:
        print(
            f"   El p-valor ({p_value_anova:.4e}) es mayor que alpha. NO se rechaza H_0.")
        print("   No hay evidencia de diferencias significativas en las ventas medias.")
        run_post_hoc = False

    # --- 5. CONTRASTE A POSTERIORI ---
    if run_post_hoc:
        print("\n--- 5. CONTRASTE A POSTERIORI (TUKEY HSD) ---")
        print("Buscando qué grupos específicos son diferentes entre sí... ")
        tukey_result = pairwise_tukeyhsd(endog=df_all_sales['Ventas'],
                                         groups=df_all_sales['Supermercado'],
                                         alpha=alpha)
        print(tukey_result)
        print("\n--- 6. CONCLUSIÓN FINAL ---")
        print("La tabla de Tukey HSD muestra los pares de supermercados con diferencias significativas ('reject'=True). ")
        print("Esto permite identificar qué tiendas tienen un rendimiento de ventas significativamente distinto a otras. ")

    else:
        print("\n--- 5 y 6. NO SE REQUIERE ANÁLISIS POST-HOC ---")


# --- EJECUCIÓN DEL SCRIPT ---
# Para ejecutarlo, primero sube tu archivo y luego reemplaza el texto de abajo
# con la ruta correcta del archivo Excel.
ruta_del_archivo = "Datos_examen_final_21Co20258_a2119.xlsx"
analisis_anova_completo(ruta_del_archivo)
