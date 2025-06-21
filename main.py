import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def analyze_sales_data(file_path, santa_ana_sheet_name='Santa Ana'):
    """
    Realiza los análisis de ventas para los supermercados.

    Args:
        file_path (str): Ruta al archivo XLSX.
        santa_ana_sheet_name (str): Nombre de la hoja del supermercado 'Santa Ana'.
    """
    xls = pd.ExcelFile(file_path)
    sheet_names = xls.sheet_names

    # 1. Intervalos de Confianza Empíricos para 'Santa Ana'
    print("--- 1. Intervalos de Confianza Empíricos para 'Santa Ana' ---")
    if santa_ana_sheet_name in sheet_names:
        df_santa_ana = pd.read_excel(xls, sheet_name=santa_ana_sheet_name)
        df_santa_ana.columns = ['Fecha', 'Ventas']
        df_santa_ana['Fecha'] = pd.to_datetime(
            df_santa_ana['Fecha'])  # Convertir a datetime

        df_santa_ana['AñoMes'] = df_santa_ana['Fecha'].dt.to_period('M')

        monthly_sales = df_santa_ana.groupby('AñoMes')['Ventas'].agg(
            ['mean', 'std', 'count']).reset_index()
        monthly_sales.rename(columns={
                             'mean': 'Media_Ventas', 'std': 'Desviacion_Estandar', 'count': 'N_Observaciones'}, inplace=True)

        print("\nEstadísticas mensuales de ventas (agrupadas por mes):")
        print(monthly_sales)
        print("-" * 50)  # Separador para mejor legibilidad

        for index, row in monthly_sales.iterrows():
            mes_año = row['AñoMes']
            media_ventas = row['Media_Ventas']
            desviacion_estandar = row['Desviacion_Estandar']
            n_observaciones = row['N_Observaciones']

            if n_observaciones > 1 and not pd.isna(desviacion_estandar):
                se = desviacion_estandar / np.sqrt(n_observaciones)

                t_crit_95 = stats.t.ppf(0.975, n_observaciones - 1)
                lower_95 = media_ventas - t_crit_95 * se
                upper_95 = media_ventas + t_crit_95 * se

                t_crit_99 = stats.t.ppf(0.995, n_observaciones - 1)
                lower_99 = media_ventas - t_crit_99 * se
                upper_99 = media_ventas + t_crit_99 * se

                print(f"Mes: {mes_año}")
                print(f"  Ventas Promedio: ${media_ventas:,.2f}")
                print(f"  Nº de Días de Venta: {n_observaciones}")
                print(
                    f"  Desviación Estándar (Ventas Diarias): ${desviacion_estandar:,.2f}")
                print(f"  IC 95%: [${lower_95:,.2f}, ${upper_95:,.2f}]")
                print(f"  IC 99%: [${lower_99:,.2f}, ${upper_99:,.2f}]")
                print("-" * 30)
            else:
                print(f"Mes: {mes_año} - No hay suficientes datos ({n_observaciones} observaciones) o la desviación estándar es nula para calcular el IC. Se necesitan al menos 2 días de venta por mes.")
                print("-" * 30)
        print("\n")
    else:
        print(
            f"La hoja '{santa_ana_sheet_name}' no se encontró en el archivo.")
        print("\n")

    # 2. Pruebas ANOVA para determinar si las ventas esperadas de todas las tiendas son iguales
    print("--- 2. Pruebas ANOVA para Comparar Ventas de Todas las Tiendas ---")
    all_sales_data = []
    for sheet_name in sheet_names:
        df_temp = pd.read_excel(xls, sheet_name=sheet_name)
        df_temp.columns = ['Fecha', 'Ventas']
        df_temp['Supermercado'] = sheet_name
        all_sales_data.append(df_temp)

    df_all_sales = pd.concat(all_sales_data, ignore_index=True)

    if len(df_all_sales['Supermercado'].unique()) < 2:
        print("Se necesitan al menos dos supermercados para realizar ANOVA.")
    elif df_all_sales['Ventas'].isnull().any():
        print("Advertencia: Se encontraron valores nulos en las ventas. Eliminándolos para ANOVA.")
        df_all_sales.dropna(subset=['Ventas'], inplace=True)
        if df_all_sales.empty:
            print(
                "No quedan datos después de eliminar los nulos. No se puede realizar ANOVA.")
            return
    print(
        f"Datos combinados de ventas de {len(df_all_sales['Supermercado'].unique())} supermercados.")
    print("Primeras filas de datos combinados:")
    print(df_all_sales.head())
    # Realizar la prueba ANOVA
    model = ols('Ventas ~ C(Supermercado)', data=df_all_sales).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    print("Tabla ANOVA:")
    print(anova_table)

    alpha = 0.05
    if 'C(Supermercado)' in anova_table.index:
        p_value_anova = anova_table.loc['C(Supermercado)', 'PR(>F)']
    else:

        p_value_anova = anova_table['PR(>F)'][0]

    if p_value_anova < alpha:
        print(
            f"\nCon un nivel de significancia del {alpha*100}%, rechazamos la hipótesis nula.")
        print("Hay evidencia estadística de que las ventas esperadas de al menos un supermercado son diferentes.")
        print("Esto sugiere que no todas las tiendas se comportan de la misma manera en términos de ventas.")
    else:
        print(
            f"\nCon un nivel de significancia del {alpha*100}%, no rechazamos la hipótesis nula.")
        print("No hay evidencia estadística de que las ventas esperadas de los supermercados sean diferentes.")
        print("Las ventas promedio de los supermercados son estadísticamente similares.")
    print("\n")

    # 3. Identificar la tienda con mayor y menor promedio de ventas y prueba de hipótesis
    print("--- 3. Comparación de la Tienda con Mayor y Menor Promedio de Ventas ---")

    if not df_all_sales.empty:
        avg_sales_by_supermarket = df_all_sales.groupby(
            'Supermercado')['Ventas'].mean().sort_values(ascending=False)
        print("Promedio de ventas por supermercado:")
        print(avg_sales_by_supermarket)

        supermarket_highest_sales = avg_sales_by_supermarket.index[0]
        supermarket_lowest_sales = avg_sales_by_supermarket.index[-1]

        print(
            f"\nTienda con mayor promedio de ventas: '{supermarket_highest_sales}' (${avg_sales_by_supermarket.iloc[0]:,.2f})")
        print(
            f"Tienda con menor promedio de ventas: '{supermarket_lowest_sales}' (${avg_sales_by_supermarket.iloc[-1]:,.2f})")

        if p_value_anova < alpha:
            print(
                "\nRealizando prueba de Tukey HSD (Post-hoc ANOVA) para comparaciones pareadas:")
            tukey_result = pairwise_tukeyhsd(endog=df_all_sales['Ventas'],
                                             groups=df_all_sales['Supermercado'],
                                             alpha=0.05)
            print(tukey_result)

            tukey_df = pd.DataFrame(data=tukey_result._results_table.data[1:],
                                    columns=tukey_result._results_table.data[0])

            found_comparison = False
            specific_comparison = tukey_df[
                ((tukey_df['group1'] == supermarket_highest_sales) & (tukey_df['group2'] == supermarket_lowest_sales)) |
                ((tukey_df['group1'] == supermarket_lowest_sales) &
                 (tukey_df['group2'] == supermarket_highest_sales))
            ]

            if not specific_comparison.empty:
                p_value_tukey = specific_comparison['p-adj'].iloc[0]
                mean_diff_tukey = specific_comparison['meandiff'].iloc[0]

                print(
                    f"\nComparación específica entre '{supermarket_highest_sales}' y '{supermarket_lowest_sales}':")
                if specific_comparison['group1'].iloc[0] == supermarket_lowest_sales:
                    mean_diff_tukey *= -1
                print(f"  Diferencia de medias: ${mean_diff_tukey:,.2f}")
                print(f"  P-value (Tukey HSD): {p_value_tukey:.4f}")
                found_comparison = True

                if p_value_tukey < alpha:
                    print(
                        "La diferencia entre las ventas de la tienda con mayor y menor promedio es estadísticamente significativa.")
                    print(
                        f"Esto sugiere que '{supermarket_lowest_sales}' realmente tiene ventas peores y podría necesitar más atención.")
                else:
                    print(
                        "La diferencia entre las ventas de la tienda con mayor y menor promedio NO es estadísticamente significativa.")
                    print("Las diferencias observadas podrían ser debidas al azar.")

            if not found_comparison:
                print(
                    f"No se encontró una comparación directa entre '{supermarket_highest_sales}' y '{supermarket_lowest_sales}' en los resultados de Tukey HSD.")
                print(
                    "Esto es inusual si ambas tiendas están en el dataset. Verifique los nombres.")
        else:
            print("\nEl ANOVA no encontró diferencias significativas, por lo que una prueba post-hoc directa entre las tiendas con mayor y menor promedio no es estadísticamente necesaria para confirmar una diferencia global. Las diferencias observadas son probablemente debidas al azar.")
    else:
        print("No hay datos disponibles para realizar la comparación de tiendas.")
    print("\n")


if __name__ == "__main__":
    excel_file_path = "Datos_examen_final_21Co20258_a2119.xlsx"
    analyze_sales_data(excel_file_path)
