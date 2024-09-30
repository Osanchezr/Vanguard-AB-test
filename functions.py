

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest
import matplotlib.pyplot as plt
import seaborn as sns

#FUNCIONES EXPLORACIÓN DE DATOS

def plot_histograms(df):
    """
    Genera histogramas para cada columna numérica en el DataFrame proporcionado.
    
    Parameters:
        df (DataFrame): DataFrame que contiene las columnas numéricas.
    """
    df.hist(figsize=(9, 7), bins=60, xlabelsize=8, ylabelsize=8)
    plt.suptitle('Histogramas de Variables Numéricas')
    plt.show()

def plot_boxplots(df):
    """
    Genera boxplots para cada columna numérica en el DataFrame proporcionado.
    
    Parameters:
        df (DataFrame): DataFrame que contiene las columnas numéricas.
    """
    # Verificar que haya columnas en el DataFrame
    if df.empty:
        print("El DataFrame está vacío.")
        return

    # Configurar el tamaño de la figura
    num_columns = df.shape[1]
    num_rows = (num_columns // 3) + (num_columns % 3 > 0)  # Calcular número de filas necesarias
    plt.figure(figsize=(10, num_rows * 4))  # Ajustar el tamaño en función del número de filas

    # Iterar sobre cada columna y generar un boxplot
    for i, column in enumerate(df.columns):
        plt.subplot(num_rows, 3, i + 1)  # Cambia el tamaño de la cuadrícula según el número de columnas
        plt.boxplot(df[column].dropna())  # Elimina valores NaN antes de graficar
        plt.title(column)
        plt.grid(True)

    plt.tight_layout()  # Ajustar el espaciado entre los gráficos
    plt.suptitle('Boxplots de Variables Numéricas', y=1.02)  # Ajustar el título para que no se superponga
    plt.show()

def plot_correlation_matrix(df):
    """
    Genera un mapa de calor de la matriz de correlación para el DataFrame proporcionado.
    
    Parameters:
    df (DataFrame): DataFrame que contiene las columnas numéricas.
    """
    # Calcular la matriz de correlación
    correlation_matrix = df.corr()

    # Configurar el tamaño de la figura
    plt.figure(figsize=(8, 6))

    # Crear un mapa de calor (heatmap) de la matriz de correlación
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, 
                cbar_kws={"shrink": .8}, linewidths=0.5)

    # Añadir título y etiquetas
    plt.title('Mapa de Correlación')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()  # Ajustar el espaciado
    plt.show()


#FUNCIONES DE LIMPIEZA DE DATOS

def clean_demo_data(df):
    """
    Realiza la limpieza y transformación del dataset demográfico.

    Operaciones:
    ------------
    1. Elimina duplicados y valores nulos.
    2. Convierte columnas numéricas a enteros ('clnt_tenure_yr', 'clnt_age', 'num_accts', 'logons_6_mnth', 'calls_6_mnth').Reemplaza valores en 'gendr' (de 'X' a 'U').
    3. Filtra registros: elimina menores de 18 años y clientes con más de 40 años de permanencia.
    4. Agrupa 'bal' en categorías ('bajo', 'medio', 'alto').
    5. Elimina filas donde los años de permanencia superan la edad del cliente.
    6. Crea la columna 'account_since_child' para indicar si la cuenta fue abierta antes de los 18 años.
    7. Retiene solo las columnas relevantes para el análisis.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos demográficos originales.

    Returns:
    --------
    pd.DataFrame
        DataFrame limpio y transformado con las columnas relevantes.
    """
    # 1
    df = df.drop_duplicates()
    df = df.dropna()

    # 2
    df[["clnt_tenure_yr","clnt_age","num_accts","logons_6_mnth","calls_6_mnth"]] = df[["clnt_tenure_yr","clnt_age","num_accts","logons_6_mnth","calls_6_mnth"]].astype(int)
    df["gendr"] = df["gendr"].replace({"X": "U"})

    # 3
    df = df[(df["clnt_age"] >= 18) & (df["clnt_tenure_yr"] < 40)]

    # 4
    df['bal_category'] = pd.qcut(df['bal'], q=[0, 0.25, 0.75, 1], labels=['bajo', 'medio', 'alto'])

    # 5
    indices_a_eliminar = df[df["clnt_tenure_yr"] > df["clnt_age"]].index
    df = df.drop(indices_a_eliminar)

    # 6
    df['account_since_child'] = np.where((df["clnt_age"] - df["clnt_tenure_yr"]) < 18, 'yes', 'no')

    # 7
    df = df[["client_id", "clnt_tenure_yr", "clnt_age", "gendr", "num_accts", "bal", "calls_6_mnth", "logons_6_mnth", "bal_category", "account_since_child"]]

    return df


def clean_experiment_clients_data(df):
    """
    Limpieza de datos para el dataset de clientes experimentales.

    Operaciones realizadas:
    -----------------------
    1. Eliminar valores nulos:
       - Se eliminan las filas que contienen valores nulos en cualquier columna del DataFrame.
       - La accion corresponde a trabajar solo con clientes que pertenecieron a un grupo control o test.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos de clientes experimentales.

    Returns:
    --------
    pd.DataFrame
        DataFrame limpio sin valores nulos.
    """
    # 1
    df = df.dropna()

    return df


def clean_web_data(df1, df2):
    """
    Fusiona dos DataFrames de datos web y realiza varias transformaciones.

    Operaciones realizadas:
    -----------------------
    1. Fusión de los DataFrames.
    2. Conversión de la columna 'date_time' a formato datetime.
    3. Ordenación por 'client_id', 'visit_id' y 'date_time'.
    4. Creación de la columna 'is_complete' para identificar procesos completos.
    5. Cálculo del tiempo entre pasos ('time_diff_step').
    6. Ordenación lógica de los pasos del proceso.
    7. Identificación del paso anterior y detección de errores en el proceso.

    Parameters:
    -----------
    df1, df2 : pd.DataFrame
        DataFrames que contienen los datos web.

    Returns:
    --------
    pd.DataFrame
        DataFrame fusionado y transformado.
    """
    # 1
    merged_df = pd.concat([df1, df2], ignore_index=True)

    # 2
    merged_df["date_time"] = pd.to_datetime(merged_df["date_time"])

    # 3
    merged_df = merged_df.sort_values(by=['client_id', 'visit_id', 'date_time'])

    # 4
    final_steps = ['confirm']
    merged_df['is_complete'] = merged_df.groupby('visit_id')['process_step'].transform(lambda x: 'Complete' if final_steps[0] in x.values else 'Incomplete')
    merged_df['is_complete'] = merged_df['is_complete'].replace({'Complete': 1, 'Incomplete': 0})

    # 5
    merged_df['time_before_step'] = merged_df.groupby(['client_id', 'visitor_id', 'visit_id'])['date_time'].shift(1)
    merged_df['time_diff_step'] = (merged_df['date_time'] - merged_df['time_before_step']).dt.total_seconds()

    # 6
    step_order = ['start', 'step_1', 'step_2', 'step_3', 'confirm']
    merged_df['process_step'] = pd.Categorical(merged_df['process_step'], categories=step_order, ordered=True)
    merged_df['prev_process_step'] = merged_df.groupby(['client_id', 'visit_id'])['process_step'].shift(1)

    # 7
    merged_df['is_error'] = merged_df['process_step'] < merged_df['prev_process_step']
    merged_df['is_error'] = merged_df['is_error'].fillna(False)


    return merged_df


def merge_datasets(web_data, experiment_data, demo_data):
    """
    Une múltiples datasets en un solo DataFrame.

    Acciones:
    ---------
    1. Fusiona los dataframes `web_data` y `experiment_data` por 'client_id'.
    2. Fusiona el resultado con `demo_data` también por 'client_id'.
 
    Parameters:
    -----------
    web_data : pd.DataFrame
        DataFrame con datos web.
    experiment_data : pd.DataFrame
        DataFrame con datos de experimentación.
    demo_data : pd.DataFrame
        DataFrame con datos demográficos.

    Returns:
    --------
    pd.DataFrame
        DataFrame final combinado y transformado.
    """
    # Unir los dataframes por client_id
    df_merged = pd.merge(web_data, experiment_data, on="client_id", how="inner")
    df_final = pd.merge(df_merged, demo_data, on="client_id", how="inner")

    return df_final


#FUNCIONES DE OPERACIONES CON DATOS

#KPIS

def calculate_unique_counts(df, completion_status):
    """
    Calcula la cantidad de clientes únicos que completaron o no completaron el proceso.
    
    Parameters:
        df (DataFrame): DataFrame que contiene la información de los clientes.
        completion_status (int): Estado de completación (1 para completado, 0 para incompleto).
        
    Returns:
        int: Cantidad de clientes únicos en el estado especificado.
    """
    return df[df['is_complete'] == completion_status]['client_id'].nunique()


def calculate_completion_stats(df):
    """
    Calcula la cantidad de clientes únicos que completaron o no completaron el proceso,
    y devuelve las tasas de finalización para el grupo general, de control y de prueba en un DataFrame.
    
    Parameters:
        df (DataFrame): DataFrame que contiene la información de los clientes.
        
    Returns:
        DataFrame: Un DataFrame que contiene las tasas de finalización para el grupo general, 
                    de control y de prueba.
    """
    # Lista para almacenar resultados
    results = []

    for variation in ['general', 'Control', 'Test']:
        # Filtrar el DataFrame según la variación
        filtered_df = df if variation == 'general' else df[df["Variation"] == variation]
        
        # Calcular la cantidad de clientes únicos
        completed = filtered_df[filtered_df['is_complete'] == 1]['client_id'].nunique()
        incomplete = filtered_df[filtered_df['is_complete'] == 0]['client_id'].nunique()
        total = completed + incomplete
        
        # Calcular tasas de finalización
        completion_rate = (completed / total) * 100 if total > 0 else 0
        incompletion_rate = (incomplete / total) * 100 if total > 0 else 0
        
        # Añadir resultados a la lista
        results.append({
            'Variation': variation,
            'Completed Clients': completed,
            'Incomplete Clients': incomplete,
            'Completion Rate (%)': completion_rate,
            'Incompletion Rate (%)': incompletion_rate
        })
        
    # Crear un DataFrame a partir de los resultados
    results_df = pd.DataFrame(results)
    
    return results_df


def calculate_time_stats(df):
    """
    Calcula estadísticas de tiempo por paso de proceso y por grupo (Control y Test).
    
    Parameters:
        df (DataFrame): DataFrame que contiene la información del tiempo por paso.
        
    Returns:
        DataFrame: Estadísticas de tiempo para cada paso, desglosadas por grupo.
    """
    # Agrupar por 'Variation' y 'process_step' y calcular estadísticas de tiempo
    time_stats_df = df.groupby(['Variation', 'process_step'])['time_diff_step'].agg(['mean', 'min', 'max', 'count']).rename(
        columns={'mean': 'mean_time', 'min': 'min_time', 'max': 'max_time', 'count': 'num_entries'}
    ).reset_index()
    
    return time_stats_df

def calculate_error_rate(df):
    """
    Calcula la tasa de errores en el DataFrame proporcionado para los grupos Control y Test.
    
    Parameters:
        df (DataFrame): DataFrame que contiene la información sobre errores y el grupo (Control o Test).
        
    Returns:
        DataFrame: Un DataFrame con la tasa de errores y el total de registros y errores para ambos grupos.
    """
    # Función interna para calcular la tasa de errores
    def compute_error_stats(group_df):
        total_records = len(group_df)
        total_errors = group_df['is_error'].sum()
        error_rate = (total_errors / total_records) * 100 if total_records > 0 else 0
        return total_records, total_errors, round(error_rate, 2)

    # Calcular estadísticas para el grupo Control
    control_df = df[df['Variation'] == 'Control']
    control_stats = compute_error_stats(control_df)
    
    # Calcular estadísticas para el grupo Test
    test_df = df[df['Variation'] == 'Test']
    test_stats = compute_error_stats(test_df)

    # Crear un DataFrame para devolver los resultados
    result_df = pd.DataFrame({
        'Group': ['Control', 'Test'],
        'Total Records': [control_stats[0], test_stats[0]],
        'Total Errors': [control_stats[1], test_stats[1]],
        'Error Rate (%)': [control_stats[2], test_stats[2]]
    })

    return result_df

##PRUEBA DE HIPOTESIS

def hypothesis_testing_completion(df, alpha=0.05):
    """
    Realiza una prueba de hipótesis sobre las tasas de finalización entre grupos de prueba y control.
    
    Parameters:
        df (DataFrame): DataFrame que contiene información de ambos grupos, incluyendo la columna 'Variation'.
        alpha (float): Nivel de significancia para la prueba de hipótesis (por defecto 0.05).
        
    Returns:
        dict: Resultados de la prueba, incluyendo estadística z, p-value y decisión sobre la hipótesis nula.
    """
    # Filtrar los grupos de prueba y control
    df_test = df[df['Variation'] == 'Test']
    df_control = df[df['Variation'] == 'Control']
    
    # Calcular completaciones
    test_completions = calculate_unique_counts(df_test, 1)
    control_completions = calculate_unique_counts(df_control, 1)

  # Contar el número de clientes únicos en cada grupo
    unique_test_clients = df_test['client_id'].nunique()
    unique_control_clients = df_control['client_id'].nunique()

    # Preparar datos para la prueba de hipótesis
    counts = np.array([test_completions, control_completions])
    nobs = np.array([unique_test_clients, unique_control_clients]) 

    # Realizar la prueba de hipótesis
    stat, p_value = proportions_ztest(counts, nobs, alternative='larger')

    # Decidir sobre la hipótesis nula
    if p_value < alpha:
        hypothesis_decision = "Rechaza la hipótesis nula: hay una diferencia significativa en las tasas de finalización."
    else:
        hypothesis_decision = "No se rechaza la hipótesis nula: no hay una diferencia significativa en las tasas de finalización."

    # Devolver resultados
    return {
        'z_statistic': stat,
        'p_value': p_value,
        'decision': hypothesis_decision
    }


def hypothesis_test_error_rate(df, alpha=0.05):
    """
    Realiza una prueba de hipótesis para comparar las tasas de error entre los grupos Control y Test.
    
    Parameters:
        df (DataFrame): DataFrame que contiene la información sobre errores y el grupo (Control o Test).
        alpha (float): Nivel de significancia (por defecto es 0.05).
        
    Returns:
        dict: Un diccionario con los resultados de la prueba de hipótesis, incluyendo la decisión
              de rechazar o no la hipótesis nula.
    """
    # Función interna para calcular la tasa de errores
    def compute_error_stats(group_df):
        total_records = len(group_df)
        total_errors = group_df['is_error'].sum()
        error_rate = total_errors / total_records if total_records > 0 else 0
        return total_records, total_errors, error_rate

    # Calcular estadísticas para el grupo Control
    control_df = df[df['Variation'] == 'Control']
    control_stats = compute_error_stats(control_df)
    
    # Calcular estadísticas para el grupo Test
    test_df = df[df['Variation'] == 'Test']
    test_stats = compute_error_stats(test_df)

    # Extraer valores de cada grupo
    n_control, errors_control, p_control = control_stats
    n_test, errors_test, p_test = test_stats

    # Proporción combinada
    p_combined = (errors_control + errors_test) / (n_control + n_test)

    # Error estándar conjunto
    se_combined = np.sqrt(p_combined * (1 - p_combined) * (1/n_control + 1/n_test))

    # Cálculo del estadístico Z
    z_stat = (p_control - p_test) / se_combined

    # Cálculo del p-valor (prueba bilateral)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    # Decisión sobre la hipótesis nula
    reject_null = p_value < alpha
    decision = "Se rechaza H₀" if reject_null else "No se rechaza H₀"

    # Explicación de la decisión
    explanation = ("Las diferencias en las tasas de error son estadísticamente significativas" 
                   if reject_null else 
                   "No hay evidencia suficiente para decir que las tasas de error son significativamente diferentes")

    # Resultados
    results = {
        'Control Error Rate (%)': round(p_control * 100, 2),
        'Test Error Rate (%)': round(p_test * 100, 2),
        'Z-Statistic': round(z_stat, 4),
        'P-Value': round(p_value, 4),
        'Alpha': alpha,
        'Decision': decision,
        'Explanation': explanation
    }
    
    return results

def hypothesis_test_time_stats(df, alpha=0.05):
    """
    Realiza una prueba de hipótesis para comparar las medias de duración por paso entre los grupos Control y Test.
    
    Parameters:
        df (DataFrame): DataFrame que contiene la información del tiempo por paso ('time_diff_step')
                        y el grupo ('Variation').
        alpha (float): Nivel de significancia (por defecto es 0.05).
        
    Returns:
        DataFrame: Un DataFrame con los resultados de la prueba t para cada paso.
    """
    # Lista para almacenar los resultados de cada paso
    results = []
    
    # Obtener todos los pasos únicos del proceso
    process_steps = df['process_step'].unique()
    
    # Iterar sobre cada paso
    for step in process_steps:
        # Separar los datos por grupo y paso del proceso
        control_times = df[(df['Variation'] == 'Control') & (df['process_step'] == step)]['time_diff_step']
        test_times = df[(df['Variation'] == 'Test') & (df['process_step'] == step)]['time_diff_step']
        
        # Prueba t para dos muestras independientes
        t_stat, p_value = stats.ttest_ind(control_times, test_times, equal_var=False, nan_policy='omit')
        
        # Decisión sobre la hipótesis nula
        reject_null = p_value < alpha
        decision = "Se rechaza H₀" if reject_null else "No se rechaza H₀"
        
        # Agregar los resultados a la lista
        results.append({
            'process_step': step,
            'Control Mean Time': round(control_times.mean(), 2),
            'Test Mean Time': round(test_times.mean(), 2),
            'T-Statistic': round(t_stat, 4),
            'P-Value': round(p_value, 4),
            'Alpha': alpha,
            'Decision': decision
        })
    
    # Convertir la lista de resultados a un DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df

def hypothesis_testing_with_threshold(df, cost_effectiveness_threshold=0.05, alpha=0.05):
    """
    Realiza una prueba de hipótesis sobre las tasas de finalización entre grupos de prueba y control,
    considerando un umbral de mejora costo-efectiva.
    
    Parameters:
        df (DataFrame): DataFrame que contiene información de ambos grupos, incluyendo la columna 'Variation'.
        cost_effectiveness_threshold (float): Umbral de mejora costo-efectiva (por defecto 0.05).
        alpha (float): Nivel de significancia para la prueba de hipótesis (por defecto 0.05).
        
    Returns:
        dict: Resultados de la prueba, incluyendo estadística z, p-value y decisión sobre la hipótesis nula.
    """
    # Filtrar los grupos de prueba y control
    df_test = df[df['Variation'] == 'Test']
    df_control = df[df['Variation'] == 'Control']

    # Calcular completaciones únicas
    test_completions = calculate_unique_counts(df_test, 1)
    control_completions = calculate_unique_counts(df_control, 1)

    # Contar el número total de clientes únicos en cada grupo
    test_total = df_test['client_id'].nunique()
    control_total = df_control['client_id'].nunique()

    # Ajustar la tasa de control añadiendo el umbral
    adjusted_control_rate = (control_completions / control_total) + cost_effectiveness_threshold

    # Realizar Z-test para comprobar si la tasa de Test es mayor que la tasa de Control ajustada
    stat, p_value = proportions_ztest([test_completions, control_completions],
                                      [test_total, control_total],
                                      value=adjusted_control_rate,
                                      alternative='larger')

    # Decisión basada en el p-value
    if p_value < alpha:
        hypothesis_decision = "Rechaza la hipótesis nula: el aumento en la tasa de completación es significativo y cumple con el umbral del 5%."
    else:
        hypothesis_decision = "No se rechaza la hipótesis nula: el aumento en la tasa de completación no cumple con el umbral del 5%."

    # Devolver resultados
    return {
        'z_statistic': stat,
        'p_value': p_value,
        'decision': hypothesis_decision
    }

def hypothesis_testing(df_test, df_control, column, alpha=0.05):
    """
    Realiza una prueba t de dos muestras sobre una columna específica entre grupos de prueba y control.
    
    Parameters:
        df_test (DataFrame): DataFrame del grupo de prueba.
        df_control (DataFrame): DataFrame del grupo de control.
        column (str): Nombre de la columna a comparar.
        alpha (float): Nivel de significancia para la prueba de hipótesis (default es 0.05).
        
    Returns:
        dict: Resultados de la prueba, incluyendo estadística t, p-value y decisión sobre la hipótesis nula.
    """
    # Tomar muestras aleatorias de ambos grupos
    control_sample = df_control[column].sample(n=1000, random_state=42)
    test_sample = df_test[column].sample(n=1000, random_state=42)
    
    # Realizar la prueba t
    stat, p_value = stats.ttest_ind(control_sample, test_sample, alternative="two-sided")
    
    # Decidir sobre la hipótesis nula
    if p_value < alpha:
        hypothesis_decision = "Rechaza la hipótesis nula: hay una diferencia significativa entre los grupos."
    else:
        hypothesis_decision = "No se rechaza la hipótesis nula: no hay una diferencia significativa entre los grupos."

    # Devolver resultados
    return {
        't_statistic': stat,
        'p_value': p_value,
        'decision': hypothesis_decision
    }

