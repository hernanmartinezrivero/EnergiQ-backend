# Revision 03 - Removed stray indented lines after commented section
# Revision 02 - Fixed duplicate create_sequences definition causing IndentationError
from IPython.display import display
from datetime import datetime
from datetime import timedelta
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import gdown
import ipywidgets as widgets
import joblib  # For caching results (optional)
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import tensorflow as tf


# === FUNCIONES DEFINIDAS ===


def find_outliers(df, limits):
    df.columns = df.columns.astype(str).str.strip().str.replace('.0', '', regex=False)
    outliers = pd.DataFrame(columns=df.columns)
    for column, (lower, upper) in limits.items():
        mask = (df[column] < lower) | (df[column] > upper)
        outliers = pd.concat([outliers, df[mask]], axis=0)
    return outliers.drop_duplicates()


def adf_test(series):
    result = adfuller(series)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    for key, value in result[4].items():
        print(f'Critical Value {key}: {value}')


def create_sequences(data, target_col, time_steps=1):
    X, y = [], []
    data_array = data.values  # Convertir una sola vez
    target_index = data.columns.get_loc(target_col)

    for i in range(len(data) - time_steps):
        X.append(data_array[i:(i + time_steps)])
        y.append(data_array[i + time_steps, target_index])

    return np.array(X), np.array(y)

# === BLOQUE PRINCIPAL ===

# Definir rutas
ruta_entrada = sys.argv[1] if len(sys.argv) > 1 else "archivo.csv"
#hay que poner las columnas timestamp1 y hora adelante

ruta_guardado = r'D:\HERNAN\Hernan\educacion\UNC\FAMAF\100-EnergIQ\df_limpio.csv'

# Leer archivo como texto para preprocesar encabezado
with open(ruta_entrada, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Limpiar encabezado duplicado
headers = lines[0].strip().split(',')
# Eliminar duplicados desde la segunda aparición de '3'
if headers.count('3') > 1:
    idx = headers.index('3', headers.index('3') + 1)
    headers = headers[:idx]

# Cambiar segunda aparición de "2" a "hora"
indices_2 = [i for i, x in enumerate(headers) if x == '2']
if len(indices_2) > 1:
    headers[indices_2[1]] = 'hora'

# Cargar datos desde la segunda línea
import io
data_str = ''.join(lines[1:])
df = pd.read_csv(io.StringIO(data_str), sep=',', header=None)
df.columns = headers

# Eliminar filas con valores nulos
df = df.dropna()

# Eliminar filas que contengan 'error' en cualquier columna
mask = df.astype(str).apply(lambda x: x.str.contains('error', case=False, na=False)).any(axis=1)
df = df[~mask]

# Mostrar primeras filas para verificar
print(df.head())

# Guardar el DataFrame limpio
df.to_csv(ruta_guardado, sep=';', index=False)


print(df.columns.tolist())

# Intentar convertir todas las columnas a numéricas (las que se pueda)
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='ignore')

# Ahora seleccionar solo columnas numéricas
desc = df.select_dtypes(include=['number']).describe()

# Formatear a 5 decimales
desc_formateado = desc.applymap(lambda x: f"{x:.5f}")

# Mostrar
print(desc_formateado)

#missing_values = df_trabajo.isnull().sum()
missing_values = df.isnull().sum()
print(missing_values)

msno.bar(df, figsize=(10, 8), fontsize=7, color='violet')

limites = {
    '1': {'Q1': 200, 'Q3': 245},
    '2': {'Q1': 200, 'Q3': 245},
    '3': {'Q1': 200, 'Q3': 245},
    '14': {'Q1': 0, 'Q3': 70},
    '15': {'Q1': 0, 'Q3': 70},
    '16': {'Q1': 0, 'Q3': 70},
    '26': {'Q1': 0, 'Q3': 50},
    '27': {'Q1': 0, 'Q3': 50},
    '28': {'Q1': 0, 'Q3': 50},
    '29': {'Q1': 0, 'Q3': 50},
    '30': {'Q1': 0, 'Q3': 100}
}

# aplicar los límites usando los valores de Q1 y Q3 proporcionados manualmente
limits = {}
for column, q_values in limites.items():
    Q1 = q_values['Q1']
    Q3 = q_values['Q3']

    lower_limit = Q1
    upper_limit = Q3
    limits[column] = (lower_limit, upper_limit)

# Imprimir los límites para verificación
for column, (lower, upper) in limits.items():
    print(f"{column}: Lower Limit = {lower}, Upper Limit = {upper}")


# Función para filtrar y encontrar outliers
# Encontrar los outliers en el DataFrame
outliers = find_outliers(df, limits)

# Imprimir las filas que tienen outliers
#print(outliers)

outliers.info()


# Verificamos el número de outliers
print(f"Number of outliers: {len(outliers)}")

#indice = 50
#print(df.loc[indice])

# Obtener los índices de los outliers
outliers_indices = outliers.index

# Eliminamos los outliers del dataframe original
df_trabajo = df.drop(outliers_indices)

# Verificar que los outliers hayan sido eliminados
#print(df_trabajo2.info())

pd.set_option('display.max_columns', None)  # Muestra todas las columnas


# Conversión de columnas de fecha y hora
df_trabajo['fecha_str'] = df_trabajo['timestamp1'].astype(str)
df_trabajo['hora_str'] = df_trabajo['hora'].astype(str)
df_trabajo['timestamp_full'] = pd.to_datetime(df_trabajo['fecha_str'] + ' ' + df_trabajo['hora_str'], errors='coerce')

# Filtrar fechas válidas
df_valid = df_trabajo[df_trabajo['timestamp_full'].notna()]
df_valid = df_valid.sort_values('timestamp_full')

# 📅 Selección del período de tiempo
fecha_inicio = pd.to_datetime('2025-04-13')
fecha_fin = pd.to_datetime('2025-05-10')  # inclusive hasta antes de medianoche del día siguiente

# Filtrar rango de fechas
df_filtrado = df_valid[(df_valid['timestamp_full'] >= fecha_inicio) & (df_valid['timestamp_full'] < fecha_fin + pd.Timedelta(days=1))]

# Graficar
plt.figure(figsize=(12, 6))
plt.plot(df_filtrado['timestamp_full'], df_filtrado['14'], color='blue', label='Current_R')
plt.plot(df_filtrado['timestamp_full'], df_filtrado['15'], color='green', label='Current_S')
plt.plot(df_filtrado['timestamp_full'], df_filtrado['16'], color='red', label='Current_T')
# plt.plot(df_filtrado['timestamp_full'], df_filtrado['26'], color='black', label='Current_N')

plt.title(f'Evolución de Corrientes ({fecha_inicio.date()} a {fecha_fin.date()})', fontsize=16)
plt.xlabel('Fecha y hora', fontsize=14)
plt.ylabel('Corriente', fontsize=14)
plt.legend(loc='upper right', fontsize=12)
plt.grid(True)

# Formatear eje X
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.DayLocator())  # Ticks diarios
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

# Agregar líneas verticales a las 08:00 y 18:00
dias = pd.date_range(start=fecha_inicio, end=fecha_fin, freq='D')

for dia in dias:
    hora_8 = dia + pd.Timedelta(hours=8)
    hora_18 = dia + pd.Timedelta(hours=18)
    ax.axvline(hora_8, color='gray', linestyle='--', linewidth=0.8)   # Línea fina a las 8:00
    ax.axvline(hora_18, color='gray', linestyle='--', linewidth=0.8)  # Línea fina a las 18:00

plt.gcf().autofmt_xdate()  # Rotar fechas
plt.tight_layout()
plt.show()

# Ruta base para guardar las imágenes
output_dir = r"D:\HERNAN\Hernan\educacion\UNC\FAMAF\100-EnergIQ"
os.makedirs(output_dir, exist_ok=True)  # Crear directorio si no existe

# Definir columnas y colores
columnas = {
    '14': ("Current_R", "blue"),
    '15': ("Current_S", "green"),
    '16': ("Current_T", "red")
}

# Crear un gráfico por cada corriente
for col, (nombre, color) in columnas.items():
    plt.figure(figsize=(12, 6))
    plt.plot(df_filtrado['timestamp_full'], df_filtrado[col], color=color, label=nombre)

    plt.title(f'{nombre} - Evolución ({fecha_inicio.date()} a {fecha_fin.date()})', fontsize=16)
    plt.xlabel('Fecha y hora', fontsize=14)
    plt.ylabel('Corriente', fontsize=14)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True)

    # Eje X con fechas
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    # Líneas verticales a las 08:00 y 18:00
    dias = pd.date_range(start=fecha_inicio, end=fecha_fin, freq='D')
    for dia in dias:
        hora_8 = dia + pd.Timedelta(hours=8)
        hora_18 = dia + pd.Timedelta(hours=18)
        ax.axvline(hora_8, color='gray', linestyle='--', linewidth=0.8)
        ax.axvline(hora_18, color='gray', linestyle='--', linewidth=0.8)

    plt.gcf().autofmt_xdate()
    plt.tight_layout()

    # Guardar imagen
    plt.savefig(os.path.join(output_dir, f"{nombre}.png"))
    plt.close()

# resumen de datos para el Factor de potencia total ('Power_Factor_Total')
resumen = df_trabajo['63'].describe()
print(resumen)

# Paso 1: Construir timestamp completo
df_trabajo['fecha_str'] = df_trabajo['timestamp1'].astype(str)
df_trabajo['hora_str'] = df_trabajo['hora'].astype(str)
df_trabajo['Fecha y hora'] = pd.to_datetime(df_trabajo['fecha_str'] + ' ' + df_trabajo['hora_str'], errors='coerce')

# Paso 2: Filtrar fechas válidas y ordenar
df_trabajo = df_trabajo[df_trabajo['Fecha y hora'].notna()]
df_trabajo = df_trabajo.sort_values('Fecha y hora')

# Paso 3: Definir período de análisis
#fecha_inicio = pd.to_datetime('2025-03-27')
#fecha_fin = pd.to_datetime('2025-04-10')

df_filtrado = df_trabajo[(df_trabajo['Fecha y hora'] >= fecha_inicio) & (df_trabajo['Fecha y hora'] <= fecha_fin)]

# Paso 4: Graficar
plt.figure(figsize=(20, 12))
plt.plot(df_filtrado['Fecha y hora'], df_filtrado['63'], marker='.', color='purple')
plt.title('FP (total) en función del tiempo')
plt.xlabel('Fecha y hora')
plt.ylabel('Potencia Aparente R')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print(list(df_filtrado.columns))


# Ruta base para guardar las imágenes
output_dir = r"D:\HERNAN\Hernan\educacion\UNC\FAMAF\100-EnergIQ"
os.makedirs(output_dir, exist_ok=True)

df_filtrado.columns = df_filtrado.columns.map(str)

# Diccionario de columnas y nombres de archivo
columnas_voltaje = {
    '1': "Voltaje1",
    '2': "Voltaje2",
    '3': "Voltaje3"
}

# Ticks de tiempo (00:00, 08:00, 18:00)
ticks = []
dias = pd.date_range(start=fecha_inicio.normalize(), end=fecha_fin.normalize(), freq='D')
for dia in dias:
    ticks.extend([
        dia,
        dia + pd.Timedelta(hours=8),
        dia + pd.Timedelta(hours=18)
    ])

# Crear y mostrar los gráficos uno a uno
for col, nombre in columnas_voltaje.items():
    plt.figure(figsize=(15, 10))
    
    plt.plot(df_filtrado['timestamp_full'], df_filtrado[col], label=nombre, marker='s', color='red')
    
    plt.xlabel('Fecha y hora', fontsize=14)
    plt.ylabel('Voltaje (V)', fontsize=14)
    plt.title(f'{nombre} en el Tiempo', fontsize=16)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True)

    ax = plt.gca()
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.xticks(rotation=45)

    # Líneas verticales
    for dia in dias:
        ax.axvline(dia, color='black', linestyle='-', linewidth=2.0)  # 00:00
        ax.axvline(dia + pd.Timedelta(hours=8), color='gray', linestyle='--', linewidth=0.8)  # 08:00
        ax.axvline(dia + pd.Timedelta(hours=18), color='gray', linestyle='--', linewidth=0.8)  # 18:00

    plt.tight_layout()
    plt.show()

    # Guardar gráfico
    plt.savefig(os.path.join(output_dir, f"{nombre}.png"))
    plt.close()




# --- Paso 2: Definir umbrales y duración mínima
threshold_upper = 5
threshold_lower = 5
min_duration = timedelta(minutes=5)

# --- Paso 3: Filtrar fechas y horas definidas

hora_inicio = pd.to_datetime('00:00').time()
hora_fin = pd.to_datetime('23:00').time()

df_trabajo_dias = df_trabajo[
    (df_trabajo['Fecha y hora'].dt.date >= fecha_inicio.date()) &
    (df_trabajo['Fecha y hora'].dt.date <= fecha_fin.date()) &
    (df_trabajo['Fecha y hora'].dt.time >= hora_inicio) &
    (df_trabajo['Fecha y hora'].dt.time <= hora_fin)
].copy()

# --- Paso 4: Evaluar condiciones de umbral
df_trabajo_dias['above_upper_threshold'] = df_trabajo_dias["30"] > threshold_upper
df_trabajo_dias['below_lower_threshold'] = df_trabajo_dias["30"] < threshold_lower
df_trabajo_dias['change_upper'] = df_trabajo_dias['above_upper_threshold'].ne(
    df_trabajo_dias['above_upper_threshold'].shift()
)
df_trabajo_dias['group_upper'] = df_trabajo_dias['change_upper'].cumsum()

# --- Paso 5: Agrupar y filtrar por duración
above_upper_threshold_groups = df_trabajo_dias[df_trabajo_dias['above_upper_threshold']].groupby('group_upper')
long_duration_upper_groups = above_upper_threshold_groups.filter(
    lambda x: x['Fecha y hora'].max() - x['Fecha y hora'].min() > min_duration
)

# --- Paso 6: Graficar
plt.figure(figsize=(20, 10))
plt.plot(df_trabajo_dias['Fecha y hora'], df_trabajo_dias["30"], label='Potencia')

# Resaltar los períodos de potencia alta prolongada
for _, group in long_duration_upper_groups.groupby('group_upper'):
    plt.plot(group['Fecha y hora'], group["30"], color='red')

# Configurar eje X
ax = plt.gca()

# Definir las horas que interesan
horas_interes = [0, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
ticks = []

# Crear los ticks en las horas específicas
dias = pd.date_range(start=fecha_inicio.normalize(), end=fecha_fin.normalize(), freq='D')

for dia in dias:
    for hora in horas_interes:
        ticks.append(dia + pd.Timedelta(hours=hora))

# Aplicar los ticks personalizados
ax.set_xticks(ticks)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.xticks(rotation=45)

# Dibujar líneas verticales diferenciadas
for dia in dias:
    for hora in horas_interes:
        momento = dia + pd.Timedelta(hours=hora)
        if hora == 0:
            ax.axvline(momento, color='black', linestyle='-', linewidth=2.0)
        elif hora in [8, 18]:
            ax.axvline(momento, color='gray', linestyle='-', linewidth=1.5)
        else:
            ax.axvline(momento, color='lightgray', linestyle='--', linewidth=0.8)

# Configurar líneas horizontales de umbrales
plt.axhline(y=threshold_upper, color='r', linestyle='--', label='Umbral Superior')
plt.axhline(y=threshold_lower, color='g', linestyle='--', label='Umbral Inferior')

# Títulos y ajustes finales
plt.xlabel('Tiempo')
plt.ylabel('Potencia')
plt.title(f'Detección de Potencia Superior al Umbral para los días {fecha_inicio.date()} a {fecha_fin.date()} entre {hora_inicio} y {hora_fin}')
plt.legend()
plt.grid(True)
plt.tight_layout()

# --- Guardar gráfico antes de mostrarlo
output_dir = r"D:\HERNAN\Hernan\educacion\UNC\FAMAF\100-EnergIQ"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "Potencia_umbral_superior.png")
plt.savefig(output_path)

## Agragado el 11-05
# Crear un set con los índices de los grupos prolongados
indices_eventos = long_duration_upper_groups.index
# Agregar la columna de evento al DataFrame principal
df_trabajo_dias['evento_umbral_superior'] = df_trabajo_dias.index.isin(indices_eventos)
# Guardar el CSV con la columna de umbral superior marcada
csv_output_path = os.path.join(output_dir, "Potencia_filtrada_con_evento_umbral.csv")
df_trabajo_dias.to_csv(csv_output_path, index=False)


# Mostrar el gráfico
plt.show()


# Cargar la serie temporal (asegúrate de tener tu DataFrame preparado)
serie_temporal = df_trabajo["30"] #'Active_Power_Total'

# Graficar la serie temporal
plt.figure(figsize=(20, 10))
plt.plot(serie_temporal)
plt.title('Serie Temporal')
plt.show()

# Prueba ADF (Augmented Dickey-Fuller) para verificar estacionaridad
# Realizar la prueba ADF
adf_test(serie_temporal)

# MAPA DE CALOR para estudio gráfico de correlación

# solo las columnas numéricas estén presentes
df_numerico = df_trabajo.select_dtypes(include=['float64', 'int64'])

plt.figure(figsize=(20, 10))
plt.suptitle('Estudio de correlación entre variables')

# matriz de correlación
sns.heatmap(df_numerico.corr(), annot=True, cmap='magma')
plt.show()

print(df_trabajo.columns)

# Reparación: estandarizar los nombres de columnas antes de usarlos
df_trabajo.columns = df_trabajo.columns.astype(str).str.strip().str.replace('.0', '', regex=False)

# Seleccionamos las columnas de interés para comparar con 'Active_Power_Total'
variables = [1, 2, 3, 13, 14, 15, 16, 26, 27, 28, 29, 57, 58, 59, 60, 61, 62, 63, 64, 65, 78]
variables = list(map(str, variables))
target = '30'

# Verificar columnas existentes
missing = [col for col in variables + [target] if col not in df_trabajo.columns]
print("Columnas faltantes:", missing)

# Reemplazo de comas por puntos y conversión a float
df_trabajo[variables + [target]] = (
    df_trabajo[variables + [target]]
    .replace(',', '.', regex=True)
    .apply(pd.to_numeric, errors='coerce')
)

# Gráfico hexbin
num_vars = len(variables)
cols = 3
rows = (num_vars + cols - 1) // cols

fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(18, 6 * rows))
fig.suptitle('Gráficos Hexbin - Variables vs 30 (Active_Power_Total)', fontsize=16)

for i, var in enumerate(variables):
    ax = axs[i // cols, i % cols]
    hb = ax.hexbin(df_trabajo[var], df_trabajo[target], gridsize=30, cmap='viridis', mincnt=1)
    ax.set_xlabel(var)
    ax.set_ylabel('30')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()



df_numerico = df_trabajo.select_dtypes(include=['float64', 'int64']) # Columnas numéricas

# Cálculo de correlaciones
correlaciones = df_numerico.corr()

# SOlo mostramos la columna de correlación con nuestro target
correlacion_con_target = correlaciones["30"].sort_values(ascending=False)
#print(correlacion_con_target)
correlacion_con_target

variables_to_drop = [27, 26, 14, 29, 64, 28, 63, 78, 'fecha_str', 'hora_str', 'timestamp_full', 'Fecha y hora']
variables_to_drop = list(map(str, variables_to_drop))  # convertir a string

df_trabajo_reduced = df_trabajo.drop(columns=variables_to_drop)

df_trabajo_reduced.columns


df_trabajo_reduced.info()

# Verificar si hay datos faltantes
missing_values = df_trabajo_reduced.isnull().sum()

# Mostrar las columnas con datos faltantes
print(missing_values[missing_values > 0])



# Eliminar las columnas especificadas
df_trabajo_reduced


####correccion de grafico pronostico
df_trabajo_reduced['timestamp1'] = pd.to_datetime(df_trabajo['timestamp1'] + ' ' + df_trabajo['hora'], errors='coerce')
# Eliminar filas con timestamps nulos (por si acaso)
df_trabajo_reduced = df_trabajo_reduced[df_trabajo_reduced['timestamp1'].notna()].reset_index(drop=True)

# Rellenar valores nulos
df_trabajo_reduced = df_trabajo_reduced.ffill()

cols_all_nan = df_trabajo_reduced.columns[df_trabajo_reduced.isna().all()].tolist()
print("Columnas con solo NaN:", cols_all_nan)

df_trabajo_reduced = df_trabajo_reduced.drop(columns=cols_all_nan)




# Escalar datos
#scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = RobustScaler()
#scaler = MaxAbsScaler()

#scaled_data = scaler.fit_transform(df_trabajo_reduced)
#print("¿NaNs después de escalar?", np.isnan(scaled_data).any())  # ← ahora debe ser False

#scaled_df = pd.DataFrame(scaled_data, columns=df_trabajo_reduced.columns)


# Guardar la columna de tiempo real por separado
timestamps_reales = pd.to_datetime(
    df_trabajo_reduced['timestamp1'].astype(str) + ' ' + df_trabajo_reduced['hora'],
    errors='coerce'
)

# Separar características y target (excluyendo timestamp y hora)
features = df_trabajo_reduced.drop(columns=['30', 'timestamp1', 'hora'])
target = df_trabajo_reduced[['30']]

# Escalar
scaler_X = RobustScaler()
scaler_y = RobustScaler()

scaled_X = scaler_X.fit_transform(features)
scaled_y = scaler_y.fit_transform(target)

# Reconstruir DataFrame escalado
scaled_df = pd.DataFrame(scaled_X, columns=features.columns)
scaled_df['30'] = scaled_y
scaled_df['timestamp'] = timestamps_reales.values  # timestamp ya combinado y en formato datetime


# Crear una versión del DataFrame sin la columna 'timestamp' para la red LSTM
scaled_df_lstm = scaled_df.drop(columns=['timestamp'])

# Crear secuencias para LSTM

# Parámetro de tamaño de ventana
time_steps = 50
X, y = create_sequences(scaled_df_lstm, target_col='30', time_steps=time_steps)

# Dividir en entrenamiento y prueba
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Verificar dimensiones y NaNs
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")
print("¿NaNs en X_train?", np.isnan(X_train).sum())
print("¿NaNs en y_train?", np.isnan(y_train).sum())




#model.compile(optimizer='adam', loss='mean_squared_error')


print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")
print(np.isnan(X_train).sum(), np.isnan(y_train).sum())  # Verificar NaN en los datos



model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error')

history = model.fit(X_train, y_train, epochs=23, batch_size=32, validation_data=(X_test, y_test))





# Predecir con 10% de los datos
num_predictions = int(0.1 * len(X_test))
X_test_reducido = X_test[:num_predictions]

predictions = model.predict(X_test_reducido)

# Invertir la escala correctamente usando scaler_y
predictions_actual = scaler_y.inverse_transform(predictions)
y_test_actual = scaler_y.inverse_transform(y_test[:num_predictions].reshape(-1, 1))

print("Valores reales:", y_test_actual[:5].flatten())
print("Predicciones desescaladas:", predictions_actual[:5].flatten())


# Crear el eje X real para los datos originales
timestamps_reales = df_trabajo_reduced['timestamp1']
timestamp_pred_start = timestamps_reales.iloc[-1] + pd.Timedelta(seconds=1)
timestamp_pred = pd.date_range(start=timestamp_pred_start, periods=len(predictions), freq='T')

# Crear el gráfico
plt.figure(figsize=(14, 5))
plt.plot(timestamps_reales, df_trabajo_reduced['30'], label='df_trabajo_reduced Active_Power_Total', color='green')
plt.plot(timestamp_pred, predictions_actual, label='Predicted Consumption', color='orange')
plt.axvline(x=timestamps_reales.iloc[-1], color='red', linestyle='--', label='Start of Predictions')

# Configuraciones del gráfico
plt.ylim(0, 50)
plt.xlabel('Fecha y hora')
plt.ylabel('Consumption')
plt.title('Predicted Consumption vs Active Power Total (respetando tiempo real)')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Guardar el gráfico en disco
output_dir = r"D:\HERNAN\Hernan\educacion\UNC\FAMAF\100-EnergIQ"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "Predicted_Consumption_LSTM.png")
plt.savefig(output_path)

# Mostrar en pantalla
plt.show()


# Guardar datos en CSV
df_real = pd.DataFrame({
    'timestamp': timestamps_reales,
    'actual_consumption': df_trabajo_reduced['30']
})
# Asegurarse de que predictions_actual sea un vector 1D
predictions_actual_flat = predictions_actual.ravel() if hasattr(predictions_actual, 'ravel') else predictions_actual
df_pred = pd.DataFrame({
    'timestamp': timestamp_pred,
    'predicted_consumption': predictions_actual_flat
})

df_combined = pd.concat([df_real.set_index('timestamp'), df_pred.set_index('timestamp')], axis=1).reset_index()

csv_output_path = os.path.join(output_dir, "Predicted_vs_Actual_Consumption.csv")
df_combined.to_csv(csv_output_path, index=False)


# Asegúrate de incluir esto en un entorno Jupyter o Colab

# Entrenamiento con control de pérdida
train_loss = []
val_loss = []

for epoch in range(25):  # Número de épocas
    model.fit(X_train, y_train, epochs=1, batch_size=32, validation_data=(X_test, y_test), verbose=0)
    train_loss.append(model.evaluate(X_train, y_train, verbose=0))
    val_loss.append(model.evaluate(X_test, y_test, verbose=0))

# Graficar pérdidas
plt.figure(figsize=(12, 6))
plt.plot(train_loss, label='Train loss')
plt.plot(val_loss, label='Validation loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Train and Validation Loss')
plt.show()