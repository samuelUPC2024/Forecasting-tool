# --------------------------------------------------------------------------
# |                   IMPORTAR LIBRERÍAS                                   |
# --------------------------------------------------------------------------
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
# --- IMPORTACIONES DE PROPHET ---
from prophet import Prophet
from prophet.plot import plot_components_plotly
# --------------------------------
import os
import requests
import glob 

# --------------------------------------------------------------------------
# |                   CONFIGURACIÓN DE LA PÁGINA                             |
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="SAMPHEL ENERGY - Prophet",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------------------------------
# |                   FUNCIONES DE CARGA DE DATOS                          |
# --------------------------------------------------------------------------

@st.cache_data
def load_asepeyo_energy_data(file_path):
    """Carga y procesa el archivo de consumo energético desde una ruta."""
    try:
        # Intenta con ; si falla ,
        try:
            df = pd.read_csv(file_path, sep=';', decimal=',')
            if 'Fecha' not in df.columns or 'Energía activa (kWh)' not in df.columns:
                df = pd.read_csv(file_path, sep=',', decimal='.')
        except Exception:
             df = pd.read_csv(file_path, sep=',', decimal='.')

        if 'Fecha' not in df.columns or 'Energía activa (kWh)' not in df.columns:
            st.error(f"El archivo {file_path} debe contener 'Fecha' y 'Energía activa (kWh)'.")
            return pd.DataFrame()
            
        df.rename(columns={'Fecha': 'fecha', 'Energía activa (kWh)': 'consumo_kwh'}, inplace=True)
        # Asumiendo formato DD/MM/YYYY o similar (dayfirst=True)
        df['fecha'] = pd.to_datetime(df['fecha'], dayfirst=True, errors='coerce')
        df.dropna(subset=['fecha'], inplace=True)
        return df
    except Exception as e:
        st.error(f"Error al procesar el archivo de consumo {file_path}: {e}")
        return pd.DataFrame()


@st.cache_data
def load_nasa_weather_data(file_path):
    """Carga y procesa el archivo de clima histórico de NASA POWER desde una ruta."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        start_row = 0
        for i, line in enumerate(lines):
            # Busca el inicio de los datos
            if line.strip().startswith("YEAR,MO,DY,HR"):
                start_row = i
                break
        
        df = pd.read_csv(file_path, skiprows=start_row)
        
        expected_cols = ['YEAR', 'MO', 'DY', 'HR', 'T2M']
        if not all(col in df.columns for col in expected_cols):
            st.error(f"El archivo {file_path} debe contener 'YEAR', 'MO', 'DY', 'HR', 'T2M'.")
            return pd.DataFrame()

        df['fecha'] = pd.to_datetime(df[['YEAR', 'MO', 'DY', 'HR']].rename(columns={'YEAR': 'year', 'MO': 'month', 'DY': 'day', 'HR': 'hour'}))
        df.rename(columns={'T2M': 'temperatura_c'}, inplace=True)
        df['temperatura_c'] = df['temperatura_c'].replace(-999, np.nan).ffill()
        
        return df[['fecha', 'temperatura_c']]
    except Exception as e:
        st.error(f"Error al procesar el archivo de clima de la NASA {file_path}: {e}")
        return pd.DataFrame()


@st.cache_data
def get_weather_forecast(api_key, lat, lon):
    """Obtiene el pronóstico del tiempo diario desde la API de Meteosource."""
    BASE_URL = "https://www.meteosource.com/api/v1/free/point"
    params = {
        "lat": lat,
        "lon": lon,
        "sections": "daily",
        "units": "metric",
        "key": api_key
    }
    try:
        response = requests.get(BASE_URL, params=params)
        if response.status_code == 200:
            data = response.json()
            daily_data = data.get('daily', {}).get('data', [])
            if not daily_data:
                st.error("La API no devolvió datos de pronóstico diario.")
                return pd.DataFrame()
            processed_data = []
            for day in daily_data:
                processed_data.append({
                    'fecha': day.get('day'),
                    'temp_max_c': day.get('all_day', {}).get('temperature_max'),
                    'temp_min_c': day.get('all_day', {}).get('temperature_min')
                })
            df_clima_futuro = pd.DataFrame(processed_data)
            df_clima_futuro['fecha'] = pd.to_datetime(df_clima_futuro['fecha'])
            df_clima_futuro.dropna(inplace=True)
            df_clima_futuro['temp_avg_c'] = (df_clima_futuro['temp_max_c'] + df_clima_futuro['temp_min_c']) / 2
            return df_clima_futuro
        else:
            st.error(f"Error en la API de Meteosource (Código {response.status_code}): {response.json().get('detail', 'Error desconocido')}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error al conectar con la API del clima: {e}")
        return pd.DataFrame()

# --------------------------------------------------------------------------
# |                   BARRA LATERAL (SIDEBAR)                              |
# --------------------------------------------------------------------------

# Nota: Asumo que 'unnamed.jpg' no está disponible en este entorno.
st.sidebar.title("Configuración de la Predicción (Prophet)")
st.sidebar.markdown("---")

st.sidebar.header("1. Carga de Datos Históricos")

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.path.abspath('.')

DATA_DIR = os.path.join(SCRIPT_DIR, "data")

if not os.path.exists(DATA_DIR):
    st.sidebar.warning(f"La carpeta de datos '{DATA_DIR}' no existe. Creando...")
    os.makedirs(DATA_DIR, exist_ok=True)
    st.sidebar.info(f"Por favor, coloca tus archivos 'energy_*.csv' y 'weather_*.csv' en la carpeta '{DATA_DIR}'.")

selected_energy_file = None
selected_weather_file = None

try:
    energy_pattern = os.path.join(DATA_DIR, "energy_*.csv")
    weather_pattern = os.path.join(DATA_DIR, "weather_*.csv")
    
    energy_files = [os.path.basename(f) for f in glob.glob(energy_pattern)]
    weather_files = [os.path.basename(f) for f in glob.glob(weather_pattern)]

    if not energy_files:
        st.sidebar.error(f"No se encontraron archivos 'energy_*.csv' en la carpeta '{DATA_DIR}'.")
    else:
        selected_energy_file = st.sidebar.selectbox("Selecciona el archivo de Consumo", energy_files)

    if not weather_files:
        st.sidebar.error(f"No se encontraron archivos 'weather_*.csv' en la carpeta '{DATA_DIR}'.")
    else:
        selected_weather_file = st.sidebar.selectbox("Selecciona el archivo de Clima Histórico", weather_files)
        
except Exception as e:
    st.sidebar.error(f"Error al leer la carpeta '{DATA_DIR}': {e}")


st.sidebar.markdown("---")

st.sidebar.header("2. Configuración del Pronóstico (API)")
api_key = st.sidebar.text_input("API Key de Meteosource", type="password")
lat = st.sidebar.text_input("Latitud", "40.4168")
lon = st.sidebar.text_input("Longitud", "-3.7038")

st.sidebar.markdown("---")

st.sidebar.header("3. Variables Adicionales")
ocupacion_media = st.sidebar.slider("Ocupación Media (%) del Centro", 0, 100, 80)
# Número de días a pronosticar (basado en la disponibilidad del clima futuro, que es ~15 días)
dias_pronostico = st.sidebar.slider("Días a Pronosticar (API limit)", 7, 30, 15)

# --------------------------------------------------------------------------
# |                   CUERPO DE LA APLICACIÓN                              |
# --------------------------------------------------------------------------

st.title("PREDICCIÓN DE CONSUMO ENERGÉTICO (MODELO PROPHET)")
st.subheader("Utilizando Regresores de Clima y Ocupación")
st.markdown("---")


if selected_energy_file and selected_weather_file and api_key and lat and lon:
    with st.spinner('Procesando datos, contactando API y entrenando el modelo Prophet...'):
        
        energy_path = os.path.join(DATA_DIR, selected_energy_file)
        weather_path = os.path.join(DATA_DIR, selected_weather_file)
        
        df_energia = load_asepeyo_energy_data(energy_path)
        df_clima_pasado = load_nasa_weather_data(weather_path)
        
        # Obtener solo el número de días que el usuario seleccionó o el máximo disponible de la API.
        df_clima_futuro_full = get_weather_forecast(api_key, lat, lon)
        df_clima_futuro = df_clima_futuro_full.head(dias_pronostico)

        if df_clima_futuro.empty:
            st.error("No se pudo obtener el pronóstico del clima. Revisa tu API Key o las coordenadas e inténtalo de nuevo.")
            st.stop()
            
        if not df_energia.empty and not df_clima_pasado.empty:
            
            # 1. Agregación de datos históricos (a diario)
            df_historico_horario = pd.merge(df_energia, df_clima_pasado, on='fecha', how='inner')
            df_historico_horario.dropna(inplace=True)
            df_historico_horario.set_index('fecha', inplace=True)
            
            # Consumo Diario y Clima Diario Promedio
            consumo_diario = df_historico_horario['consumo_kwh'].resample('D').sum()
            clima_diario_avg = df_historico_horario['temperatura_c'].resample('D').mean()
            
            df_historico_daily = pd.concat([consumo_diario, clima_diario_avg], axis=1)
            df_historico_daily.rename(columns={
                'temperatura_c': 'temp_avg_c'
            }, inplace=True)
            df_historico_daily.reset_index(inplace=True)
            df_historico_daily.dropna(subset=['consumo_kwh'], inplace=True)
            
            # 2. Ingeniería de Features/Regresores
            df_historico_daily['ocupacion'] = ocupacion_media
            
            # Preparar DataFrames para Prophet
            # Prophet requiere 'ds' (datestamp) y 'y' (target)
            prophet_df = df_historico_daily[['fecha', 'consumo_kwh', 'temp_avg_c', 'ocupacion']].rename(
                columns={'fecha': 'ds', 'consumo_kwh': 'y'}
            )
            
            # 3. Inicializar, configurar y entrenar el modelo Prophet
            modelo = Prophet(
                growth='linear', # Tendencia lineal
                seasonality_mode='multiplicative', # Estacionalidad multiplicativa (común para energía)
                daily_seasonality=False, # Ya que estamos a nivel diario
                weekly_seasonality=True,
                yearly_seasonality=True
            )

            # Agregar regresores (deben estar presentes en el historial y en el dataframe futuro)
            modelo.add_regressor('temp_avg_c')
            modelo.add_regressor('ocupacion')

            modelo.fit(prophet_df)

            # 4. Crear el DataFrame Futuro para la Predicción
            # Crear un dataframe con el historial + las fechas de pronóstico
            future_dates = modelo.make_future_dataframe(periods=dias_pronostico, freq='D', include_history=False)
            
            # Unir los regresores de clima y ocupación para el futuro
            df_futuro_regressors = df_clima_futuro[['fecha', 'temp_avg_c']].rename(columns={'fecha': 'ds'})
            df_futuro_regressors['ocupacion'] = ocupacion_media
            
            # Asegurarse de que el dataframe futuro tenga los regresores correctos para todas las fechas
            future_with_regressors = pd.merge(future_dates, df_futuro_regressors, on='ds', how='left')
            
            # Si hay fechas en 'future_dates' que no están en 'df_futuro_regressors' (por ejemplo, si la API dio menos días), 
            # rellenamos los valores de clima (solo la ocupación es constante)
            future_with_regressors['ocupacion'].fillna(ocupacion_media, inplace=True)
            future_with_regressors['temp_avg_c'].ffill(inplace=True) # Uso ffill simple para temperaturas faltantes (si las hubiera)

            # 5. Realizar la Predicción
            # Se requiere que el dataframe de predicción contenga solo las columnas 'ds' y los regresores
            forecast = modelo.predict(future_with_regressors[['ds', 'temp_avg_c', 'ocupacion']])
            
            # 6. Preparar Resultados
            df_futuro = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(
                columns={'ds': 'fecha', 'yhat': 'consumo_predicho_kwh'}
            )

            # Métrica de bondad de ajuste simple (R² del historial)
            prophet_train_pred = modelo.predict(prophet_df[['ds', 'temp_avg_c', 'ocupacion']])
            r2_score = 1 - (np.sum((prophet_df['y'].values - prophet_train_pred['yhat'].values)**2) / np.sum((prophet_df['y'].values - np.mean(prophet_df['y'].values))**2))
            
            st.success("✅ ¡Modelo Prophet entrenado y predicción completada!")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Ajuste Histórico (R²)", f"{r2_score:.2f}")
            col2.metric("Período Pronosticado", f"{dias_pronostico} días")
            col3.metric("Consumo Total Predicho", f"{df_futuro['consumo_predicho_kwh'].sum():,.0f} kWh")
            st.markdown("---")

            # --- GRÁFICOS ---
            tab1, tab2 = st.tabs(["Gráfico de Predicción", "Componentes del Modelo"])

            with tab1:
                st.subheader("Consumo Histórico vs. Predicción Futura (Diario)")
                
                # Plotear historial y pronóstico
                df_historico_plot = prophet_df[['ds', 'y']].rename(columns={'ds': 'fecha', 'y': 'Consumo'})
                df_historico_plot['Tipo'] = 'Histórico'
                
                df_futuro_plot = df_futuro[['fecha', 'consumo_predicho_kwh']].rename(columns={'consumo_predicho_kwh': 'Consumo'})
                df_futuro_plot['Tipo'] = 'Predicción'
                
                df_plot = pd.concat([df_historico_plot, df_futuro_plot])
                
                fig = px.line(df_plot, x='fecha', y='Consumo', color='Tipo', 
                            title='Consumo Histórico vs. Predicción Futura (Prophet)',
                            labels={'fecha': 'Fecha', 'Consumo': 'Consumo Total (kWh/día)'},
                            color_discrete_map={'Histórico': 'blue', 'Predicción': 'orange'})
                
                # Línea de separación entre el historial y el futuro
                fig.add_vline(x=prophet_df['ds'].max(), line_width=2, line_dash="dash", line_color="red", annotation_text="Inicio Predicción")
                
                # Añadir rango de incertidumbre al gráfico de predicción
                fig.add_trace(px.scatter(df_futuro, x='fecha', y='yhat_lower', line_color='rgba(255,165,0,0.1)').data[0])
                fig.add_trace(px.scatter(df_futuro, x='fecha', y='yhat_upper', line_color='rgba(255,165,0,0.1)').data[0])
                fig.update_traces(showlegend=False)

                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                st.subheader("Componentes de la Estacionalidad del Modelo Prophet")
                # Gráfico estándar de Prophet para componentes (Tendencia, Estacionalidad Semanal/Anual, Regresores)
                fig_comp = plot_components_plotly(modelo, forecast)
                # Prophet genera subplots, se necesita un pequeño ajuste para el diseño de Streamlit
                fig_comp.update_layout(height=800)
                st.plotly_chart(fig_comp, use_container_width=True)


            st.subheader("Datos Detallados de la Predicción (Diaria)")
            st.dataframe(df_futuro[['fecha', 'consumo_predicho_kwh', 'yhat_lower', 'yhat_upper']].round(2))

else:
    st.info("ℹ️ **Para comenzar**, por favor asegúrate de que tus archivos de datos históricos estén en la carpeta `data` y completa la configuración de la API en la barra lateral izquierda.")
    st.markdown("""
    Esta herramienta utiliza el **Modelo Prophet** para:
    1.  **Preparar** los datos históricos de consumo y clima (`ds` y `y`).
    2.  **Obtener** el pronóstico del tiempo diario más reciente (temperatura media) como *regresor adicional*.
    3.  **Entrenar** el modelo Prophet, capturando la tendencia, la estacionalidad (semanal y anual) y el efecto de la temperatura y ocupación.
    4.  **Predecir** el **consumo total diario** futuro.
    """)
