import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import requests
import time

st.set_page_config(page_title="Comparador de Activos", layout="wide")

st.title("Comparador de Activos Financieros")
st.write(
    "Esta herramienta compara activos financieros a partir de datos históricos de precio, "
    "rendimiento y riesgo. El resultado es orientativo y no constituye una recomendación financiera."
)

if st.button("Actualizar datos"):
    st.cache_data.clear()
    st.rerun()

# -----------------------------
# Selección del usuario
# -----------------------------

activos_disponibles = {
    "Bitcoin": "BTC-USD",
    "Ether": "ETH-USD",
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Tesla": "TSLA",
    "S&P 500 ETF": "SPY",
    "Nasdaq 100 ETF": "QQQ",
    "Gold ETF": "GLD"
}

col1, col2, col3 = st.columns(3)

with col1:
    activos_elegidos = st.multiselect(
        "Elegí los activos a comparar",
        options=list(activos_disponibles.keys()),
        default=["Bitcoin", "Ether", "S&P 500 ETF"]
    )

with col2:
    periodo = st.selectbox(
        "Elegí el período de análisis",
        options=["6mo", "1y", "2y", "5y"],
        index=1
    )

with col3:
    perfil = st.selectbox(
        "Elegí el perfil de inversor",
        options=["conservador", "moderado", "agresivo"],
        index=1
    )

if len(activos_elegidos) < 2:
    st.warning("Elegí al menos dos activos para comparar.")
    st.stop()

tickers = [activos_disponibles[a] for a in activos_elegidos]

# -----------------------------
# Descargar datos
# -----------------------------

@st.cache_data(ttl=1800)
def descargar_datos(tickers, periodo):
    series = {}
    errores = []

    periodos_segundos = {
        "6mo": 60 * 60 * 24 * 183,
        "1y": 60 * 60 * 24 * 365,
        "2y": 60 * 60 * 24 * 365 * 2,
        "5y": 60 * 60 * 24 * 365 * 5
    }

    ahora = int(time.time())
    inicio = ahora - periodos_segundos.get(periodo, 60 * 60 * 24 * 365)

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    for ticker in tickers:
        try:
            url = (
                f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
                f"?period1={inicio}&period2={ahora}&interval=1d"
            )

            response = requests.get(url, headers=headers, timeout=15)

            if response.status_code != 200:
                errores.append(f"{ticker}: error HTTP {response.status_code}.")
                continue

            data = response.json()

            result = data.get("chart", {}).get("result", None)

            if not result:
                errores.append(f"{ticker}: Yahoo no devolvió datos.")
                continue

            result = result[0]

            timestamps = result.get("timestamp", [])
            quote = result.get("indicators", {}).get("quote", [{}])[0]
            closes = quote.get("close", [])

            if not timestamps or not closes:
                errores.append(f"{ticker}: datos incompletos.")
                continue

            df = pd.DataFrame({
                "Fecha": pd.to_datetime(timestamps, unit="s").normalize(),
                ticker: closes
            })

            df = df.dropna()

            if len(df) < 30:
                errores.append(f"{ticker}: tiene menos de 30 observaciones.")
                continue

            serie = df.set_index("Fecha")[ticker]
            series[ticker] = serie

        except Exception as e:
            errores.append(f"{ticker}: error al descargar datos ({e}).")

    if len(series) == 0:
        return pd.DataFrame(), errores

    precios = pd.concat(series.values(), axis=1, join="inner")
    precios = precios.sort_index()
    precios = precios.replace([np.inf, -np.inf], np.nan).dropna()

    return precios, errores


datos, errores_descarga = descargar_datos(tickers, periodo)

# Renombrar columnas para que aparezcan con nombres lindos
mapa_nombres = dict(zip(tickers, activos_elegidos))
datos = datos.rename(columns=mapa_nombres)

with st.expander("Diagnóstico de descarga"):
    st.write(f"Activos seleccionados: {activos_elegidos}")
    st.write(f"Tickers usados: {tickers}")
    st.write(f"Período seleccionado: {periodo}")
    st.write(f"Filas descargadas: {datos.shape[0]}")
    st.write(f"Columnas descargadas: {datos.shape[1]}")
    if errores_descarga:
        st.write("Errores detectados:")
        for error in errores_descarga:
            st.write(error)
    else:
        st.write("No se detectaron errores de descarga.")

if datos.empty or len(datos) < 30:
    st.error("No hay datos suficientes para los activos seleccionados. Probá con otros activos o con otro período.")
    st.stop()

if len(datos.columns) < 2:
    st.error("No hay al menos dos activos con datos suficientes para comparar.")
    st.stop()

datos = datos.ffill().bfill()

# -----------------------------
# Cálculo de métricas
# -----------------------------

retornos = datos.pct_change().replace([np.inf, -np.inf], np.nan).dropna()

if retornos.empty:
    st.error("No se pudieron calcular retornos para los activos seleccionados.")
    st.stop()

rendimiento_acumulado = (datos.iloc[-1] / datos.iloc[0]) - 1
volatilidad_anualizada = retornos.std() * np.sqrt(252)
retorno_anualizado = retornos.mean() * 252
sharpe = retorno_anualizado / volatilidad_anualizada.replace(0, np.nan)

maximos_acumulados = datos.cummax()
drawdown = (datos / maximos_acumulados) - 1
maxima_caida = drawdown.min()

tabla_metricas = pd.DataFrame({
    "Rendimiento acumulado (%)": rendimiento_acumulado * 100,
    "Retorno anualizado (%)": retorno_anualizado * 100,
    "Volatilidad anualizada (%)": volatilidad_anualizada * 100,
    "Sharpe simplificado": sharpe,
    "Máxima caída (%)": maxima_caida * 100
})

tabla_metricas = tabla_metricas.replace([np.inf, -np.inf], np.nan).dropna()

if tabla_metricas.empty or len(tabla_metricas) < 2:
    st.error("No se pudieron calcular métricas suficientes para comparar los activos seleccionados.")
    st.stop()

# -----------------------------
# Score según perfil
# -----------------------------

ranking = tabla_metricas.copy()
ranking["Caída máxima absoluta"] = ranking["Máxima caída (%)"].abs()

def normalizar_mayor_mejor(serie):
    if serie.max() == serie.min():
        return serie * 0 + 1
    return (serie - serie.min()) / (serie.max() - serie.min())

def normalizar_menor_mejor(serie):
    if serie.max() == serie.min():
        return serie * 0 + 1
    return (serie.max() - serie) / (serie.max() - serie.min())

ranking["score_rendimiento"] = normalizar_mayor_mejor(ranking["Rendimiento acumulado (%)"])
ranking["score_sharpe"] = normalizar_mayor_mejor(ranking["Sharpe simplificado"])
ranking["score_volatilidad"] = normalizar_menor_mejor(ranking["Volatilidad anualizada (%)"])
ranking["score_caida"] = normalizar_menor_mejor(ranking["Caída máxima absoluta"])

if perfil == "conservador":
    pesos = {
        "rendimiento": 0.10,
        "sharpe": 0.20,
        "volatilidad": 0.40,
        "caida": 0.30
    }
elif perfil == "moderado":
    pesos = {
        "rendimiento": 0.25,
        "sharpe": 0.35,
        "volatilidad": 0.20,
        "caida": 0.20
    }
else:
    pesos = {
        "rendimiento": 0.45,
        "sharpe": 0.30,
        "volatilidad": 0.10,
        "caida": 0.15
    }

ranking["Score final"] = (
    pesos["rendimiento"] * ranking["score_rendimiento"] +
    pesos["sharpe"] * ranking["score_sharpe"] +
    pesos["volatilidad"] * ranking["score_volatilidad"] +
    pesos["caida"] * ranking["score_caida"]
)

ranking_final = ranking.sort_values("Score final", ascending=False)

# -----------------------------
# Resultados principales
# -----------------------------

st.subheader("Tabla de métricas")

st.dataframe(tabla_metricas.round(2), use_container_width=True)

mejor_activo = ranking_final.index[0]
peor_activo = ranking_final.index[-1]

st.subheader("Resultado según el perfil elegido")

st.success(
    f"Para un perfil **{perfil}**, el activo mejor posicionado según las métricas históricas analizadas es **{mejor_activo}**."
)

st.write(
    f"Este activo obtuvo el mayor score al combinar rendimiento acumulado, volatilidad, "
    f"máxima caída y Sharpe simplificado. El activo peor posicionado bajo estos criterios fue **{peor_activo}**."
)

st.caption(
    "Aclaración: el resultado se basa únicamente en datos históricos y en una ponderación simplificada de métricas. "
    "No constituye una recomendación financiera."
)

# -----------------------------
# Gráfico 1: precios normalizados
# -----------------------------

st.subheader("Evolución normalizada de los activos")

precios_normalizados = datos / datos.iloc[0] * 100
precios_plot = precios_normalizados.reset_index()
precios_plot = precios_plot.rename(columns={precios_plot.columns[0]: "Fecha"})
precios_plot = precios_plot.melt(
    id_vars="Fecha",
    var_name="Activo",
    value_name="Valor normalizado"
)

chart_precios = (
    alt.Chart(precios_plot)
    .mark_line()
    .encode(
        x="Fecha:T",
        y="Valor normalizado:Q",
        color="Activo:N",
        tooltip=["Fecha:T", "Activo:N", "Valor normalizado:Q"]
    )
    .properties(height=420)
    .interactive()
)

st.altair_chart(chart_precios, use_container_width=True)

# -----------------------------
# Gráfico 2: riesgo vs rendimiento
# -----------------------------

st.subheader("Riesgo vs rendimiento")

scatter_data = tabla_metricas.reset_index().rename(columns={"index": "Activo"})

base = alt.Chart(scatter_data).encode(
    x=alt.X("Volatilidad anualizada (%):Q", title="Volatilidad anualizada (%)"),
    y=alt.Y("Rendimiento acumulado (%):Q", title="Rendimiento acumulado (%)")
)

points = base.mark_circle(size=130).encode(
    color="Activo:N",
    tooltip=[
        "Activo:N",
        "Rendimiento acumulado (%):Q",
        "Volatilidad anualizada (%):Q",
        "Sharpe simplificado:Q",
        "Máxima caída (%):Q"
    ]
)

labels = base.mark_text(align="left", dx=8, dy=-5).encode(
    text="Activo:N"
)

st.altair_chart((points + labels).properties(height=420).interactive(), use_container_width=True)

# -----------------------------
# Gráfico 3: ranking
# -----------------------------

st.subheader("Ranking final según perfil")

ranking_plot = ranking_final.reset_index().rename(columns={"index": "Activo"})

chart_ranking = (
    alt.Chart(ranking_plot)
    .mark_bar()
    .encode(
        x=alt.X("Activo:N", sort="-y"),
        y=alt.Y("Score final:Q"),
        color="Activo:N",
        tooltip=["Activo:N", "Score final:Q"]
    )
    .properties(height=420)
)

st.altair_chart(chart_ranking, use_container_width=True)

# -----------------------------
# Explicación metodológica
# -----------------------------

st.subheader("Cómo se calcula el ranking")

st.write(
    "El ranking combina cuatro elementos: rendimiento acumulado, Sharpe simplificado, volatilidad anualizada "
    "y máxima caída. Según el perfil seleccionado, cambian las ponderaciones."
)

st.write(
    "- Perfil conservador: prioriza menor volatilidad y menor caída máxima.\n"
    "- Perfil moderado: busca equilibrio entre rendimiento y riesgo.\n"
    "- Perfil agresivo: prioriza mayor rendimiento, aceptando más volatilidad."
)

st.warning(
    "Esta herramienta es educativa y demostrativa. No incorpora variables como noticias, balances, tasas de interés, "
    "liquidez, contexto macroeconómico ni expectativas futuras."
)
