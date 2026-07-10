import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Comparador de Activos", layout="wide")

st.title("Comparador de Activos Financieros")
st.write(
    "Esta herramienta compara activos financieros a partir de datos históricos de precio, "
    "rendimiento y riesgo. El resultado es orientativo y no constituye una recomendación financiera."
)

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
    try:
        datos = yf.download(
            tickers,
            period=periodo,
            auto_adjust=True,
            progress=False,
            threads=False
        )

        if datos.empty:
            return pd.DataFrame()

        if isinstance(datos.columns, pd.MultiIndex):
            if "Close" in datos.columns.get_level_values(0):
                precios = datos["Close"]
            else:
                return pd.DataFrame()
        else:
            if "Close" in datos.columns:
                precios = datos[["Close"]]
                precios.columns = tickers
            else:
                return pd.DataFrame()

        precios = precios.dropna(how="all")
        precios = precios.ffill().dropna()

        return precios

    except Exception as e:
        st.error("No se pudieron descargar correctamente los datos. Puede ser un error temporal de Yahoo Finance.")
        st.write("Detalle técnico:", e)
        return pd.DataFrame()

datos = descargar_datos(tickers, periodo)
if datos.empty or len(datos) < 2:
    st.error("No hay datos suficientes para los activos seleccionados. Probá con otros activos o con otro período.")
    st.stop()

# Renombrar columnas para que aparezcan con nombres lindos
mapa_nombres = dict(zip(tickers, activos_elegidos))
datos = datos.rename(columns=mapa_nombres)

if datos.empty:
    st.error("No se pudieron descargar datos para los activos seleccionados.")
    st.stop()

# -----------------------------
# Cálculo de métricas
# -----------------------------

retornos = datos.pct_change().dropna()

rendimiento_acumulado = (datos.iloc[-1] / datos.iloc[0]) - 1
volatilidad_anualizada = retornos.std() * np.sqrt(252)
retorno_anualizado = retornos.mean() * 252
sharpe = retorno_anualizado / volatilidad_anualizada

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

st.dataframe(
    tabla_metricas.round(2),
    use_container_width=True
)

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

fig1, ax1 = plt.subplots(figsize=(10, 5))

for activo in precios_normalizados.columns:
    ax1.plot(precios_normalizados.index, precios_normalizados[activo], label=activo)

ax1.set_title("Evolución normalizada base 100")
ax1.set_xlabel("Fecha")
ax1.set_ylabel("Valor normalizado")
ax1.legend()
ax1.grid(True)

st.pyplot(fig1)

# -----------------------------
# Gráfico 2: riesgo vs rendimiento
# -----------------------------

st.subheader("Riesgo vs rendimiento")

fig2, ax2 = plt.subplots(figsize=(10, 5))

for activo in tabla_metricas.index:
    x = tabla_metricas.loc[activo, "Volatilidad anualizada (%)"]
    y = tabla_metricas.loc[activo, "Rendimiento acumulado (%)"]
    ax2.scatter(x, y, s=120)
    ax2.text(x, y, activo, fontsize=10, ha="left", va="bottom")

ax2.set_title("Riesgo vs rendimiento")
ax2.set_xlabel("Volatilidad anualizada (%)")
ax2.set_ylabel("Rendimiento acumulado (%)")
ax2.grid(True)

st.pyplot(fig2)

# -----------------------------
# Gráfico 3: ranking
# -----------------------------

st.subheader("Ranking final según perfil")

fig3, ax3 = plt.subplots(figsize=(10, 5))

ax3.bar(ranking_final.index, ranking_final["Score final"])
ax3.set_title(f"Ranking para perfil {perfil}")
ax3.set_xlabel("Activo")
ax3.set_ylabel("Score final")
ax3.tick_params(axis="x", rotation=45)
ax3.grid(axis="y")

st.pyplot(fig3)

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
