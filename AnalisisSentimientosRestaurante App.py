import io
import re
import json
import urllib.parse
import urllib.request
from collections import Counter
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from textblob import TextBlob
from deep_translator import GoogleTranslator
import nltk

# ------------------------------------------------------------
# DESCARGA AUTOMÁTICA DE DEPENDENCIAS DE TEXTBLOB
# ------------------------------------------------------------
@st.cache_resource
def preparar_nltk():
    try:
        nltk.data.find('corpora/movie_reviews')
    except LookupError:
        nltk.download('movie_reviews', quiet=True)
        nltk.download('punkt', quiet=True)

preparar_nltk()

# ------------------------------------------------------------
# CONFIGURACIÓN DE LA PÁGINA Y HABILITACIÓN DE TRADUCCIÓN NATIVA
# ------------------------------------------------------------
st.set_page_config(
    page_title="Análisis de Sentimientos - Restaurantes El Salvador",
    page_icon="🍽️",
    layout="wide"
)

components.html(
    """
    <script>
        var htmlElement = window.parent.document.querySelector('html');
        if (htmlElement) {
            htmlElement.setAttribute('lang', 'en');
            htmlElement.removeAttribute('translate');
            htmlElement.classList.remove('notranslate');
        }
    </script>
    """,
    height=0,
    width=0
)

# ------------------------------------------------------------
# CONFIGURACIÓN DE LA BARRA LATERAL
# ------------------------------------------------------------
st.sidebar.header("⚙️ Configuración del Restaurante")
nombre_lugar = st.sidebar.text_input("Nombre del Restaurante:", value="Clifest")
ubicacion = st.sidebar.text_input("Ubicación:", value="Ahuachapán, El Salvador")

st.sidebar.divider()
st.sidebar.info("💡 Modifica el nombre y la ubicación para personalizar la interfaz y el reporte descargable.")

# ------------------------------------------------------------
# INICIALIZACIÓN DEL HISTORIAL ACUMULATIVO
# ------------------------------------------------------------
if "historial" not in st.session_state:
    st.session_state.historial = []

# ------------------------------------------------------------
# DATOS DE DEMOSTRACIÓN
# ------------------------------------------------------------
COMENTARIOS_DEMO = """Cibo mediocre e prezzi troppo alti per quello che offrono.
La comida estuvo excelente y la atención del mesero fue de primera.
Le service était très lent et désagréable.
A comida estava maravilhosa, adoramos tudo!
The atmosphere is nice, but prices are high.
Das Essen war hervorragend und das Personal sehr freundlich."""

# Mapeo de códigos ISO a nombres legibles de idiomas
NOMBRES_IDIOMAS = {
    "es": "Español",
    "en": "Inglés",
    "fr": "Francés",
    "it": "Italiano",
    "pt": "Portugués",
    "de": "Alemán"
}

# ------------------------------------------------------------
# MOTOR DE TRADUCCIÓN ROBUSTO CON DETECCIÓN DE IDIOMA
# ------------------------------------------------------------
def traducir_texto(texto, target_lang="es"):
    if not texto or not texto.strip():
        return texto, True, "es"

    lang_detectado = "es"

    # Método 1: API Directa de Google (Obtiene traducción e idioma detectado)
    try:
        url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl={target_lang}&dt=t&q={urllib.parse.quote(texto)}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
        with urllib.request.urlopen(req, timeout=5) as response:
            res = json.loads(response.read().decode('utf-8'))
            traducido = "".join([segmento[0] for segmento in res[0] if segmento and segmento[0]])
            if len(res) > 2 and isinstance(res[2], str):
                lang_detectado = res[2].lower()
            if traducido and traducido.strip():
                return traducido, True, lang_detectado
    except Exception:
        pass

    # Método 2: Respaldo con deep-translator
    try:
        res = GoogleTranslator(source="auto", target=target_lang).translate(texto)
        if res and res.strip():
            return res, True, lang_detectado
    except Exception:
        pass

    return texto, False, lang_detectado

def predict_sentiment(text):
    if not text or not text.strip():
        return None

    texto_original = text.strip()

    # 1. Traducción al Español y Detección de Idioma
    texto_es, ok_es, lang_code = traducir_texto(texto_original, target_lang="es")

    # 2. Traducción al Inglés (Para análisis con TextBlob)
    texto_en, ok_en, _ = traducir_texto(texto_original, target_lang="en")

    error_traduccion = not (ok_es and ok_en)

    # Identificar nombre del idioma
    idioma_nombre = NOMBRES_IDIOMAS.get(lang_code, lang_code.capitalize())

    # Análisis de sentimiento con TextBlob
    blob = TextBlob(texto_en)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    if polarity > 0.1:
        sentiment, emoji = "Positivo", "😊"
    elif polarity < -0.1:
        sentiment, emoji = "Negativo", "😞"
    else:
        sentiment, emoji = "Neutral", "😐"

    return {
        "text": texto_original,
        "language": idioma_nombre,
        "translation_es": texto_es,
        "translation_en": texto_en,
        "sentiment": sentiment,
        "emoji": emoji,
        "polarity": polarity,
        "subjectivity": subjectivity,
        "translation_error": error_traduccion
    }

def limpiar_comentarios(texto):
    if not texto:
        return []
    return [linea.strip() for linea in texto.splitlines() if linea.strip()]

def analizar_comentarios(comentarios):
    resultados = []
    for comentario in comentarios:
        res = predict_sentiment(comentario)
        if res:
            resultados.append(res)
    return resultados

def obtener_resumen(resultados):
    if not resultados:
        return None

    total = len(resultados)
    positivos = sum(1 for r in resultados if r["sentiment"] == "Positivo")
    negativos = sum(1 for r in resultados if r["sentiment"] == "Negativo")
    neutrales = sum(1 for r in resultados if r["sentiment"] == "Neutral")

    promedio_polaridad = sum(r["polarity"] for r in resultados) / total
    promedio_subjetividad = sum(r["subjectivity"] for r in resultados) / total

    cantidades = {"Positivo": positivos, "Negativo": negativos, "Neutral": neutrales}
    sentimiento_predominante = max(cantidades, key=cantidades.get)

    return {
        "total": total,
        "positivos": positivos,
        "negativos": negativos,
        "neutrales": neutrales,
        "porcentaje_positivos": (positivos / total) * 100,
        "porcentaje_negativos": (negativos / total) * 100,
        "porcentaje_neutrales": (neutrales / total) * 100,
        "promedio_polaridad": promedio_polaridad,
        "promedio_subjetividad": promedio_subjetividad,
        "sentimiento_predominante": sentimiento_predominante
    }

def obtener_palabras_frecuentes(resultados, top_n=8):
    texto_completo = " ".join([r["translation_es"].lower() for r in resultados])
    palabras = re.findall(r'\b[a-záéíóúñ]{4,}\b', texto_completo)
    
    stopwords = {
        "para", "como", "pero", "este", "esta", "estos", "estas", "muy", "mas", "más", 
        "con", "sin", "por", "sobre", "entre", "hasta", "desde", "todo", "toda", "todos",
        "todas", "donde", "cuando", "quien", "cual", "sino", "bien", "tambien", "también"
    }
    
    palabras_filtradas = [p for p in palabras if p not in stopwords]
    conteo = Counter(palabras_filtradas)
    return pd.DataFrame(conteo.most_common(top_n), columns=["Palabra", "Frecuencia"])

# ------------------------------------------------------------
# INTERFAZ DE USUARIO (STREAMLIT)
# ------------------------------------------------------------
st.title(f"🍽️ Análisis de Sentimientos - {nombre_lugar}")
st.caption("Sistema de análisis de comentarios para restaurantes de El Salvador.")
st.info(f"📍 Restaurante seleccionado: **{nombre_lugar}** | Ubicación: **{ubicacion}**")

tab_comentarios, tab_resultados, tab_filtrar_exportar, tab_tecnico = st.tabs([
    "💬 Comentarios e Historial", "📊 Resultados y Palabras Clave", "🔍 Filtrar y Exportar CSV", "📋 Detalle técnico"
])

# --- PESTAÑA: COMENTARIOS E HISTORIAL ---
with tab_comentarios:
    st.subheader("💬 Agregar comentarios al historial")
    st.write("Ingresa o carga comentarios. Se irán sumando al historial acumulativo.")

    fuente = st.radio("Fuente de los comentarios", ["Escribir manualmente", "Datos de demostración", "Cargar CSV"], horizontal=True)
    comentarios = []

    if fuente == "Escribir manualmente":
        texto_input = st.text_area("Comentarios (Uno por línea)", placeholder="La comida estuvo excelente...\nEl servicio fue bastante lento.\nGreat restaurant experience!", height=150)
        comentarios = limpiar_comentarios(texto_input)
    elif fuente == "Datos de demostración":
        comentarios = limpiar_comentarios(COMENTARIOS_DEMO)
        st.text_area("Datos de demostración", value="\n".join(comentarios), height=150, disabled=True)
    elif fuente == "Cargar CSV":
        st.write("El archivo debe incluir una columna llamada `comentario`.")
        archivo = st.file_uploader("Selecciona un archivo CSV", type=["csv"])
        if archivo:
            try:
                df = pd.read_csv(io.BytesIO(archivo.getvalue()), encoding="utf-8-sig")
            except UnicodeDecodeError:
                df = pd.read_csv(io.BytesIO(archivo.getvalue()), encoding="latin-1")
            
            df.columns = [str(c).strip().lower() for c in df.columns]
            if "comentario" not in df.columns:
                st.error("❌ El CSV debe contener la columna 'comentario'.")
            else:
                comentarios = df["comentario"].dropna().astype(str).str.strip().tolist()
                comentarios = [c for c in comentarios if c]
                st.success(f"✅ Cargados {len(comentarios)} comentarios listos para procesar.")

    if st.button("🔎Analizar Comentarios y Traducir ➕", type="primary", use_container_width=True):
        if not comentarios:
            st.warning("⚠️ Ingresa o carga al menos un comentario.")
        else:
            with st.spinner("Procesando y agregando al historial..."):
                nuevos_resultados = analizar_comentarios(comentarios)
                st.session_state.historial.extend(nuevos_resultados)

            errores = sum(1 for r in nuevos_resultados if r.get("translation_error", False))
            if errores > 0:
                st.warning("⚠️ Ocurrió un problema de traducción en algunos comentarios. Se evaluó el texto original.")
            
            st.success(f"✅ ¡Se agregaron {len(nuevos_resultados)} comentarios al historial! Total acumulado: {len(st.session_state.historial)}")

    # TABLA DE HISTORIAL CON COLUMNA DE IDIOMA Y TRADUCCIÓN CONDICIONAL
    st.divider()
    st.subheader(f"📜 Historial Acumulado ({len(st.session_state.historial)} comentarios)")

    if st.session_state.historial:
        df_historial = pd.DataFrame([
            {
                "#": i + 1,
                "Comentario Original": r["text"],
                "Idioma": r.get("language", "Español"),
                "Traducción al Español": "" if r.get("language", "Español") == "Español" else r["translation_es"],
                "Sentimiento": f"{r['emoji']} {r['sentiment']}",
                "Polaridad": round(r["polarity"], 3)
            } for i, r in enumerate(st.session_state.historial)
        ])
        st.dataframe(df_historial, use_container_width=True, hide_index=True)

        col_del1, col_del2 = st.columns([2, 1])
        with col_del1:
            opciones_borrar = [f"#{i+1} - {r['text'][:40]}..." for i, r in enumerate(st.session_state.historial)]
            seleccion_borrar = st.selectbox("Selecciona un comentario para eliminar:", opciones_borrar)
            idx_borrar = opciones_borrar.index(seleccion_borrar)

            if st.button("🗑️ Eliminar comentario seleccionado", use_container_width=True):
                st.session_state.historial.pop(idx_borrar)
                st.success("Comentario eliminado del historial.")
                st.rerun()

        with col_del2:
            st.write("---")
            if st.button("🧹 Limpiar TODO el Historial", type="secondary", use_container_width=True):
                st.session_state.historial = []
                st.success("Historial vaciado.")
                st.rerun()
    else:
        st.info("El historial está vacío. Agrega comentarios arriba para comenzar.")

# --- PESTAÑA: RESULTADOS Y PALABRAS CLAVE ---
with tab_resultados:
    st.subheader("📊 Resultados generales del Historial")
    resumen = obtener_resumen(st.session_state.historial)

    if not resumen:
        st.info("💡 Agrega comentarios en la pestaña '💬 Comentarios e Historial' para ver las gráficas y la clasificación.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total en Historial", resumen["total"])
        m2.metric("😊 Positivos", resumen["positivos"], f"{resumen['porcentaje_positivos']:.1f}%")
        m3.metric("😞 Negativos", resumen["negativos"], f"{resumen['porcentaje_negativos']:.1f}%")
        m4.metric("😐 Neutrales", resumen["neutrales"], f"{resumen['porcentaje_neutrales']:.1f}%")

        st.divider()
        p1, p2, p3 = st.columns(3)
        p1.metric("Polaridad promedio", f"{resumen['promedio_polaridad']:.3f}")
        p2.metric("Subjetividad promedio", f"{resumen['promedio_subjetividad']:.3f}")
        p3.metric("Sentimiento Predominante", resumen["sentimiento_predominante"])

        st.divider()
        col_graf, col_words = st.columns(2)

        with col_graf:
            st.subheader("📈 Distribución de sentimientos")
            datos_sentimientos = pd.DataFrame({"Cantidad": {
                "😊 Positivos": resumen["positivos"], 
                "😞 Negativos": resumen["negativos"], 
                "😐 Neutrales": resumen["neutrales"]
            }})
            st.bar_chart(datos_sentimientos)

        with col_words:
            st.subheader("🔤 Palabras más recurrentes")
            df_words = obtener_palabras_frecuentes(st.session_state.historial)
            if not df_words.empty:
                st.bar_chart(df_words.set_index("Palabra"))
            else:
                st.write("No hay suficiente texto para calcular palabras frecuentes.")

        st.divider()

        # --- SECCIÓN: CLASIFICACIÓN DE COMENTARIOS ---
        st.subheader("📑 Clasificación de comentarios")

        positivos_list = [r for r in st.session_state.historial if r["sentiment"] == "Positivo"]
        negativos_list = [r for r in st.session_state.historial if r["sentiment"] == "Negativo"]
        neutrales_list = [r for r in st.session_state.historial if r["sentiment"] == "Neutral"]

        col_pos, col_neg, col_neu = st.columns(3)

        with col_pos:
            st.markdown(f"#### 😊 Comentarios positivos ({len(positivos_list)})")
            if positivos_list:
                for idx, item in enumerate(positivos_list, 1):
                    st.success(f"**{idx}.** {item['text']}")
            else:
                st.caption("No hay comentarios positivos aún.")

        with col_neg:
            st.markdown(f"#### 😞 Comentarios negativos ({len(negativos_list)})")
            if negativos_list:
                for idx, item in enumerate(negativos_list, 1):
                    st.error(f"**{idx}.** {item['text']}")
            else:
                st.caption("No hay comentarios negativos aún.")

        with col_neu:
            st.markdown(f"#### 😐 Comentarios neutrales ({len(neutrales_list)})")
            if neutrales_list:
                for idx, item in enumerate(neutrales_list, 1):
                    st.info(f"**{idx}.** {item['text']}")
            else:
                st.caption("No hay comentarios neutrales aún.")

# --- PESTAÑA: FILTRAR Y EXPORTAR ---
with tab_filtrar_exportar:
    st.subheader("🔍 Filtrar Comentarios del Historial y Exportar")

    if not st.session_state.historial:
        st.info("💡 El historial está vacío. Agrega comentarios para exportar un reporte.")
    else:
        filtro_sentimiento = st.multiselect(
            "Filtrar por sentimiento:",
            options=["Positivo", "Negativo", "Neutral"],
            default=["Positivo", "Negativo", "Neutral"]
        )

        resultados_filtrados = [r for r in st.session_state.historial if r["sentiment"] in filtro_sentimiento]

        df_exportar = pd.DataFrame([
            {
                "Restaurante": nombre_lugar,
                "Ubicacion": ubicacion,
                "Comentario Original": r["text"],
                "Traduccion Espanol": r["translation_es"],
                "Traduccion Ingles": r["translation_en"],
                "Sentimiento": r["sentiment"],
                "Emoji": r["emoji"],
                "Polaridad": round(r["polarity"], 3),
                "Subjetividad": round(r["subjectivity"], 3)
            } for r in resultados_filtrados
        ])

        st.dataframe(df_exportar, use_container_width=True)

        csv_data = df_exportar.to_csv(index=False, encoding="utf-8-sig")
        nombre_archivo_limpio = re.sub(r'[^\w\-]', '_', nombre_lugar.lower())

        st.download_button(
            label=f"📥 Descargar Reporte Completo de {nombre_lugar} (CSV)",
            data=csv_data,
            file_name=f"reporte_sentimientos_{nombre_archivo_limpio}.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True
        )

# --- PESTAÑA: DETALLE TÉCNICO ---
with tab_tecnico:
    st.subheader("📋 Detalle técnico individual")
    if not st.session_state.historial:
        st.info("Sin datos en el historial.")
    else:
        opciones = [f"#{i + 1} - {r['text'][:50]}" for i, r in enumerate(st.session_state.historial)]
        seleccion = st.selectbox("Selecciona un comentario para inspeccionar", opciones)
        idx = opciones.index(seleccion)
        r = st.session_state.historial[idx]

        st.write(f"**Original:** {r['text']}")
        st.write(f"**Traducción al Español:** {r['translation_es']}")
        st.write(f"**Traducción al Inglés:** {r['translation_en']}")
        st.metric("Polaridad", f"{r['polarity']:.3f}")
        st.metric("Subjetividad", f"{r['subjectivity']:.3f}")