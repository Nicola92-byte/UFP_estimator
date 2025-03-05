import streamlit as st
import tempfile
import os
import agente_calcolo as agent

# Imposta la configurazione della pagina con icona e layout
st.set_page_config(page_title="Function Point Estimator", layout="wide", page_icon=":chart_with_upwards_trend:")

# CSS personalizzato per uno stile accattivante (toni di blu) e header
st.markdown(
    """
    <style>
    /* Impostazioni di base */
    body {
        background-color: #f0f8ff;
        font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
    }
    .main {
        background-color: #e6f2ff;
        padding: 20px;
        border-radius: 10px;
    }
    /* Stile dei pulsanti */
    .stButton>button {
        background-color: #007acc;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #005f99;
    }
    /* Header e istruzioni */
    .header {
        text-align: center;
        padding: 20px;
    }
    .instructions {
        text-align: center;
        font-size: 18px;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header: immagine e titolo
st.markdown("<div class='header'>", unsafe_allow_html=True)
st.image(
    r"img1_calcolo.png",
    width=150)
st.markdown("<h1>Function Point Estimator</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='font-size: 20px;'>Calcola i Function Point secondo gli standard IFPUG in modo semplice e intuitivo!</p>",
    unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Istruzioni con icona di calcolatrice
st.markdown("<div class='instructions'>", unsafe_allow_html=True)
st.markdown("""
### Istruzioni:
1. Carica un file **.docx** contenente l'Analisi Requisiti Utente (ARU).
2. L'app genererà automaticamente una Specifica Funzionale (SF) completa.
3. Verrà visualizzato un sommario, la stima dei Function Point e informazioni aggiuntive (ufp_info).
""", unsafe_allow_html=True)
st.image(r"img2_software.png", width=80)
st.markdown("</div>", unsafe_allow_html=True)

# File uploader per il file DOCX
uploaded_file = st.file_uploader("Scegli un file .docx", type=["docx"])

if uploaded_file is not None:
    # Salva il file in un percorso temporaneo
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    st.success("File caricato con successo!")

    # Quando l'utente clicca il pulsante, chiama l'agente per elaborare il file
    if st.button("Elabora il file"):
        with st.spinner("Elaborazione in corso..."):
            try:
                # run_agent ora restituisce 3 valori: short_summary, final_text, ufp_info
                summary, spec, ufp_info = agent.run_agent(tmp_file_path)
                st.markdown("### Sommario")
                st.info(summary)
                st.markdown("### INFO UFP")
                st.info(ufp_info)
                st.markdown("### Specifica Funzionale Generata")
                st.write(spec)
            except Exception as e:
                st.error(f"Si è verificato un errore: {e}")
    # Rimuove il file temporaneo dopo l'elaborazione
    os.remove(tmp_file_path)
