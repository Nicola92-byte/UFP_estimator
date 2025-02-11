import streamlit as st
import tempfile
import os
import Agente_logging as agent  # Importa il modulo dell'agente con logging

# Imposta la configurazione della pagina e lo stile con toni di blu
st.set_page_config(page_title="Function Point Estimator", layout="wide")
st.markdown(
    """
    <style>
    body {
        background-color: #f0f8ff;
    }
    .main {
        background-color: #e6f2ff;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #007acc;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #005f99;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Function Point Estimator")
st.markdown(
    "Carica un file **.docx** per generare una Specifica Funzionale completa e stimare i Function Point secondo IFPUG.")

# Caricamento del file DOCX tramite file uploader
uploaded_file = st.file_uploader("Scegli un file .docx", type=["docx"])

if uploaded_file is not None:
    # Salva il file in una posizione temporanea
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    st.success("File caricato con successo!")

    # Quando l'utente clicca il pulsante, chiama l'agente per elaborare il file
    if st.button("Elabora il file"):
        with st.spinner("Elaborazione in corso..."):
            try:
                # Richiama la funzione run_agent definita in agente_logging.py
                summary, spec = agent.run_agent(tmp_file_path)
                st.markdown("### Sommario")
                st.info(summary)
                st.markdown("### Specifica Funzionale Generata")
                st.write(spec)
            except Exception as e:
                st.error(f"Si è verificato un errore: {e}")

    # Dopo l'elaborazione, rimuove il file temporaneo
    os.remove(tmp_file_path)
