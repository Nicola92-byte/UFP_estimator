# import os
# import re
# import pickle
# import logging
# import numpy as np
# import faiss
# import openai
# from sentence_transformers import SentenceTransformer
# from PyPDF2 import PdfReader
#
# # 1) Import delle funzioni dai tuoi script
# from estrazione_damas_wave import get_functional_requirements
# from estrazione_dati_utili_wave import parse_aru_docx
#
#
# #######################################################
# # Configurazione del logging avanzato
# #######################################################
# def init_logger(log_file='app.log'):
#     logger = logging.getLogger('FP_Calc')
#     logger.setLevel(logging.DEBUG)
#
#     # Handler per scrivere su file (livello DEBUG)
#     fh = logging.FileHandler(log_file)
#     fh.setLevel(logging.DEBUG)
#
#     # Handler per la console (livello INFO)
#     ch = logging.StreamHandler()
#     ch.setLevel(logging.INFO)
#
#     # Formatter per entrambi gli handler
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     fh.setFormatter(formatter)
#     ch.setFormatter(formatter)
#
#     # Aggiungi handler al logger (evita duplicazioni se il logger è già configurato)
#     if not logger.handlers:
#         logger.addHandler(fh)
#         logger.addHandler(ch)
#
#     return logger
#
#
# # Inizializza il logger
# logger = init_logger()
#
# #######################################################
# # Configurazione OpenAI
# #######################################################
# from dotenv import load_dotenv
#
# load_dotenv()
#
# openai.api_type = os.getenv("OPENAI_API_TYPE")
# openai.api_base = os.getenv("OPENAI_API_BASE")
# openai.api_version = os.getenv("OPENAI_API_VERSION")
# openai.api_key = os.getenv("OPENAI_API_KEY")
# DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")
#
# if not openai.api_key or not DEPLOYMENT_NAME:
#     logger.critical(
#         "Le variabili d'ambiente OPENAI_API_KEY e/o DEPLOYMENT_NAME non sono state impostate correttamente nel file .env.")
#     raise ValueError("Variabili d'ambiente mancanti.")
#
#
# #######################################################
# # Lettura e chunk del PDF (manuale IFPUG) + Indice FAISS
# #######################################################
# def read_pdf_and_chunk(pdf_path, chunk_size=500):
#     logger.info(f"Lettura del PDF: {pdf_path}")
#     with open(pdf_path, "rb") as f:
#         reader = PdfReader(f)
#         all_text = []
#         for page in reader.pages:
#             page_text = page.extract_text()
#             if page_text:
#                 all_text.append(page_text.strip())
#     big_text = "\n".join(all_text)
#     chunks = [big_text[i:i + chunk_size] for i in range(0, len(big_text), chunk_size)]
#     logger.info(f"Suddiviso in {len(chunks)} chunk.")
#     return chunks
#
#
# def create_faiss_index(embeddings):
#     dimension = embeddings.shape[1]
#     index = faiss.IndexFlatL2(dimension)
#     index.add(embeddings.astype("float32"))
#     return index
#
#
# def build_faiss_index(chunks, model,
#                       faiss_index_path="manual_damas.index",
#                       embeddings_path="manual_damas.npy"):
#     """
#     Se i file dell'indice ed embeddings esistono, li carica, altrimenti li crea.
#     """
#     if os.path.exists(faiss_index_path) and os.path.exists(embeddings_path):
#         logger.info("Caricamento FAISS index ed embeddings da file di cache.")
#         embeddings = np.load(embeddings_path)
#         faiss_index = faiss.read_index(faiss_index_path)
#         return faiss_index
#     else:
#         logger.info("Nessun indice FAISS trovato in cache, creazione da zero.")
#         embeddings = model.encode(chunks)
#         np.save(embeddings_path, embeddings)
#         faiss_index = create_faiss_index(embeddings)
#         faiss.write_index(faiss_index, faiss_index_path)
#         return faiss_index
#
#
# def retrieve_context(query, faiss_index, chunks, model, k=1):
#     logger.info("Recupero contesto dal manuale tramite FAISS.")
#     query_emb = model.encode([query]).astype("float32")
#     distances, indices = faiss_index.search(query_emb, k)
#     relevant_chunks = [chunks[idx] for idx in indices[0] if idx < len(chunks)]
#     context = "\n".join(relevant_chunks)
#     if len(context) > 2000:
#         context = context[:2000]
#     return context
#
#
# #######################################################
# # Caching del contenuto del manuale (chunki del PDF)
# #######################################################
# def get_manual_chunks(pdf_path, chunk_size=500, cache_file="manual_chunks.pkl"):
#     if os.path.exists(cache_file):
#         logger.info(f"Caricamento dei chunk dal file di cache: {cache_file}")
#         with open(cache_file, 'rb') as f:
#             chunks = pickle.load(f)
#     else:
#         logger.info("Cache non trovata, leggo e chunko il PDF manuale.")
#         chunks = read_pdf_and_chunk(pdf_path, chunk_size)
#         with open(cache_file, 'wb') as f:
#             pickle.dump(chunks, f)
#         logger.info(f"Cache salvata in {cache_file}.")
#     return chunks
#
#
# #######################################################
# # Funzioni di supporto: pre-analisi e clamp
# #######################################################
# def quick_pre_analysis(requirements_text):
#     """
#     Conta "RF" e alcune parole chiave per dare un punteggio e ricavarne un range ipotetico di UFP.
#     """
#     lines = requirements_text.splitlines()
#     rf_count = sum(1 for line in lines if "RF" in line)
#
#     text_lower = requirements_text.lower()
#     keywords = {
#         "nomine": 2,
#         "matching": 2,
#         "cas": 1,
#         "sas": 1,
#         "backlog": 2,
#         "gui": 1,
#         "db": 1,
#         "design thinking": 2,
#         "microservizi": 2
#     }
#     score = sum(text_lower.count(kw) * weight for kw, weight in keywords.items())
#     score += rf_count
#
#     if score < 5:
#         suggested_range = "20–30"
#     elif score < 10:
#         suggested_range = "25–35"
#     elif score < 15:
#         suggested_range = "30–40"
#     elif score < 20:
#         suggested_range = "35–45"
#     else:
#         suggested_range = "40–50"
#
#     pre_analysis_text = (
#         f"Numero di requisiti (RF) trovati: {rf_count}\n"
#         f"Score totale: {score}\n"
#         f"Range ipotizzato: {suggested_range}\n"
#     )
#     return pre_analysis_text
#
#
# def clamp_range_in_text(answer, min_val, max_val):
#     """
#     Se nella risposta viene indicato il totale UFP, questo viene forzato a rientrare nel range [min_val, max_val].
#     """
#     pattern = re.compile(r'Totale UFP\s*=\s*(\d+)')
#     matches = pattern.findall(answer)
#     if matches:
#         total_fp = int(matches[0])
#         total_fp = max(min_val, min(total_fp, max_val))
#         answer = re.sub(r'Totale UFP\s*=\s*\d+', f'Totale UFP = {total_fp}', answer)
#     return answer
#
#
# #######################################################
# # Generazione stima UFP e Specifica Funzionale con few-shot
# #######################################################
# def generate_fp_estimate_detailed(requirements_text, manual_context, pre_analysis_text, ufp_info):
#     """
#     Genera il documento di Specifica Funzionale (SF) integrando le informazioni già estratte.
#     Implementa un meccanismo di retry con logging avanzato.
#     """
#     prompt = f"""[ARU Da Analizzare]
# Requisiti funzionali (già estratti):
# {requirements_text}
#
# Informazioni utili e Sommario (già estratti):
# {ufp_info}
#
# PRE-ANALISI AUTOMATICA:
# {pre_analysis_text}
#
# ESTRATTO MANUALE IFPUG:
# {manual_context}
#
# Richiesta:
# 1) Genera un documento di Specifica Funzionale (SF) completo (almeno 3-4 pagine) utile al calcolo dei Function Point (IFPUG) seguendo questa struttura:
#    - Introduzione: contesto, obiettivi, committente, vincoli di pianificazione.
#    - Descrizione Generale del Sistema: ambito, utenti principali, interfacce con altri sistemi.
#    - Definizione dei Boundary del Sistema: confini interni ed esterni, specificando gli archivi logici interni (ILF) e quelli esterni (EIF).
#    - Requisiti Non Funzionali: prestazioni, sicurezza, usabilità, affidabilità, manutenibilità, portabilità.
#    - Regole di Business: regole che influenzano operazioni e calcoli.
#    - Eccezioni e Condizioni Speciali.
#    - Report e Output del Sistema: contenuto, formato, frequenza.
#    - Interfacce Utente: schermate e interazioni.
#    - Processi di Interfacciamento con Altri Sistemi: flusso dati e interfacce.
#    - Casi d'Uso e Scenari Operativi.
#    - Dettagli sull'Architettura del Sistema.
#    - Allegati e Appendici (Glossario, Diagrammi, Prototipi).
# 2) Se il contenuto supera la capacità di una singola risposta, dividilo in più parti.
# 3) Integra le informazioni già estratte senza duplicazioni.
# """
#     messages = [
#         {
#             "role": "system",
#             "content": (
#                 "Sei un esperto di Function Point Analysis (IFPUG). "
#                 "Quando rispondi, integra le informazioni già estratte e segui uno schema di calcolo passo-passo. "
#                 "Fornisci il documento di Specifica Funzionale completo, con il totale dei Function Point alla fine, "
#                 "che rientri nel range desiderato."
#             )
#         },
#         {"role": "user", "content": prompt}
#     ]
#
#     max_retries = 3
#     for attempt in range(max_retries):
#         try:
#             logger.info(f"Tentativo {attempt + 1} di generazione del documento SF.")
#             response = openai.ChatCompletion.create(
#                 engine=DEPLOYMENT_NAME,  # Per AzureOpenAI standard
#                 messages=messages,
#                 max_tokens=3000,
#                 temperature=0.0,
#                 top_p=1.0,
#                 presence_penalty=0.0,
#                 frequency_penalty=0.0
#             )
#             answer = response["choices"][0]["message"]["content"].strip()
#             # Forza il totale UFP a rientrare nel range (ad es. 20-50)
#             answer = clamp_range_in_text(answer, 20, 50)
#             logger.info("Documento SF generato con successo.")
#             return answer
#         except Exception as e:
#             logger.error(f"Errore durante la richiesta OpenAI (tentativo {attempt + 1}/{max_retries}): {e}")
#             if attempt == max_retries - 1:
#                 logger.critical("Numero massimo di tentativi raggiunto. Interruzione del processo.")
#                 raise
#     return "Nessuna risposta ottenuta."
#
#
# #######################################################
# # MAIN
# #######################################################
# if __name__ == "__main__":
#     # Percorso del manuale IFPUG (PDF)
#     pdf_path = r"C:\Users\A395959\PycharmProjects\pyMilvus\Function_Point_calcManual.pdf"
#
#     # Usa il caching per ottenere i chunk del manuale
#     chunks = get_manual_chunks(pdf_path, chunk_size=500, cache_file="manual_chunks.pkl")
#
#     # Carica il modello per le embedding
#     logger.info("Caricamento del modello SentenceTransformer.")
#     model = SentenceTransformer("all-MiniLM-L6-v2")
#
#     # Costruisci o carica l'indice FAISS
#     faiss_index = build_faiss_index(chunks, model)
#
#     # Percorso del documento ARU
#     docx_path = r"C:\Users\A395959\PycharmProjects\pyMilvus\ARU_inventate\aru_fittizia_GPT o1.docx"
#
#     # Estrazione dei requisiti funzionali (gestita da estrazione_damas_wave)
#     logger.info("Estrazione dei requisiti funzionali dal documento ARU.")
#     requirements_text = get_functional_requirements(docx_path)
#
#     # Estrazione delle informazioni utili e del sommario (gestita da estrazione_dati_utili_wave)
#     ufp_info, _, short_summary = parse_aru_docx(docx_path)
#
#     # Recupera il contesto dal manuale IFPUG tramite FAISS
#     query = (
#         "Recupera linee guida IFPUG su EI, EO, EQ, ILF, EIF e range complessità, "
#         "così posso stimare i function points su una ARU con requisiti incompleti."
#     )
#     manual_context = retrieve_context(query, faiss_index, chunks, model, k=1)
#
#     # Pre-analisi dei requisiti
#     pre_analysis_txt = quick_pre_analysis(requirements_text)
#
#     # Genera la stima e il documento di Specifica Funzionale
#     final_answer = generate_fp_estimate_detailed(
#         requirements_text=requirements_text,
#         manual_context=manual_context,
#         pre_analysis_text=pre_analysis_txt,
#         ufp_info=ufp_info
#     )
#
#     logger.info("STAMPA UN PICCOLO RIASSUNTO:")
#     logger.info(short_summary)
#
#     # Stampa il documento finale
#     print(final_answer)


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################

# agente_logging.py
import os
import re
import pickle
import logging
import numpy as np
import faiss
import openai
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from estrazione_damas_wave import get_functional_requirements
from estrazione_dati_utili_wave import parse_aru_docx
from dotenv import load_dotenv

# Carica le variabili d'ambiente dal file .env
load_dotenv()

openai.api_type = os.getenv("OPENAI_API_TYPE")
openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_version = os.getenv("OPENAI_API_VERSION")
openai.api_key = os.getenv("OPENAI_API_KEY")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")

if not openai.api_key or not DEPLOYMENT_NAME:
    raise ValueError("Le variabili d'ambiente OPENAI_API_KEY e/o DEPLOYMENT_NAME non sono state impostate correttamente nel file .env.")


###############################################
# Configurazione del logging avanzato
###############################################
def init_logger(log_file='app.log'):
    logger = logging.getLogger('FP_Calc')
    logger.setLevel(logging.DEBUG)
    # Handler per scrivere su file (DEBUG)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    # Handler per la console (INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger

logger = init_logger()


###############################################
# Funzioni per la gestione del manuale IFPUG
###############################################
def read_pdf_and_chunk(pdf_path, chunk_size=500):
    logger.info(f"Lettura del PDF: {pdf_path}")
    with open(pdf_path, "rb") as f:
        reader = PdfReader(f)
        all_text = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                all_text.append(page_text.strip())
    big_text = "\n".join(all_text)
    chunks = [big_text[i:i + chunk_size] for i in range(0, len(big_text), chunk_size)]
    logger.info(f"Suddiviso in {len(chunks)} chunk.")
    return chunks

def build_faiss_index(chunks, model,
                      faiss_index_path="manual_damas.index",
                      embeddings_path="manual_damas.npy"):
    if os.path.exists(faiss_index_path) and os.path.exists(embeddings_path):
        logger.info("Caricamento FAISS index ed embeddings da cache.")
        embeddings = np.load(embeddings_path)
        faiss_index = faiss.read_index(faiss_index_path)
        return faiss_index
    else:
        logger.info("Cache non trovata, creazione indice FAISS.")
        embeddings = model.encode(chunks)
        np.save(embeddings_path, embeddings)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings.astype("float32"))
        faiss.write_index(index, faiss_index_path)
        return index

def retrieve_context(query, faiss_index, chunks, model, k=1):
    logger.info("Recupero contesto tramite FAISS.")
    query_emb = model.encode([query]).astype("float32")
    distances, indices = faiss_index.search(query_emb, k)
    relevant_chunks = [chunks[idx] for idx in indices[0] if idx < len(chunks)]
    context = "\n".join(relevant_chunks)
    if len(context) > 2000:
        context = context[:2000]
    return context


###############################################
# Funzioni di supporto: pre-analisi e clamp
###############################################
def quick_pre_analysis(requirements_text):
    lines = requirements_text.splitlines()
    rf_count = sum(1 for line in lines if "RF" in line)
    text_lower = requirements_text.lower()
    keywords = {
        "nomine": 2, "matching": 2, "cas": 1, "sas": 1,
        "backlog": 2, "gui": 1, "db": 1,
        "design thinking": 2, "microservizi": 2
    }
    score = sum(text_lower.count(kw) * weight for kw, weight in keywords.items()) + rf_count
    if score < 5:
        suggested_range = "20–30"
    elif score < 10:
        suggested_range = "25–35"
    elif score < 15:
        suggested_range = "30–40"
    elif score < 20:
        suggested_range = "35–45"
    else:
        suggested_range = "40–50"
    return f"Numero di requisiti (RF) trovati: {rf_count}\nScore totale: {score}\nRange ipotizzato: {suggested_range}\n"

def clamp_range_in_text(answer, min_val, max_val):
    pattern = re.compile(r'Totale UFP\s*=\s*(\d+)')
    matches = pattern.findall(answer)
    if matches:
        total_fp = int(matches[0])
        total_fp = max(min_val, min(total_fp, max_val))
        answer = re.sub(r'Totale UFP\s*=\s*\d+', f'Totale UFP = {total_fp}', answer)
    return answer


###############################################
# Generazione della Specifica Funzionale
###############################################
def generate_fp_estimate_detailed(requirements_text, manual_context, pre_analysis_text, ufp_info):
    prompt = f"""[ARU Da Analizzare]
Requisiti funzionali (già estratti):
{requirements_text}

Informazioni utili e Sommario (già estratti):
{ufp_info}

PRE-ANALISI AUTOMATICA:
{pre_analysis_text}

ESTRATTO MANUALE IFPUG:
{manual_context}

Richiesta:
1) Genera un documento di Specifica Funzionale (SF) completo (almeno 3-4 pagine) utile al calcolo dei Function Point (IFPUG) seguendo questa struttura:
   - Introduzione: contesto, obiettivi, committente, vincoli di pianificazione.
   - Descrizione Generale del Sistema: ambito, utenti principali, interfacce con altri sistemi.
   - Definizione dei Boundary del Sistema: confini interni ed esterni, specificando gli archivi logici interni (ILF) e quelli esterni (EIF).
   - Requisiti Non Funzionali: prestazioni, sicurezza, usabilità, affidabilità, manutenibilità, portabilità.
   - Regole di Business: regole che influenzano operazioni e calcoli.
   - Eccezioni e Condizioni Speciali.
   - Report e Output del Sistema: contenuto, formato, frequenza.
   - Interfacce Utente: schermate e interazioni.
   - Processi di Interfacciamento con Altri Sistemi: flusso dati e interfacce.
   - Casi d'Uso e Scenari Operativi.
   - Dettagli sull'Architettura del Sistema.
   - Allegati e Appendici (Glossario, Diagrammi, Prototipi).
   - Associa a ogni step delle considerazioni su come calcoli EI, EO, EQ e rispettivi DET e FRT e su come consideri EIF e ILF e come arrivi ai rispettivi RET e DET
2) Se il contenuto supera la capacità di una singola risposta, dividilo in più parti.
3) Integra le informazioni già estratte senza duplicazioni.
"""
    messages = [
        {"role": "system", "content": (
            "Sei un esperto di Function Point Analysis (IFPUG). "
            "Quando rispondi, integra le informazioni già estratte e segui uno schema di calcolo passo-passo. "
            "Fornisci il documento di Specifica Funzionale completo, con il totale dei Function Point alla fine, "
            "che rientri nel range desiderato."
        )},
        {"role": "user", "content": prompt}
    ]
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                engine=DEPLOYMENT_NAME,
                messages=messages,
                max_tokens=3000,
                temperature=0.0,
                top_p=1.0,
                presence_penalty=0.0,
                frequency_penalty=0.0
            )
            answer = response["choices"][0]["message"]["content"].strip()
            answer = clamp_range_in_text(answer, 20, 50)
            logger.info("Documento SF generato con successo.")
            return answer
        except Exception as e:
            logger.error(f"Errore durante la richiesta OpenAI (tentativo {attempt+1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                logger.critical("Numero massimo di tentativi raggiunto. Interruzione del processo.")
                raise
    return "Nessuna risposta ottenuta."


###############################################
# Funzione principale per richiamare l'agente
###############################################
def run_agent(docx_path):
    pdf_path = "Function_Point_calcManual.pdf"  # Assicurati che il file PDF sia presente nella stessa cartella
    chunks = get_manual_chunks(pdf_path, chunk_size=500, cache_file="manual_chunks.pkl")
    logger.info("Caricamento del modello SentenceTransformer.")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    faiss_index = build_faiss_index(chunks, model)
    logger.info("Estrazione dei requisiti funzionali dal documento ARU.")
    requirements_text = get_functional_requirements(docx_path)
    ufp_info, _, short_summary = parse_aru_docx(docx_path)
    query = ("Recupera linee guida IFPUG su EI, EO, EQ, ILF, EIF e range complessità, "
             "così posso stimare i function points su una ARU con requisiti incompleti.")
    manual_context = retrieve_context(query, faiss_index, chunks, model, k=1)
    pre_analysis_txt = quick_pre_analysis(requirements_text)
    final_answer = generate_fp_estimate_detailed(requirements_text, manual_context, pre_analysis_txt, ufp_info)
    logger.info("Documento SF generato con successo.")
    return short_summary, final_answer


###############################################
# Funzione per ottenere i chunk dal PDF con caching
###############################################
def get_manual_chunks(pdf_path, chunk_size=500, cache_file="manual_chunks.pkl"):
    if os.path.exists(cache_file):
        logger.info(f"Caricamento dei chunk dal file di cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            chunks = pickle.load(f)
    else:
        logger.info("Cache non trovata, leggo e chunko il PDF manuale.")
        chunks = read_pdf_and_chunk(pdf_path, chunk_size)
        with open(cache_file, 'wb') as f:
            pickle.dump(chunks, f)
        logger.info(f"Cache salvata in {cache_file}.")
    return chunks


###############################################
# Main – Esecuzione dell'agente (per test)
###############################################
if __name__ == "__main__":
    # Modifica questo percorso con quello del tuo file .docx da analizzare
    docx_path = r"C:\Users\A395959\PycharmProjects\pyMilvus\ARU_inventate\aru_fittizia_GPT o1.docx"
    summary, spec = run_agent(docx_path)
    print("Sommario:", summary)
    print("Specifica Funzionale:", spec)
