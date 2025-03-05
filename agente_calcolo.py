import os
import re
import pickle
import logging
import numpy as np
import faiss
import openai
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from dotenv import load_dotenv

# Le tue due funzioni di estrazione
from estrazione_damas_wave import get_functional_requirements
from estrazione_dati_utili_wave import parse_aru_docx

###############################################################################
# Caricamento ENV + Config
###############################################################################
load_dotenv()

openai.api_type = os.getenv("OPENAI_API_TYPE")
openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_version = os.getenv("OPENAI_API_VERSION")
openai.api_key = os.getenv("OPENAI_API_KEY")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")

if not openai.api_key or not DEPLOYMENT_NAME:
    raise ValueError("OPENAI_API_KEY / DEPLOYMENT_NAME non impostate in .env")

def init_logger(log_file='app.log'):
    logger = logging.getLogger('FP_Calc')
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger

logger = init_logger()

###############################################################################
# Lettura PDF e FAISS
###############################################################################
def read_pdf_and_chunk(pdf_path, chunk_size=500):
    logger.info(f"Lettura PDF: {pdf_path}")
    with open(pdf_path, "rb") as f:
        reader = PdfReader(f)
        all_txt = []
        for page in reader.pages:
            ptxt = page.extract_text()
            if ptxt:
                all_txt.append(ptxt.strip())
    big_txt = "\n".join(all_txt)
    chunks = [big_txt[i:i + chunk_size] for i in range(0, len(big_txt), chunk_size)]
    logger.info(f"Suddiviso in {len(chunks)} chunk da ~{chunk_size} char.")
    return chunks

def build_faiss_index(chunks, model,
                      faiss_index_path="manual_damas.index",
                      embeddings_path="manual_damas.npy"):
    if os.path.exists(faiss_index_path) and os.path.exists(embeddings_path):
        logger.info("Caricamento FAISS index + embeddings da cache.")
        emb = np.load(embeddings_path)
        faiss_index = faiss.read_index(faiss_index_path)
        return faiss_index
    else:
        logger.info("Creazione indice FAISS ex novo.")
        emb = model.encode(chunks)
        idx = faiss.IndexFlatL2(emb.shape[1])
        idx.add(emb.astype("float32"))
        faiss.write_index(idx, faiss_index_path)
        np.save(embeddings_path, emb)
        return idx

def retrieve_context(query, faiss_index, chunks, model, k=1):
    logger.info(f"Recupero contesto con FAISS (k={k}).")
    qemb = model.encode([query]).astype("float32")
    dist, idxs = faiss_index.search(qemb, k)
    relevant = [chunks[i] for i in idxs[0] if i < len(chunks)]
    context = "\n".join(relevant)
    if len(context) > 2000:
        context = context[:2000]
    return context

def get_manual_chunks(pdf_path="Function_Point_calcManual.pdf", chunk_size=500, cache_file="manual_chunks.pkl"):
    if os.path.exists(cache_file):
        logger.info(f"Caricamento chunk da {cache_file}")
        with open(cache_file, 'rb') as f:
            c = pickle.load(f)
        return c
    else:
        c = read_pdf_and_chunk(pdf_path, chunk_size)
        with open(cache_file, 'wb') as f:
            pickle.dump(c, f)
        return c

###############################################################################
# Support: clamp Totale UFP e pre-analisi
###############################################################################
def clamp_range_in_text(answer, min_val, max_val):
    pattern = re.compile(r'Totale UFP\s*=\s*(\d+)')
    matches = pattern.findall(answer)
    if matches:
        old_str = matches[-1]  # prendi l'ultimo match
        val = int(old_str)
        if val < min_val:
            val = min_val
        elif val > max_val:
            val = max_val
        answer = re.sub(r'Totale UFP\s*=\s*\d+', f'Totale UFP = {val}', answer)
        logger.info(f"Clamp del Totale UFP: da {old_str} a {val} (range {min_val}-{max_val}).")
    return answer

def quick_pre_analysis(req_text):
    lines = req_text.splitlines()
    # contiamo quante "RF"
    rf_count = sum(1 for ln in lines if "RF" in ln)
    return f"Trovati {rf_count} requisiti con label 'RF'."

###############################################################################
# Nuove funzioni per il contesto Agile
###############################################################################
def is_agile_context(requirements_text):
    """Verifica se nel testo sono presenti parole chiave di contesto Agile."""
    agile_keywords = ['product backlog', 'sprint', 'agile', 'metodologia agile']
    requirements_lower = requirements_text.lower()
    return any(keyword in requirements_lower for keyword in agile_keywords)

def adjust_ufp_for_agile(answer, requirements_text, reduction_factor=0.4):
    """
    Se il contesto è agile, riduce il valore 'Totale UFP' applicando il reduction_factor.
    Per una riduzione del 60% il reduction_factor deve essere 0.4 (ossia mantieni il 40% del valore originale).
    """
    if is_agile_context(requirements_text):
        pattern = r"Totale UFP\s*=\s*(\d+)"
        match = re.search(pattern, answer)
        if match:
            original_ufp = int(match.group(1))
            new_ufp = int(original_ufp * reduction_factor)
            answer = re.sub(pattern, f"Totale UFP = {new_ufp}", answer)
            logger.info(f"Contesto Agile rilevato. Riduzione UFP: da {original_ufp} a {new_ufp} (riduzione del 60%).")
    return answer

###############################################################################
# Funzione Principale di generazione (con EURIStiche)
###############################################################################
def generate_fp_estimate_text_heuristics(requirements_text, ufp_info, pre_analysis_text, manual_context):
    """
    Genera un testo unificato che includa:
      1) Un documento di Specifica Funzionale (almeno 3-4 pagine) con la struttura:
         - Introduzione (contesto, obiettivi, committente, vincoli)
         - Descrizione Generale del Sistema (ambito, utenti, interfacce)
         - Definizione dei Boundary del Sistema (ILF, EIF)
         - Requisiti Non Funzionali
         - Regole di Business
         - Eccezioni e Condizioni Speciali
         - Report e Output del Sistema
         - Interfacce Utente
         - Processi di Interfacciamento con altri sistemi
         - Casi d'Uso e Scenari Operativi
         - Dettagli sull'Architettura
         - Allegati e Appendici (Glossario, Diagrammi, Prototipi)
         - ... e considerazioni su EI, EO, EQ, DET, FTR e su come calcoli ILF/EIF
         (Se supera i token, dividere in più parti)
         (Integrare info estratte senza duplicazioni)
      2) Il calcolo dei Function Point (solo UFP): EI, EO, EQ, ILF, EIF, con DET/FTR/RET, complessità, peso, totali parziali.
      3) Una tabella Markdown finale:
           ## Riepilogo e Calcolo Totale UFP
           | Tipo | Nome | Complessità | Peso | Totale |
           ...
           **Totale UFP = X**
      4) Applicazione di "euristiche generali" per non fondere funzionalità distinte e non perdere informazioni.
      5) Non calcolare né menzionare in alcun modo AFP.
    """

    # TABELLE DI COMPLESSITÀ IFPUG (incollate nel prompt)
    complexity_tables = """
    TABELLE DI RIFERIMENTO IFPUG (semplificate per EI, EO, EQ, ILF, EIF):

    1) Valutazione ILF/EIF (complessità) in base a DET e RET:
          Data Element Types
    RET    1-19   20-50   >=51
    --------------------------
     1     Basso  Basso   Medio
    2-5    Basso  Medio   Alto
    >5     Medio  Alto    Alto

    2) Valutazione External Input (EI) in base a DET e FTR:
           Data Element Types
    FTR     1-4   5-15   >=16
    -------------------------
    <2      Basso Basso  Medio
     2      Basso Medio  Alto
    >2      Medio Alto   Alto

    3) Valutazione External Output (EO) in base a DET e FTR:
           Data Element Types
    FTR     1-5   6-19   >=20
    -------------------------
    <2      Basso Basso  Medio
    2-3     Basso Medio  Alto
    >3      Medio Alto   Alto

    4) Valutazione External Inquiry (EQ) in base a DET e FTR:
           Data Element Types
    FTR     1-5   6-19   >=20
    -------------------------
    <2      Basso Basso  Medio
    2-3     Basso Medio  Alto
    >3      Medio Alto   Alto
    """

    # Regole generali per non fondere funzionalità
    heuristics = """
Regole Generali di Separazione Funzionalità:
1) Non unire MAI in un'unica voce funzioni distinte se i requisiti le menzionano separatamente (evita fusioni).
2) Se ci sono più tipologie di input (file diversi, ecc.), enumerale come EI separati.
3) Se ci sono più output (report, log, dashboard), enumerali come EO distinti.
4) Non calcolare AFP.
5) Elenca sempre tutti gli EI, EO, EQ, ILF, EIF specificando se sono EI, EO, EQ, ILF o EIF e specificando i rispettivi DET e FRT per gli EI, EO e EQ e i rispettivi DET e RET ILF e EIF.
6) Concludi con la tabella Markdown:
   ## Riepilogo e Calcolo Totale UFP
   | Tipo            | Nome | Complessità | Peso | Totale           |
   ...
   **Totale UFP = X**
7) Per la colonna Tipo l'ordine di elencazione deve essere sempre EI, EO, EQ, ILF, EIF.
8) Se i requisiti non dicono che due funzioni siano la stessa, trattale come separate (evita "perdita di informazione").
9) Mantieni un ordine coerente (EI, EO, EQ, ILF, EIF) ma senza unire funzioni.
10) Se ci sono più EI, EO, EQ, ILF, EIF non devono mai essere considerati un'unica cosa, vanno sempre presi separati.
11) Se l'ARU (o i requisiti) menziona più sorgenti esterne (es. “Sorgente A”, “Sorgente B”),
    e l'utente/business le riconosce come sistemi/autori differenti, allora tali sorgenti vanno sempre considerate come EIF separati.
12) Se un requisito descrive più modalità o varianti di estrazione, tali modalità vanno sempre classificate come EQ separate.
13) Ogni flusso di acquisizione di dati differenti deve essere considerato un External Input (EI) separato.
14) Ogni elaborazione o visualizzazione che presenti differenze significative deve essere classificata come una transazione EO separata.
15) Ogni uscita che comporti calcoli deve essere considerata External Output (EO).
    """

    # Struttura richiesta
    request_structure = """
Richiesta:
1) Genera un documento di Specifica Funzionale (SF) completo (almeno 3-4 pagine) utile al calcolo dei Function Point (IFPUG), seguendo questa struttura:
   - Introduzione: contesto, obiettivi, committente, vincoli di pianificazione.
   - Descrizione Generale del Sistema: ambito, utenti principali, interfacce.
   - Definizione dei Boundary del Sistema: confini interni ed esterni (ILF, EIF).
   - Requisiti Non Funzionali: prestazioni, sicurezza, usabilità, affidabilità, manutenibilità, portabilità.
   - Regole di Business.
   - Eccezioni e Condizioni Speciali.
   - Report e Output del Sistema: contenuto, formato, frequenza.
   - Interfacce Utente.
   - Processi di Interfacciamento con altri sistemi.
   - Casi d'Uso e Scenari Operativi.
   - Dettagli sull'Architettura.
   - Allegati e Appendici (Glossario, Diagrammi, Prototipi).
   - Spiega come calcoli EI, EO, EQ (DET, FTR) e come consideri ILF/EIF (RET, DET).
2) Se il contenuto supera la capacità di una singola risposta, dividilo in più parti.
3) Integra le informazioni già estratte senza duplicazioni.
4) Non menzionare AFP. Concludi con una tabella di calcolo UFP (EI, EO, EQ, ILF, EIF).
5) Tratta come separate funzioni che non sono esplicitamente indicate come uguali.
6) Mantieni l'ordine EI, EO, EQ, ILF, EIF.
7) Concludi con la tabella in Markdown e "**Totale UFP = X**".
    """

    prompt = f"""
Sei un esperto di Function Point Analysis (IFPUG).

Ecco alcune indicazioni di contesto:

[PRE-ANALISI AUTOMATICA]
{pre_analysis_text}

[Requisiti Funzionali Estratti]
{requirements_text}

[UFN Info e Sommario]
{ufp_info}

[Estratto Manuale IFPUG]
{manual_context}

[TABELLE IFPUG]
{complexity_tables}

============================================================
{heuristics}

{request_structure}

Alla fine:
- Mostra il calcolo di EI, EO, EQ, ILF, EIF con DET/FTR, complessità, peso.
- Produci la tabella in Markdown, seguita da "**Totale UFP = X**".
- Niente AFP.
    """

    messages = [
        {
            "role": "system",
            "content": (
                "Sei un analista FP IFPUG: quando rispondi, devi creare un testo discorsivo completo "
                "come da struttura indicata, applicando le regole di separazione delle funzionalità. "
                "NON menzionare AFP, concludi con la tabella e Totale UFP."
            )
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    try:
        response = openai.ChatCompletion.create(
            engine=DEPLOYMENT_NAME,
            messages=messages,
            max_tokens=4000,
            temperature=0.0,
            top_p=1.0,
            presence_penalty=0.0,
            frequency_penalty=0.0
        )
        answer = response["choices"][0]["message"]["content"].strip()

        # Applica un clamp sul valore UFP se necessario (es. range 20-200)
        answer = clamp_range_in_text(answer, 20, 200)
        # Se il contesto dei requisiti indica un approccio Agile, riduci il totale UFP del 60%
        answer = adjust_ufp_for_agile(answer, requirements_text)

        return answer

    except Exception as e:
        logger.error(f"Errore generate_fp_estimate_text_heuristics: {e}")
        return "Errore nella generazione."

###############################################################################
# Pipeline Principale
###############################################################################
def run_agent(docx_path):
    """
    1) Carica e chunk PDF
    2) Costruisce l'indice FAISS
    3) Estrae requisiti dal file .docx
    4) Ottiene contesto dal manuale PDF
    5) Genera il testo completo con euristiche
    """
    pdf_path = "Function_Point_calcManual.pdf"
    chunks = get_manual_chunks(pdf_path, 500, "manual_chunks.pkl")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    idx = build_faiss_index(chunks, model)

    query = "Tabelle IFPUG e calcolo EI, EO, EQ, ILF, EIF"
    manual_context = retrieve_context(query, idx, chunks, model, k=2)

    logger.info("Estrazione requisiti dal docx.")
    req_text = get_functional_requirements(docx_path)
    ufp_info, _, short_summary = parse_aru_docx(docx_path)

    pre_analysis_txt = quick_pre_analysis(req_text)

    print("\n=== DEBUG ===")
    print(f"Manual Context: {manual_context}")
    # print(f"Pre-Analysis: {pre_analysis_txt}")
    # print(f"Functional Requirements:\n{req_text}")
    # print(f"UFP Info:\n{ufp_info}")

    final_text = generate_fp_estimate_text_heuristics(
        requirements_text=req_text,
        ufp_info=ufp_info,
        pre_analysis_text=pre_analysis_txt,
        manual_context=manual_context
    )

    return short_summary, final_text

###############################################################################
# MAIN
###############################################################################
if __name__ == "__main__":
    # Sostituisci con il percorso del file .docx da elaborare
    docx_path = r"C:\Users\A395959\PycharmProjects\pyMilvus\ARU_dir\ARU-Mercato-Re-factoringDamas(Analisi&DesignSprint17-18)_20240725103817.490_X.docx"


    summary, spec= run_agent(docx_path)

    print("\n=== SHORT SUMMARY ARU ===\n")
    print(summary)


    print("\n=== SPECIFICA FUNZIONALE + TABELLA UFP ===\n")
    print(spec)
