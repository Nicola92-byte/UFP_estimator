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
from dotenv import load_dotenv

# Le tue due funzioni di estrazione
from estrazione_damas_wave import get_functional_requirements
# from estrazione_dati_utili_wave import parse_aru_docx
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

    # Ecco le regole "generiche" per non fondere funzionalità.
    # Più generali, senza casi specifici come "Gestione Anagrafiche" o "File CSV".
    heuristics = """
Regole Generali di Separazione Funzionalità:
1) Non unire MAI in un'unica voce funzioni distinte se i requisiti le menzionano separatamente (evita fusioni).
2) Se ci sono più tipologie di input (file diversi, ecc.), enumerale come EI separati.
3) Se ci sono più output (report, log, dashboard), enumerali come EO distinti.
4) Non calcolare AFP.
5) Elenca sempre tutti gli EI, EO, EQ, ILF, EIF specificando se sono EI, EO, EQ, ILF o EIF e specificando i rispettivi DET e FRT per gli EI, EO e EQ e i rispettivi DET e RET ILF e EIF
6) Concludi con la tabella Markdown:
   ## Riepilogo e Calcolo Totale UFP
   | Tipo            | Nome | Complessità | Peso | Totale           |
   ...
  
   **Totale UFP = X**
7) Per la colonna Tipo l'ordine di elencazione deve essere sempre EI,EO,EQ, ILF, EIF
8) Se i requisiti non dicono che due funzioni siano la stessa, trattale come separate (evita "perdita di informazione").
9) Mantieni un ordine coerente (EI, EO, EQ, ILF, EIF) ma senza unire funzioni.
10) Se ci sono più EI, EO, EQ, ILF, EIF non devono mai essere considerati un'unico elemento, vanno sempre presi separati sia nel Report e Output del Sistema che nel Riepilogo e Calcolo Totale UFP
11) Se l'ARU (o i requisiti) menziona più sorgenti esterne (ad esempio “Sorgente A”, “Sorgente B”...)
    e l'utente/business le riconosce come sistemi/autori differenti con scopi e dati distinti,
    allora tali sorgenti vanno sempre considerate come EIF separati.
    Non fonderle mai in un unico EIF a meno che il documento non precisi espressamente
    che si tratta di un unico archivio logico.
12) "Se un requisito descrive più modalità o varianti di estrazione (ad esempio diverse tipologie di filtro, 
    output, finalità o parametri significativi) che per l’utente si concretizzano in funzioni distinte, 
    allora tali modalità vanno sempre classificate come EQ separate. 
    Non unire mai estrazioni diverse in un’unica EQ, a meno che non venga esplicitamente detto 
    che si tratta di un'unica funzione con un semplice parametro di input."
13) Ogni procedura o flusso di acquisizione che riguardi fonti di dati differenti, obiettivi o anagrafiche diverse, o che avvenga in momenti/eventi separati, deve essere considerata una funzione elementare autonoma e pertanto classificata come un External Input (EI) separato. In nessun caso vanno unificate più acquisizioni in un unico EI, a meno che non si tratti esplicitamente di un’unica transazione con un unico trigger comune.
14) “Ogni elaborazione o visualizzazione che presenti differenze significative ai fini del business o dell’utente (ad esempio calcoli specifici, logiche di formattazione, scopi di presentazione, input o parametri diversi, filtri, viste, layout, fasce temporali, destinazioni di output) deve essere sempre classificata come una transazione EO separata. Non è consentito riunire in un singolo EO funzioni che, dal punto di vista dell’utente, si manifestano come output differenti o con logiche/elaborazioni distinte.

In particolare, se un requisito descrive più modalità di output (ad es. un calcolo X e un calcolo Y che l’utente richiama separatamente, oppure viste/grafici diversi per eolico/fotovoltaico/COI, o ancora report differenziati per finalità), tali modalità costituiscono transazioni EO autonome e non vanno unificate, a meno che la documentazione non attesti esplicitamente che si tratta della stessa identica funzione con un semplice parametro aggiuntivo, privo di differenze di elaborazione o formattazione.”

15)"Ogni uscita che comporti calcoli (somma, differenza, stima, formattazione per aree, colorazione, ecc.) deve essere considerata External Output (EO). Di conseguenza, la rappresentazione del fotovoltaico (rilevante + non rilevante) e l’heatmap eolica (con raggruppamento per province) costituiscono funzioni EO distinte e non vanno fuse con la semplice visualizzazione del COI, anch’essa un EO autonomo.”

"""

    # Questo blocco descrive la "struttura" 3-4 pagine + regole "dividi se supera token" + "integra info"
    request_structure = """
Richiesta:
1) Genera un documento di Specifica Funzionale (SF) completo (almeno 3-4 pagine) utile al calcolo dei Function Point (IFPUG), seguendo questa struttura:
   - Introduzione: contesto, obiettivi, committente, vincoli di pianificazione.
   - Descrizione Generale del Sistema: ambito, utenti principali, interfacce con altri sistemi.
   - Definizione dei Boundary del Sistema: confini interni ed esterni (ILF, EIF).
   - Requisiti Non Funzionali: prestazioni, sicurezza, usabilità, affidabilità, manutenibilità, portabilità.
   - Regole di Business
   - Eccezioni e Condizioni Speciali
   - Report e Output del Sistema: contenuto, formato, frequenza.
   - Interfacce Utente
   - Processi di Interfacciamento con Altri Sistemi
   - Casi d'Uso e Scenari Operativi
   - Dettagli sull'Architettura del Sistema
   - Allegati e Appendici (Glossario, Diagrammi, Prototipi)
   - Per ogni step, spiega come calcoli EI, EO, EQ (DET, FTR) e come consideri ILF/EIF (RET, DET).
2) Se il contenuto supera la capacità di una singola risposta, dividilo in più parti.
3) Integra le informazioni già estratte senza duplicazioni.
4) Non menzionare gli AFP (Adjusted Function Points). Concludi con una tabella di calcolo UFP (EI, EO, EQ, ILF, EIF).
5) Se i requisiti non dicono che due funzioni siano la stessa, trattale come separate (evita "perdita di informazione").
6) Mantieni un ordine coerente (EI, EO, EQ, ILF, EIF) ma senza unire funzioni.
7) Se ci sono più EI, EO, EQ, ILF, EIF non devono mai essere considerati un'unica cosa, vanno sempre presi separati sia nel Report e Output del Sistema che nel Riepilogo e Calcolo Totale UFP
8) Se l'ARU (o i requisiti) menziona più sorgenti esterne (ad esempio “Sorgente A”, “Sorgente B”...)
    e l'utente/business le riconosce come sistemi/autori differenti con scopi e dati distinti,
    allora tali sorgenti vanno sempre considerate come EIF separati.
    Non fonderle mai in un unico EIF a meno che il documento non precisi espressamente
    che si tratta di un unico archivio logico.
9) "Se un requisito descrive più modalità o varianti di estrazione (ad esempio diverse tipologie di filtro, 
    output, finalità o parametri significativi) che per l’utente si concretizzano in funzioni distinte, 
    allora tali modalità vanno sempre classificate come EQ separate. 
    Non unire mai estrazioni diverse in un’unica EQ, a meno che non venga esplicitamente detto 
    che si tratta di un'unica funzione con un semplice parametro di input."
10) Ogni procedura o flusso di acquisizione che riguardi fonti di dati differenti, obiettivi o anagrafiche diverse, o che avvenga in momenti/eventi separati, deve essere considerata una funzione elementare autonoma e pertanto classificata come un External Input (EI) separato. In nessun caso vanno unificate più acquisizioni in un unico EI, a meno che non si tratti esplicitamente di un’unica transazione con un unico trigger comune.
11) “Ogni elaborazione o visualizzazione che presenti differenze significative ai fini del business o dell’utente (ad esempio calcoli specifici, logiche di formattazione, scopi di presentazione, input o parametri diversi, filtri, viste, layout, fasce temporali, destinazioni di output) deve essere sempre classificata come una transazione EO separata. Non è consentito riunire in un singolo EO funzioni che, dal punto di vista dell’utente, si manifestano come output differenti o con logiche/elaborazioni distinte.

In particolare, se un requisito descrive più modalità di output (ad es. un calcolo X e un calcolo Y che l’utente richiama separatamente, oppure viste/grafici diversi per eolico/fotovoltaico/COI, o ancora report differenziati per finalità), tali modalità costituiscono transazioni EO autonome e non vanno unificate, a meno che la documentazione non attesti esplicitamente che si tratta della stessa identica funzione con un semplice parametro aggiuntivo, privo di differenze di elaborazione o formattazione.”
12) "Ogni uscita che comporti calcoli (somma, differenza, stima, formattazione per aree, colorazione, ecc.) deve essere considerata External Output (EO). Di conseguenza, la rappresentazione del fotovoltaico (rilevante + non rilevante) e l’heatmap eolica (con raggruppamento per province) costituiscono funzioni EO distinte e non vanno fuse con la semplice visualizzazione del COI, anch’essa un EO autonomo.”


"""

    # Mettiamo insieme i vari pezzi in un "prompt"
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
        # Esegui la chiamata a OpenAI
        response = openai.ChatCompletion.create(
            engine=DEPLOYMENT_NAME,
            messages=messages,
            max_tokens=4000,   # Aumenta se serve un testo più lungo
            temperature=0.0,
            top_p=1.0,
            presence_penalty=0.0,
            frequency_penalty=0.0
        )
        answer = response["choices"][0]["message"]["content"].strip()

        # Se vuoi forzare un range di TOT UFP, es. [20..200], usa clamp_range_in_text
        # (Assumendo che clamp_range_in_text sia definita altrove)
        answer = clamp_range_in_text(answer, 20, 200)

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
    2) build FAISS
    3) Estrae requisiti .docx
    4) Ottiene contesto
    5) genera il testo con 'heuristics'
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

    # # ✅ Modifica per restituire TUTTE le variabili richieste
    # return {
    #     "pre_analysis_text": pre_analysis_txt,
    #     "requirements_text": req_text,
    #     "ufp_info": ufp_info,
    #     "manual_context": manual_context,
    #     "summary": short_summary,
    #     "final_text": final_text
    # }


###############################################################################
# MAIN
###############################################################################
if __name__ == "__main__":
    # Sostituisci con il tuo file .docx
    # docx_path = r"C:\Users\A395959\PycharmProjects\pyMilvus\ARU_inventate\aru_fittizia_GPT o1.docx"
    # docx_path = r"C:\Users\A395959\PycharmProjects\pyMilvus\ARU_dir\ARU-Mercato-Re-factoringDamas(Analisi&DesignSprint17-18)_20240725103817.490_X.docx"
    docx_path = r"C:\Users\A395959\PycharmProjects\pyMilvus\ARU_dir\ARU -Inerzia 2.1 Evolutive 2022 Fase 1 20220331.docx"

    summary, spec, ufp_info = run_agent(docx_path)

    print("\n=== SHORT SUMMARY ARU ===\n")
    print(summary)

  
    print("\n=== SPECIFICA FUNZIONALE + TABELLA UFP ===\n")
    print(spec)



