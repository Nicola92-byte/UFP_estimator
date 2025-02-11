# UFP_estimator

Function Point Estimator è un'applicazione web realizzata con Streamlit che analizza un file DOCX contenente l'Analisi Requisiti Utente (ARU) e genera automaticamente una Specifica Funzionale (SF) completa. L'app stima inoltre i Function Point secondo gli standard IFPUG, mostrando un sommario e la stima finale.

![Banner](https://upload.wikimedia.org/wikipedia/commons/thumb/3/3d/Chart_line_graph_icon.svg/1024px-Chart_line_graph_icon.svg.png)

## Caratteristiche

- **Upload facile:** Carica file DOCX contenenti l'ARU.
- **Analisi automatica:** Estrae i requisiti funzionali e le informazioni utili.
- **Generazione SF:** Produce una Specifica Funzionale completa (minimo 3-4 pagine).
- **Stima dei Function Point:** Calcola e visualizza il totale dei Function Point secondo gli standard IFPUG.
- **Interfaccia moderna e accattivante:** Con toni di blu e grafiche, per rendere l'esperienza utente piacevole.
- **Logging avanzato:** Implementato in `agente_logging.py` per facilitare il debug e il monitoraggio.

## Tecnologie Utilizzate

- [Streamlit](https://streamlit.io/) – Interfaccia utente web.
- [OpenAI API](https://openai.com/api/) – Generazione automatica della Specifica Funzionale.
- [Sentence Transformers](https://www.sbert.net/) – Generazione di embeddings.
- [FAISS](https://github.com/facebookresearch/faiss) – Ricerca di similarità per il manuale IFPUG.
- [PyPDF2](https://pypi.org/project/PyPDF2/) – Estrazione del testo dai PDF.
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) – OCR per estrarre testo dalle immagini in documenti DOCX.
- [python-docx](https://python-docx.readthedocs.io/) – Estrazione del testo dai file DOCX.
- [python-dotenv](https://pypi.org/project/python-dotenv/) – Caricamento delle variabili d'ambiente.

## Prerequisiti

- **Python 3.7+**
- **Git**

## Installazione

1. **Clona il repository:**

   ```bash
   git clone https://github.com/tuo-username/function-point-estimator.git
   cd function-point-estimator

2. **Crea un ambiente virtuale (opzionale ma consigliato):**
3. 

