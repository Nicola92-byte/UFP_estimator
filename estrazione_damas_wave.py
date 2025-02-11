import os
import tempfile
import re
from docx import Document
import openai
import easyocr
from dotenv import load_dotenv

# Carica le variabili d'ambiente dal file .env
load_dotenv()

# Configurazione OpenAI (senza fallback)
openai.api_type = os.getenv("OPENAI_API_TYPE")
openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_version = os.getenv("OPENAI_API_VERSION")
openai.api_key = os.getenv("OPENAI_API_KEY")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")

if not all([openai.api_type, openai.api_base, openai.api_version, openai.api_key, DEPLOYMENT_NAME]):
    raise ValueError("Una o più variabili d'ambiente non sono state impostate correttamente.")


# ==========================
# Funzioni di estrazione testo da docx
# ==========================
def extract_text_from_docx(docx_path):
    """
    Estrae testo dai paragrafi e dalle tabelle di un file DOCX.
    Ritorna il testo concatenato.
    """
    try:
        doc = Document(docx_path)
        extracted_text = []

        # Estrarre testo dai paragrafi
        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            if text:
                extracted_text.append(text)

        # Estrarre testo dalle tabelle
        for table in doc.tables:
            for row in table.rows:
                row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_text:
                    extracted_text.append(" | ".join(row_text))

        return "\n".join(extracted_text)
    except Exception as e:
        print(f"Errore durante l'estrazione del testo da docx: {e}")
        return ""

def extract_images_from_docx(docx_path, temp_dir):
    """
    Estrae tutte le immagini dal file .docx, salvandole nella cartella temp_dir.
    Restituisce la lista dei percorsi delle immagini estratte.
    """
    try:
        doc = Document(docx_path)
        image_paths = []

        for rel in doc.part._rels.values():
            if "image" in rel.target_ref:
                image_data = rel.target_part.blob
                image_filename = os.path.join(temp_dir, os.path.basename(rel.target_ref))
                with open(image_filename, "wb") as img_file:
                    img_file.write(image_data)
                image_paths.append(image_filename)

        return image_paths
    except Exception as e:
        print(f"Errore durante l'estrazione delle immagini: {e}")
        return []

def extract_text_from_images(image_paths):
    """
    Esegue OCR sulle immagini presenti in image_paths utilizzando EasyOCR.
    Ignora errori su immagini non leggibili.
    """
    try:
        reader = easyocr.Reader(['it', 'en'], gpu=False)
        ocr_texts = []
        for img_path in image_paths:
            try:
                result = reader.readtext(img_path, detail=0)
                ocr_texts.append(" ".join(result).strip())
            except Exception as e:
                print(f"Errore OCR su immagine {img_path}: {e}")
        return "\n".join(ocr_texts)
    except Exception as e:
        print(f"Errore OCR complessivo: {e}")
        return ""

def extract_all_content(docx_path):
    """
    Combina:
    1) testo da paragrafi e tabelle
    2) testo da immagini (via OCR con EasyOCR)
    Restituisce una singola stringa con tutto il contenuto.
    """
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            text_paragraphs_tables = extract_text_from_docx(docx_path)
            image_paths = extract_images_from_docx(docx_path, temp_dir)
            text_images = extract_text_from_images(image_paths)

        full_text = text_paragraphs_tables
        if text_images:
            full_text += "\n\n[Testo estratto da immagini]\n" + text_images

        # print("DEBUG: Testo estratto dal documento:\n", full_text[:1000])  # Mostra i primi 1000 caratteri

        return full_text
    except Exception as e:
        print(f"Errore durante l'estrazione del contenuto: {e}")
        return ""

def remove_index_from_text(full_text):
    """
    Rimuove le sezioni di testo che corrispondono a un indice.
    """
    try:
        # Rimuove righe che assomigliano a un indice (numeri di pagina, numerazione, ecc.)
        patterns = [
            re.compile(r"^\d+\.\s+.*\s+\d+$", re.MULTILINE),  # Formato: 1. Titolo 1
            re.compile(r"^Indice.*$", re.IGNORECASE | re.MULTILINE),  # Linee con "Indice"
            re.compile(r"^Sommario.*$", re.IGNORECASE | re.MULTILINE),  # Linee con "Sommario"
            re.compile(r"^\d+\s+[A-Za-z].*\d+$", re.MULTILINE),  # Formato: 1 Titolo 1
            re.compile(r"^[IVXLCDM]+\.\s+.*$", re.MULTILINE)  # Numeri romani: I. Titolo
        ]

        filtered_text = full_text
        for pattern in patterns:
            filtered_text = re.sub(pattern, "", filtered_text)

        return filtered_text
    except Exception as e:
        print(f"Errore durante la rimozione dell'indice: {e}")
        return full_text

def extract_functional_requirements_with_ai(full_text):
    """
    Utilizza un modello AI per isolare i "Requisiti Funzionali" dal testo completo.
    """
    try:
        system_message = (
            "Sei un assistente esperto nell'analisi dei documenti ARU. "
            "Estrai esclusivamente la sezione 'Requisiti Funzionali' come presente nel testo, "
            "senza aggiungere contenuti non esplicitamente presenti. Includi tutti i dettagli pertinenti. "
            "Assicurati di escludere l'indice o qualsiasi contenuto simile a un sommario."
        )

        # Suddividere il contenuto in parti più piccole per evitare troncamento
        parts = [full_text[i:i+3000] for i in range(0, len(full_text), 3000)]
        extracted_sections = []

        for part in parts:
            response = openai.ChatCompletion.create(
                engine=DEPLOYMENT_NAME,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": part}
                ],
                max_tokens=3000,  # Incrementato per documenti complessi
                temperature=0.0,
                top_p=1.0
            )
            extracted_sections.append(response['choices'][0]['message']['content'].strip())

        return "\n".join(extracted_sections)
    except Exception as e:
        print(f"Errore durante l'estrazione AI dei requisiti funzionali: {e}")
        return "Errore AI durante l'estrazione."

def get_functional_requirements(docx_path):
    """
    Data la path di un file docx, estrae e restituisce la sezione
    'Requisiti Funzionali' come stringa.
    """
    if not os.path.exists(docx_path):
        print(f"Errore: il file {docx_path} non esiste o non è accessibile.")
        return ""

    full_content = extract_all_content(docx_path)
    filtered_content = remove_index_from_text(full_content)
    functional_requirements = extract_functional_requirements_with_ai(filtered_content)

    return functional_requirements

if __name__ == "__main__":

    docx_path = r"C:\Users\A395959\PycharmProjects\pyMilvus\ARU_dir\ARU-Mercato-Re-factoringDamas(Analisi&DesignSprint17-18)_20240725103817.490_X.docx"  # Sostituire con il percorso reale del file
    # # docx_path = r"C:\Users\A395959\PycharmProjects\pyMilvus\ARU_dir\ARU - STL 2023 Wave 1.docx"
    #
    # functional_requirements = get_functional_requirements(docx_path)
    #
    # print("=== REQUISITI FUNZIONALI ESTRATTI ===")
    # print(functional_requirements)
