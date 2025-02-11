
#########################################################################################################################################
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// #
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// #
#########################################################################################################################################

import os
import tempfile
import re
import openai
from docx import Document
from easyocr import Reader
from dotenv import load_dotenv

# Carica le variabili d'ambiente dal file .env
load_dotenv()

# Configurazione OpenAI (senza fallback)
openai.api_type = os.getenv("OPENAI_API_TYPE")
openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_version = os.getenv("OPENAI_API_VERSION")
openai.api_key = os.getenv("OPENAI_API_KEY")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")
# ==========================
# Funzioni di supporto
# ==========================
def extract_text_from_docx(docx_path):
    """Estrae testo dai paragrafi e tabelle di un file DOCX."""
    doc = Document(docx_path)
    text = []

    for paragraph in doc.paragraphs:
        text.append(paragraph.text.strip())

    for table in doc.tables:
        for row in table.rows:
            row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if row_text:
                text.append(" | ".join(row_text))

    return "\n".join([t for t in text if t])


def extract_images_from_docx(docx_path):
    """Estrae immagini da un file DOCX e le salva temporaneamente."""
    image_paths = []
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            doc = Document(docx_path)
            for rel in doc.part.rels.values():
                try:
                    if "image" in rel.target_ref:
                        image_data = rel.target_part.blob
                        image_name = os.path.basename(rel.target_ref)
                        image_path = os.path.join(temp_dir, image_name)
                        with open(image_path, "wb") as f:
                            f.write(image_data)
                        image_paths.append(image_path)
                except Exception as e:
                    print(f"Immagine non valida ignorata: {e}")
        except Exception as e:
            print(f"Errore durante l'estrazione delle immagini: {e}")
    return image_paths


def ocr_on_images(image_paths):
    """Utilizza EasyOCR per estrarre testo dalle immagini."""
    reader = Reader(["it", "en"], gpu=False)
    extracted_texts = []
    for img_path in image_paths:
        try:
            results = reader.readtext(img_path, detail=0)
            extracted_texts.append("\n".join(results))
        except Exception as e:
            print(f"Errore OCR su immagine {img_path}: {e}")
    return "\n".join(extracted_texts)


def call_azure_openai(content, system_prompt, user_prompt):
    """Invia richieste a Azure OpenAI."""
    try:
        response = openai.ChatCompletion.create(
            engine=DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt.format(content=content)}
            ],
            temperature=0.0,
            max_tokens=2000
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"Errore durante la richiesta OpenAI: {e}")
        return "Errore durante l'elaborazione."


def extract_functional_requirements_locally(text):
    """Estrae la sezione 'Requisiti Funzionali' dal testo fornito."""
    pattern = re.compile(
        r"(?i)(?:^|\n)\s*\d*\.?\s*REQUISITI\s+FUNZIONALI(?:\s*:)?\s*\n(.*?)(?=\n\s*\d*\.?\s*(REQUISITI\s+NON\s+FUNZIONALI|$))",
        re.DOTALL
    )
    match = pattern.search(text)
    return match.group(1).strip() if match else "Sezione 'Requisiti Funzionali' non trovata."


def parse_aru_docx(docx_path):
    """Estrae informazioni utili da un file ARU."""
    text = extract_text_from_docx(docx_path)
    images = extract_images_from_docx(docx_path)
    ocr_text = ocr_on_images(images) if images else ""
    full_text = text + ("\n\n[TESTO DA IMMAGINI]\n" + ocr_text if ocr_text else "")

    # Prompt per estrarre dati utili
    ufp_prompt = (
        "Testo ARU:\n{content}\n\nEstrai informazioni utili al calcolo degli Unadjusted Function Points (UFP)."
    )
    ufp_info = call_azure_openai(full_text, "Assistente UFP", ufp_prompt)

    # Estrarre requisiti funzionali
    functional_requirements = extract_functional_requirements_locally(full_text)

    # Prompt per generare riassunto
    summary_prompt = (
        "Testo ARU:\n{content}\n\nGenera un breve riassunto del documento ARU, includendo il tipo di progetto."
    )
    summary = call_azure_openai(full_text, "Assistente Riassunto ARU", summary_prompt)

    return ufp_info, functional_requirements, summary


if __name__ == "__main__":
    # Percorso del file DOCX
    # docx_path = r"C:\Users\A395959\PycharmProjects\pyMilvus\ARU_dir\ARU -Inerzia 2.1 Evolutive 2022 Fase 1 20220331.docx"  # Sostituire con il percorso reale del file
    docx_path = r"C:\Users\A395959\PycharmProjects\pyMilvus\ARU_dir\ARU - STL 2023 Wave 1.docx"
    # docx_path = r"C:\Users\A395959\PycharmProjects\pyMilvus\ARU_dir\ARU-Mercato-Re-factoringDamas(Analisi&DesignSprint17-18)_20240725103817.490_X.docx"
    # docx_path = r"C:\Users\A395959\PycharmProjects\pyMilvus\ARU_dir\ARU_Alina2.0 Fase1_2023.docx"

    ufp_info, functional_requirements, summary = parse_aru_docx(docx_path)



    # print("=== INFO UFP ===")
    # print(ufp_info)
    #
    # # print("\n=== REQUISITI FUNZIONALI ===")
    # # print(functional_requirements)
    #
    # print("\n=== RIASSUNTO ARU ===")
    # print(summary)

