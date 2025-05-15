import logging
from pathlib import Path
import json
import dateparser
import spacy 
from collections import defaultdict
import re
from datetime import datetime

logger = logging.getLogger(__name__)

NLP_LEGAL_NER = None
MODELS_LOADED = True

try:
    NLP_LEGAL_NER = spacy.load("en_legal_ner_trf")
    logger.info("SpaCy 'en_legal_ner_trf' (legal NER) loaded successfully.")
    logger.info(f"Available pipes in NLP_LEGAL_NER: {NLP_LEGAL_NER.pipe_names}")
except OSError:
    logger.error(
        "'en_legal_ner_trf' not found or failed to load. "
        "Please ensure it's installed correctly (e.g., via its .whl file). "
        "Metadata extraction will be severely impacted."
    )
    MODELS_LOADED = False
except Exception as e:
    logger.error(f"An unexpected error occurred while loading 'en_legal_ner_trf': {e}")
    MODELS_LOADED = False

def standardize_date_format(date_obj: datetime) -> str | None:
    return date_obj.strftime('%Y-%m-%d') if date_obj else None

def extract_text_context(full_text: str, matched_text_fragment: str, window_chars: int = 50) -> str:
    try:
        start_idx = full_text.lower().find(matched_text_fragment.lower())
        if start_idx == -1:
            mid_point = len(full_text) // 2
            snippet_start = max(0, mid_point - (window_chars + len(matched_text_fragment)//2))
            snippet_end = min(len(full_text), mid_point + (window_chars + len(matched_text_fragment)//2))
            text_to_show = full_text[snippet_start:snippet_end]
            prefix = "..." if snippet_start > 0 else ""
            suffix = "..." if snippet_end < len(full_text) else ""
            return f"{prefix}{text_to_show}{suffix}"

        context_start = max(0, start_idx - window_chars)
        context_end = min(len(full_text), start_idx + len(matched_text_fragment) + window_chars)
        prefix = "..." if context_start > 0 else ""
        suffix = "..." if context_end < len(full_text) else ""
        return f"{prefix}{full_text[context_start:context_end]}{suffix}"
    except Exception:
        return full_text[:min(len(full_text), window_chars * 2 + len(matched_text_fragment))]

LETTER_REFERENCE_PATTERNS = [
    re.compile(r"\b(Letter\s*(?:No\.?|Ref\.?)\s*[:\-]?\s*[\w\d\.\-\/()]+)\b", re.IGNORECASE),
    re.compile(r"\b(Ref[:\.]\s*[\w\d\.\-\/()]+)\b", re.IGNORECASE),
    re.compile(r"\b((?:our|your|their|his|her)\s+letter\s+(?:dated|of)\s+[\w\s,\d\-\.]+)\b", re.IGNORECASE)
]

LEGAL_NER_PERSON_LABELS_MAP = {
    "JUDGE": "judges",
    "LAWYER": "lawyers",
    "OTHER_PERSON": "other_persons",
    "PETITIONER": "petitioners",
    "RESPONDENT": "respondents",
    "WITNESS": "witnesses",
}
ALL_LEGAL_NER_PERSON_LABELS = list(LEGAL_NER_PERSON_LABELS_MAP.keys())


def extract_document_metadata(
    structured_content: list,
    doc_name: str = "Unknown Document"
) -> dict:
    if not MODELS_LOADED or not NLP_LEGAL_NER:
        logger.critical(
            "NLP_LEGAL_NER model is not loaded. Metadata extraction cannot proceed effectively. "
            "Returning empty/minimal metadata."
        )
        return {
            "document_name": doc_name, "document_date": None, "dates": [],
            "references": {"letters_mentioned": [], "laws_clauses_articles_acts": []},
            "persons_by_role": {role: [] for role in LEGAL_NER_PERSON_LABELS_MAP.values()},
            "error_message": "Critical NLP model (en_legal_ner_trf) not loaded."
        }

    all_extracted_dates_from_ner = []
    # Store persons by role: e.g., {'judges': [{'name': 'X', 'page': 1}], 'lawyers': [...]}
    persons_by_role_identified = defaultdict(lambda: defaultdict(list))
    letter_refs_identified = defaultdict(list)
    legal_refs_identified = defaultdict(lambda: defaultdict(list))

    texts_for_processing = [(item.get("text", ""), {"page": item.get("page_number", 0)})
                             for item in structured_content if item.get("text")]

    if texts_for_processing:
        logger.info(f"Processing {len(texts_for_processing)} text blocks with NLP_LEGAL_NER.")
        for doc, context_dict in NLP_LEGAL_NER.pipe(texts_for_processing, as_tuples=True):
            text_block, page_num = doc.text, context_dict["page"]

            for ent in doc.ents:
                ent_text, ent_label = ent.text.strip(), ent.label_
                ent_text_lower = ent_text.lower()
                target_type = None

                if ent_label in ALL_LEGAL_NER_PERSON_LABELS:
                    role_key = LEGAL_NER_PERSON_LABELS_MAP.get(ent_label, "other_persons")
                    if len(ent_text) > 1 and not ent_text.isdigit():
                        persons_by_role_identified[role_key][ent_text].append(page_num)
                    continue
                elif ent_label == "DATE":
                    try:
                        dt_obj = dateparser.parse(ent_text, languages=['en'], settings={'STRICT_PARSING': False})
                        if dt_obj and 1800 <= dt_obj.year <= (datetime.now().year + 10):
                            std_date_str = standardize_date_format(dt_obj)
                            if std_date_str:
                                all_extracted_dates_from_ner.append({
                                    "date_obj": dt_obj, "date_str": std_date_str,
                                    "context": extract_text_context(text_block, ent_text),
                                    "page_number": page_num, "original_text_fragment": ent_text
                                })
                    except Exception as e:
                        logger.debug(f"Could not parse DATE entity '{ent_text}' from Legal NER: {e}")
                    continue
                elif ent_label == "STATUTE": target_type = "act"
                elif ent_label == "PROVISION":
                    if "clause" in ent_text_lower: target_type = "clause"
                    elif "article" in ent_text_lower: target_type = "article"
                    else: target_type = "clause"
                elif ent_label == "PRECEDENT": target_type = "precedent"
                
                if target_type and ent_text:
                    legal_refs_identified[target_type][ent_text].append(page_num)
    else:
        logger.info("No text blocks found in structured_content to process with NLP_LEGAL_NER.")

    for item in structured_content:
        text, page_num = item.get("text", ""), item.get("page_number", 0)
        if not text: continue
        for pattern in LETTER_REFERENCE_PATTERNS:
            for match in pattern.finditer(text):
                letter_name = match.group(1).strip().rstrip('.')
                if letter_name: letter_refs_identified[letter_name].append(page_num)

    unique_dates_map = {}
    for d_info in all_extracted_dates_from_ner:
        key = (d_info["date_str"], d_info["page_number"], d_info["original_text_fragment"])
        if key not in unique_dates_map: unique_dates_map[key] = d_info
    final_processed_dates = sorted(list(unique_dates_map.values()), key=lambda x: (x["date_obj"], x["page_number"]))

    document_date_val = None
    if final_processed_dates:
        page1_to_3_dates = [d for d in final_processed_dates if d["page_number"] <= 3]
        dated_keyword_dates = [d for d in page1_to_3_dates if "dated" in d["context"].lower()]
        target_dates_for_doc_heuristic = dated_keyword_dates if dated_keyword_dates else \
                                          page1_to_3_dates if page1_to_3_dates else \
                                          final_processed_dates
        if target_dates_for_doc_heuristic:
            target_dates_for_doc_heuristic.sort(key=lambda x: x["date_obj"])
            document_date_val = target_dates_for_doc_heuristic[0]["date_str"]
            logger.info(f"Selected document_date: {document_date_val} from context: '{target_dates_for_doc_heuristic[0]['context']}'")

    output_formatted_dates = [{"date": d["date_str"], "surrounding_context": d["context"]} for d in final_processed_dates]

    def _format_entity_list_with_name(items_dict: defaultdict) -> list:
        res = []
        for name, pages in items_dict.items():
            if pages:
                unique_sorted_pages = sorted(list(set(pages)))
                res.append({"name": name, "page_number": unique_sorted_pages[0]})
        return sorted(res, key=lambda x: (x["page_number"], x["name"]))

    final_persons_by_role = {}
    for role, names_dict in persons_by_role_identified.items():
        final_persons_by_role[role] = _format_entity_list_with_name(names_dict)
        
    final_letters = _format_entity_list_with_name(letter_refs_identified)
    
    final_legal_refs = []
    for type_val, refs_dict in legal_refs_identified.items():
        for ref_text, pages in refs_dict.items():
            if pages and type_val in ["act", "clause", "article", "precedent"]:
                unique_sorted_pages = sorted(list(set(pages)))
                final_legal_refs.append({
                    "reference": ref_text, "type": type_val, "page_number": unique_sorted_pages[0]
                })
    final_legal_refs.sort(key=lambda x: (x["page_number"], x.get("type","default_type"), x["reference"]))

    return {
        "document_name": doc_name,
        "document_date": document_date_val,
        "dates": output_formatted_dates,
        "references": {
            "letters_mentioned": final_letters,
            "laws_clauses_articles_acts": final_legal_refs
        },
        "persons_by_role": final_persons_by_role
    }

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
    
    example_m1_content_path = Path("RFD_m1_content.json")

    if not example_m1_content_path.exists():
        logger.error(f"Test M1 content JSON not found at {example_m1_content_path}.")
        if MODELS_LOADED and NLP_LEGAL_NER:
             logger.warning("Using minimal dummy content for metadata extraction test.")
             test_structured_content = [
                {"text": "The case was heard by Judge Smith. Mr. John Doe (PETITIONER) was represented by Lawyer Alice. The State (RESPONDENT) had Witness Bob.", "page_number": 1},
                {"text": "Other_Person Eve observed. This is Act of 1999 and Clause 1. Refer to our letter Ref: ABC.", "page_number": 1},
            ]
             test_doc_name = "dummy_detailed_persons_test"
        else:
            logger.critical("Cannot run test with dummy data as NLP_LEGAL_NER model is not loaded.")
            test_structured_content = []
            test_doc_name = "models_not_loaded_test"
    else:
        try:
            with open(example_m1_content_path, "r", encoding="utf-8") as f:
                m1_output_data = json.load(f)
            test_structured_content = m1_output_data.get("document_content") or m1_output_data.get("content")
            if not test_structured_content:
                logger.error(f"No 'document_content' or 'content' key found in {example_m1_content_path}")
                test_structured_content = []
            test_doc_name = example_m1_content_path.stem.replace("_extracted_content", "").replace("_m1_content", "")
        except Exception as e:
            logger.error(f"Error loading or parsing M1 content from {example_m1_content_path}: {e}")
            test_structured_content = []
            test_doc_name = "error_loading_m1_content"

    if test_structured_content and MODELS_LOADED and NLP_LEGAL_NER:
        logger.info(f"Running example metadata extraction for: {test_doc_name}")
        extracted_meta = extract_document_metadata(test_structured_content, doc_name=test_doc_name)
        
        output_dir = Path("./test_metadata_outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_meta_path = output_dir / f"{test_doc_name}_extracted_metadata_detailed_persons.json"
        
        with open(output_meta_path, "w", encoding="utf-8") as f:
            json.dump({"document_metadata": extracted_meta}, f, indent=2)
        logger.info(f"Example extracted metadata saved to: {output_meta_path}")
    elif not (MODELS_LOADED and NLP_LEGAL_NER):
        logger.error("Skipping metadata extraction example because essential NLP models are not loaded.")
    else:
        logger.warning("No content provided or found for example metadata extraction.")