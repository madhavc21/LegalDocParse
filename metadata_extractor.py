# metadata_extractor.py
import logging
from pathlib import Path
import json
import dateparser # type: ignore
import spacy # type: ignore
from collections import defaultdict
import re
from datetime import datetime

logger = logging.getLogger(__name__)

# --- Load SpaCy Models (load once globally) ---
NLP_GENERAL = None
NLP_LEGAL_NER = None
MODELS_LOADED = True

try:
    NLP_GENERAL = spacy.load("en_core_web_lg")
    logger.info("SpaCy 'en_core_web_lg' (general NER) loaded.")
except OSError:
    logger.error("'en_core_web_lg' not found. Person extraction will be limited/skipped.")
    MODELS_LOADED = False
try:
    NLP_LEGAL_NER = spacy.load("en_legal_ner_trf") # Assumes en_legal_ner_trf is installed
    logger.info("SpaCy 'en_legal_ner_trf' (legal NER) loaded.")
except OSError:
    logger.error("'en_legal_ner_trf' not found. Legal entity & NER-date extraction will be limited/skipped.")
    MODELS_LOADED = False

# --- Helper Functions ---
def standardize_date_format(date_obj: datetime) -> str | None:
    return date_obj.strftime('%Y-%m-%d') if date_obj else None

def extract_text_context(full_text: str, matched_text_fragment: str, window_chars: int = 50) -> str:
    try:
        start_idx = full_text.lower().find(matched_text_fragment.lower())
        if start_idx == -1: # If exact fragment not found (e.g. due to normalization)
             # Try to find a snippet around the middle of the text as a last resort
            mid_point = len(full_text) // 2
            snippet_start = max(0, mid_point - window_chars)
            snippet_end = min(len(full_text), mid_point + window_chars)
            return f"...{full_text[snippet_start:snippet_end]}..." if snippet_start > 0 or snippet_end < len(full_text) else full_text[snippet_start:snippet_end]

        context_start = max(0, start_idx - window_chars)
        context_end = min(len(full_text), start_idx + len(matched_text_fragment) + window_chars)
        
        prefix = "..." if context_start > 0 else ""
        suffix = "..." if context_end < len(full_text) else ""
        return f"{prefix}{full_text[context_start:context_end]}{suffix}"
    except Exception:
        # Fallback for any unexpected error
        return full_text[:window_chars * 2 + len(matched_text_fragment)]


# --- Regex for Letters (as Legal NER might not cover specific letter formats well) ---
LETTER_REFERENCE_PATTERNS = [
    re.compile(r"\b(Letter\s*(?:No\.?|Ref\.?)\s*[:\-]?\s*[\w\d\.\-\/()]+)\b", re.IGNORECASE),
    re.compile(r"\b(Ref[:\.]\s*[\w\d\.\-\/()]+)\b", re.IGNORECASE),
    re.compile(r"\b((?:our|your|their|his|her)\s+letter\s+(?:dated|of)\s+[\w\s,\d\-\.]+)\b", re.IGNORECASE)
]

def extract_document_metadata(
    structured_content: list, # From document_parser.py
    doc_name: str = "Unknown Document"
) -> dict:
    """
    Extracts structured metadata (dates, references, persons) from parsed document content.
    Relies primarily on NLP_LEGAL_NER for dates and legal entities.

    Args:
        structured_content: A list of dictionaries, each representing a content element
                            (e.g., paragraph) with "text" and "page_number" keys.
        doc_name: The name of the document being processed.

    Returns:
        A dictionary containing the extracted metadata.
    """
    if not MODELS_LOADED:
        logger.warning("One or more SpaCy models failed to load. Metadata extraction quality will be affected.")

    # Initialize metadata stores
    all_extracted_dates_ner = [] # Dates from Legal NER's "DATE" entities
    persons_identified = defaultdict(list)
    letter_refs_identified = defaultdict(list)
    legal_refs_identified = defaultdict(lambda: defaultdict(list)) # type: {reference: [page_numbers]}

    # 1. Person Extraction (using General NER)
    if NLP_GENERAL:
        texts_for_person_ner = [(item.get("text", ""), {"page": item.get("page_number", 0)})
                                for item in structured_content if item.get("text")]
        if texts_for_person_ner:
            for doc, context in NLP_GENERAL.pipe(texts_for_person_ner, as_tuples=True):
                page = context["page"]
                for ent in doc.ents:
                    if ent.label_ == "PERSON":
                        persons_identified[ent.text.strip()].append(page)
    
    # 2. Legal Entities (including Dates) from Legal NER and Letters via Regex
    if NLP_LEGAL_NER:
        texts_for_legal_ner = [(item.get("text", ""), {"page": item.get("page_number", 0)})
                               for item in structured_content if item.get("text")]
        if texts_for_legal_ner:
            logger.info(f"Processing {len(texts_for_legal_ner)} text blocks with Legal NER.")
            for doc, context in NLP_LEGAL_NER.pipe(texts_for_legal_ner, as_tuples=True):
                text_block, page = doc.text, context["page"]
                for ent in doc.ents:
                    ent_text, ent_label = ent.text.strip(), ent.label_
                    ent_text_lower = ent_text.lower()
                    target_type = None

                    if ent_label == "STATUTE": target_type = "act"
                    elif ent_label == "PROVISION":
                        if "clause" in ent_text_lower: target_type = "clause"
                        elif "article" in ent_text_lower: target_type = "article"
                        else: target_type = "clause" # Default for Section, Rule, etc.
                    elif ent_label == "PRECEDENT": target_type = "precedent"
                    elif ent_label == "DATE":
                        try:
                            dt = dateparser.parse(ent_text, languages=['en'], settings={'STRICT_PARSING': False})
                            if dt and 1800 <= dt.year <= (datetime.now().year + 10): # Plausibility check
                                std_date = standardize_date_format(dt)
                                if std_date:
                                    all_extracted_dates_ner.append({
                                        "date_obj": dt, "date_str": std_date,
                                        "context": extract_text_context(text_block, ent_text),
                                        "page_number": page, "original_text_fragment": ent_text
                                    })
                        except Exception: pass # Ignore if date parsing fails for an NER entity
                        continue # Processed as DATE

                    if target_type and ent_text:
                        legal_refs_identified[target_type][ent_text].append(page)
    else: # Fallback if Legal NER is not available (limited extraction)
        logger.warning("Legal NER model not available. Date and legal reference extraction will be minimal or skipped.")
        # As a minimal fallback for dates if NLP_LEGAL_NER is missing, one *could* run dateparser.search_dates
        # but per our findings, it's very noisy without extensive filtering.
        # For this "confident" version, we rely on NER for dates.

    # 3. Letter References (Regex)
    for item in structured_content:
        text, page = item.get("text", ""), item.get("page_number", 0)
        if not text: continue
        for pattern in LETTER_REFERENCE_PATTERNS:
            for match in pattern.finditer(text):
                name = match.group(1).strip().rstrip('.')
                if name: letter_refs_identified[name].append(page)

    # --- Finalize and Structure Metadata ---
    # Dates from NER (deduplicated and sorted)
    unique_dates_map = {}
    for d_info in all_extracted_dates_ner:
        key = (d_info["date_str"], d_info["page_number"], d_info["original_text_fragment"])
        if key not in unique_dates_map: unique_dates_map[key] = d_info
    final_processed_dates = sorted(list(unique_dates_map.values()), key=lambda x: (x["date_obj"], x["page_number"]))

    document_date_val = None
    if final_processed_dates:
        # Heuristic for document date (e.g., earliest date on first page with "dated" context)
        # This still needs refinement for high accuracy.
        page1_dates_with_keyword = [d for d in final_processed_dates if d["page_number"] == 1 and "dated" in d["context"].lower()]
        target_dates_for_doc = page1_dates_with_keyword if page1_dates_with_keyword else \
                              [d for d in final_processed_dates if d["page_number"] <= 3] if final_processed_dates else \
                              final_processed_dates
        if target_dates_for_doc:
            target_dates_for_doc.sort(key=lambda x: x["date_obj"])
            document_date_val = target_dates_for_doc[0]["date_str"]
            logger.info(f"Selected document_date: {document_date_val} from context: '{target_dates_for_doc[0]['context']}'")


    output_formatted_dates = [{"date": d["date_str"], "surrounding_context": d["context"]} for d in final_processed_dates]

    # Helper for formatting entities with unique page numbers
    def _format_entity_list(items_dict: defaultdict) -> list:
        res = []
        for name, pages in items_dict.items():
            if pages:
                unique_sorted_pages = sorted(list(set(pages)))
                res.append({"name": name, "page_number": unique_sorted_pages[0]})
        return sorted(res, key=lambda x: (x["page_number"], x["name"]))

    final_persons = _format_entity_list(persons_identified)
    final_letters = _format_entity_list(letter_refs_identified)
    
    final_legal_refs = []
    for type_val, refs_dict in legal_refs_identified.items():
        for ref_text, pages in refs_dict.items():
            if pages and type_val in ["act", "clause", "article", "precedent"]:
                unique_sorted_pages = sorted(list(set(pages)))
                final_legal_refs.append({
                    "reference": ref_text, "type": type_val, "page_number": unique_sorted_pages[0]
                })
    final_legal_refs.sort(key=lambda x: (x["page_number"], x.get("type",""), x["reference"]))

    return {
        "document_name": doc_name,
        "document_date": document_date_val,
        "dates": output_formatted_dates,
        "references": {
            "letters_mentioned": final_letters,
            "laws_clauses_articles_acts": final_legal_refs,
            "persons": final_persons
        }
    }

if __name__ == '__main__':
    # Example Usage (for testing this module directly)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

    # This requires a JSON file from document_parser.py (M1 output)
    # Example: RFD_extracted_content.json (if you ran document_parser.py's example)
    example_m1_content_path = Path("./temp_processing_output/RFD_extracted_content.json") # Adjust if needed

    if not example_m1_content_path.exists():
        logger.error(f"Example M1 content JSON not found at {example_m1_content_path}. Skipping example run.")
        # Create minimal dummy content for basic script test if models loaded
        if MODELS_LOADED:
             logger.warning("Using minimal dummy content for metadata extraction test.")
             example_structured_content = [
                {"text": "This agreement dated 1st January 2023 between Mr. Foo Bar and Ms. Alice Wonder.", "page_number": 1},
                {"text": "Refer to Clause 5 of the Services Act.", "page_number": 1},
                {"text": "Our letter Ref: XYZ/123 was sent.", "page_number": 2}
            ]
             doc_name_for_test = "dummy_metadata_test"
        else:
            example_structured_content = []
            doc_name_for_test = "models_not_loaded_test"

    else:
        with open(example_m1_content_path, "r", encoding="utf-8") as f:
            m1_output_data = json.load(f)
        example_structured_content = m1_output_data.get("document_content", [])
        doc_name_for_test = example_m1_content_path.stem.replace("_extracted_content", "")


    if example_structured_content:
        logger.info(f"Running example metadata extraction for: {doc_name_for_test}")
        extracted_meta = extract_document_metadata(example_structured_content, doc_name=doc_name_for_test)
        
        output_meta_path = Path(f"./{doc_name_for_test}_extracted_metadata.json")
        with open(output_meta_path, "w", encoding="utf-8") as f:
            json.dump({"document_metadata": extracted_meta}, f, indent=2) # Changed key for clarity
        logger.info(f"Example extracted metadata saved to: {output_meta_path}")
    else:
        logger.warning("No content provided for example metadata extraction.")