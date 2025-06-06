# document_parser.py
import logging
from pathlib import Path
import json
from bs4 import BeautifulSoup 

# Docling imports
from docling_core.types.doc import ImageRefMode 
from docling.datamodel.base_models import InputFormat 
from docling.datamodel.pipeline_options import PdfPipelineOptions 
from docling.document_converter import DocumentConverter, PdfFormatOption 

logger = logging.getLogger(__name__)

IMAGE_RESOLUTION_SCALE = 2.0 # Default image scale for docling

def generate_html_from_pdf(pdf_path: Path, assets_output_dir: Path) -> Path | None:
    """
    Converts a PDF document to an HTML file using Docling.
    HTML and its assets (e.g., images) are saved in assets_output_dir.

    Args:
        pdf_path: Path to the input PDF file.
        assets_output_dir: Directory to save the HTML file and its artifacts.

    Returns:
        Path to the generated HTML file, or None if conversion fails.
    """
    logger.info(f"Generating HTML for: {pdf_path.name}")
    assets_output_dir.mkdir(parents=True, exist_ok=True)

    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
    pipeline_options.generate_page_images = False # Typically not needed for structured extraction
    pipeline_options.generate_picture_images = True # To get image data for figures

    doc_converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

    try:
        conv_res = doc_converter.convert(pdf_path)
        if not conv_res or not conv_res.document:
            logger.error(f"Docling conversion failed for {pdf_path.name}.")
            return None
    except Exception as e:
        logger.error(f"Exception during Docling conversion for {pdf_path.name}: {e}")
        return None

    doc_filename_stem = conv_res.input.file.stem
    # HTML file saved in assets_output_dir, images in a subfolder like _artifacts
    html_filepath = assets_output_dir / f"{doc_filename_stem}_docling.html"

    try:
        conv_res.document.save_as_html(html_filepath, image_mode=ImageRefMode.REFERENCED)
        logger.info(f"Docling HTML saved to: {html_filepath}")
        return html_filepath
    except Exception as e:
        logger.error(f"Error saving document as HTML for {pdf_path.name}: {e}")
        return None

def parse_document_html_to_structured_content(html_filepath: Path) -> list:
    """
    Parses an HTML file (presumably generated by Docling or similar PDF-to-HTML tool)
    into a list of structured content elements (paragraphs, tables, figures, etc.).

    Args:
        html_filepath: Path to the HTML file to parse.

    Returns:
        A list of dictionaries, where each dictionary represents a content element.
    """
    logger.info(f"Parsing HTML: {html_filepath} for structured content.")
    if not html_filepath.exists():
        logger.error(f"HTML file not found: {html_filepath}")
        return []

    with open(html_filepath, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'lxml')

    structured_elements = []
    current_page_number = 1

    if not soup.body:
        logger.error("HTML <body> tag not found in {html_filepath}. Cannot parse content.")
        return []

    relevant_tags = ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'ul', 'ol', 'table', 'figure', 'hr']
    
    all_found_elements = soup.body.find_all(relevant_tags, recursive=True) # Find all, not just direct children

    for element in all_found_elements:
        if any(parent.name in ['table', 'figure'] for parent in element.find_parents(limit=3) if parent != element):
            if element.name not in ['li', 'img', 'figcaption']: 
                continue
        
        item_data = None
        item_type = None

        if element.name == 'hr' and 'page-break' in element.get('class', []):
            current_page_number += 1
            continue

        text_content = element.get_text(separator=' ', strip=True)

        if element.name == 'p':
            if text_content:
                item_type = "paragraph"
                item_data = {"text": text_content}
        
        elif element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            if text_content:
                item_type = "header" # Document structural heading
                item_data = {"text": text_content}

        elif element.name in ['ul', 'ol']:
            for li in element.find_all('li', recursive=False):
                li_text = li.get_text(separator=' ', strip=True)
                if li_text:
                    structured_elements.append({
                        "type": "list_item", 
                        "text": li_text,
                        "page_number": current_page_number,
                        "metadata": {}
                    })
            continue 

        elif element.name == 'table':
             # Avoid processing tables nested within other tables directly if the outer one is already being processed
            is_nested_table = False
            parent_table = element.find_parent('table')
            if parent_table:
                is_nested_table = True
            
            if not is_nested_table:
                table_html_str = str(element)
                if text_content or "<td>" in table_html_str.lower(): # Ensure it's a content table
                    item_type = "table"
                    item_data = {"text": text_content, "html": table_html_str}

        elif element.name == 'figure':
            img_tag = element.find('img')
            caption_tag = element.find('figcaption')
            
            img_filename_in_html = ""
            if img_tag and img_tag.get('src'):
                img_filename_in_html = Path(img_tag['src']).name 
            
            caption_text = caption_tag.get_text(strip=True) if caption_tag else ""
            
            if img_filename_in_html:
                item_type = "figure"
                item_data = {
                    "image_filename": img_filename_in_html,
                    "caption": caption_text,
                }
        
        if item_type and item_data is not None:

            is_duplicate = False
            if structured_elements:
                last_el = structured_elements[-1]
                if last_el.get("text") == item_data.get("text") and \
                   last_el.get("page_number") == current_page_number and \
                   last_el.get("type") == item_type:
                    is_duplicate = True

            if not is_duplicate:
                entry = {
                    "type": item_type,
                    **item_data,
                    "page_number": current_page_number,
                    "metadata": {} # Bounding boxes are optional and not easily extracted from HTML
                }
                structured_elements.append(entry)

    return structured_elements

def extract_structured_content_from_pdf(pdf_filepath: Path, working_dir: Path) -> list | None:
    """
    High-level function to extract structured content from a PDF.
    It first converts PDF to HTML, then parses the HTML.

    Args:
        pdf_filepath: Path to the input PDF file.
        working_dir: A directory to store intermediate files (HTML, assets).

    Returns:
        A list of structured content elements, or None on failure.
    """
    logger.info(f"Starting content extraction for PDF: {pdf_filepath.name}")
    html_assets_dir = working_dir / f"{pdf_filepath.stem}_html_assets"
    
    generated_html_path = generate_html_from_pdf(pdf_filepath, html_assets_dir)
    
    if generated_html_path and generated_html_path.exists():
        structured_content = parse_document_html_to_structured_content(generated_html_path)
        logger.info(f"Successfully extracted {len(structured_content)} structured elements from {pdf_filepath.name}")
        return structured_content
    else:
        logger.error(f"Failed to generate or find HTML for {pdf_filepath.name}. Cannot extract content.")
        return None