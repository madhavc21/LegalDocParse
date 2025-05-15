# main.py
import logging
from pathlib import Path
import shutil
import uuid
import json

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse

# Import your processing modules
# Assuming they are in the same directory or accessible via PYTHONPATH
import document_parser
import metadata_extractor

# Configure logging for the API
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PDF Document Ingestion Pipeline",
    description="Accepts a PDF, extracts content and legal metadata.",
    version="0.1.0"
)

# --- Configuration ---
# Define directories for uploads and outputs
# These should ideally be configurable (e.g., via environment variables) in a production setup
# For this assignment, hardcoding is acceptable for simplicity.
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploaded_pdfs"
OUTPUT_DIR = BASE_DIR / "processed_outputs"
TEMP_PROCESSING_DIR = BASE_DIR / "temp_processing_workspace" # For intermediate files from document_parser

# Ensure directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_PROCESSING_DIR.mkdir(parents=True, exist_ok=True)


@app.on_event("startup")
async def startup_event():
    """
    Event handler for application startup.
    Placeholder for any initial setup, like loading models if not done globally in modules.
    Our models (spaCy) are loaded when metadata_extractor.py is imported.
    """
    logger.info("Application startup: Checking if SpaCy models are loaded...")
    if not metadata_extractor.MODELS_LOADED:
        logger.warning(
            "One or more SpaCy models failed to load during module import. "
            "API functionality for metadata extraction will be impaired."
        )
    else:
        logger.info("SpaCy models appear to be loaded.")


@app.get("/health", summary="Health Check", status_code=status.HTTP_200_OK)
async def health_check():
    """
    Simple health check endpoint to confirm the API is running.
    """
    return {"status": "healthy", "message": "API is up and running."}


@app.post("/ingest", summary="Ingest and Process PDF", status_code=status.HTTP_200_OK)
@app.post("/ingest", summary="Ingest and Process PDF", status_code=status.HTTP_200_OK)
async def ingest_pdf(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No file provided.")
    
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid file type. Only PDF files are accepted.")

    unique_id = uuid.uuid4().hex
    pdf_filename_stem = Path(file.filename).stem
    uploaded_pdf_path = UPLOAD_DIR / f"{pdf_filename_stem}_{unique_id}.pdf"
    
    request_temp_dir = TEMP_PROCESSING_DIR / unique_id
    output_json_for_file_save_path = OUTPUT_DIR / f"{pdf_filename_stem}_{unique_id}_processed_output.json"

    try:
        with open(uploaded_pdf_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Uploaded PDF saved to: {uploaded_pdf_path}")

        logger.info(f"Starting content extraction for {file.filename}...")
        request_temp_dir.mkdir(parents=True, exist_ok=True)
        structured_content_list = document_parser.extract_structured_content_from_pdf(
            pdf_filepath=uploaded_pdf_path,
            working_dir=request_temp_dir 
        )

        if structured_content_list is None: # Or potentially check if it's an empty list and handle
            logger.error(f"Content extraction failed or yielded no data for {file.filename}.")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to extract content from PDF.")
        logger.info(f"Content extraction successful for {file.filename}.")

        logger.info(f"Starting metadata extraction for {file.filename}...")
        extracted_metadata_object = metadata_extractor.extract_document_metadata(
            structured_content=structured_content_list, # Corrected variable name
            doc_name=pdf_filename_stem
        )
        logger.info(f"Metadata extraction successful for {file.filename}.")

        # This is the data that will be returned directly in the API response
        api_response_data = {
            "content": structured_content_list, # Corrected variable name
            "metadata": extracted_metadata_object
        }

        # For saving to a file, we can include more audit info if desired
        file_save_data = {
            "source_pdf_filename": file.filename,
            "processing_id": unique_id,
            **api_response_data # Embed the content and metadata
        }
        
        with open(output_json_for_file_save_path, "w", encoding="utf-8") as f:
            json.dump(file_save_data, f, indent=2)
        logger.info(f"Full processed output saved to file: {output_json_for_file_save_path}")

        # --- THIS IS THE FIX: Return the actual data ---
        return api_response_data 

    except HTTPException as http_exc:
        # Re-raise HTTPException to let FastAPI handle it
        raise http_exc
    except Exception as e:
        logger.error(f"An error occurred during PDF processing for {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An internal error occurred: {str(e)}")
    finally:
        # Clean up the uploaded PDF file
        if uploaded_pdf_path.exists():
            try:
                uploaded_pdf_path.unlink()
                logger.info(f"Cleaned up uploaded PDF: {uploaded_pdf_path}")
            except Exception as e_clean:
                logger.error(f"Error cleaning up uploaded PDF {uploaded_pdf_path}: {e_clean}")
        
        # Clean up the temporary processing directory for this request
        if 'request_temp_dir' in locals() and request_temp_dir.exists():
            try:
                shutil.rmtree(request_temp_dir)
                logger.info(f"Cleaned up temporary processing directory: {request_temp_dir}")
            except Exception as e_clean_temp:
                logger.error(f"Error cleaning up temporary directory {request_temp_dir}: {e_clean_temp}")


if __name__ == "__main__":
    import uvicorn
    # This is for local development. For deployment, you'd use a production server like Gunicorn with Uvicorn workers.
    logger.info("Starting Uvicorn server for local development...")
    uvicorn.run(app, host="0.0.0.0", port=8000)