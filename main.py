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
async def ingest_pdf(file: UploadFile = File(...)):
    """
    Accepts a PDF file, processes it to extract structured content and metadata,
    saves the result to a JSON file, and returns the path to the output file.
    """
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No file provided.")
    
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid file type. Only PDF files are accepted.")

    # Generate a unique filename for the uploaded PDF to avoid conflicts
    unique_id = uuid.uuid4().hex
    pdf_filename = f"{unique_id}_{file.filename}"
    uploaded_pdf_path = UPLOAD_DIR / pdf_filename

    try:
        # Save the uploaded PDF
        with open(uploaded_pdf_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Uploaded PDF saved to: {uploaded_pdf_path}")

        # --- Milestone 1: Extract Structured Content ---
        logger.info(f"Starting content extraction for {pdf_filename}...")
        # document_parser needs a working directory for its HTML assets
        # Create a unique sub-directory within TEMP_PROCESSING_DIR for this request
        request_temp_dir = TEMP_PROCESSING_DIR / unique_id
        request_temp_dir.mkdir(parents=True, exist_ok=True)

        structured_content = document_parser.extract_structured_content_from_pdf(
            pdf_filepath=uploaded_pdf_path,
            working_dir=request_temp_dir 
        )

        if structured_content is None:
            logger.error(f"Content extraction failed for {pdf_filename}.")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to extract content from PDF.")
        logger.info(f"Content extraction successful for {pdf_filename}.")

        # --- Milestone 2: Extract Legal Metadata ---
        logger.info(f"Starting metadata extraction for {pdf_filename}...")
        # Document name for metadata can be the original filename or the processed one
        doc_name_for_metadata = Path(file.filename).stem # Use original filename stem

        extracted_metadata = metadata_extractor.extract_document_metadata(
            structured_content=structured_content,
            doc_name=doc_name_for_metadata
        )
        logger.info(f"Metadata extraction successful for {pdf_filename}.")

        # --- Combine outputs and save ---
        final_output_data = {
            "source_pdf_filename": file.filename, # Original filename
            "processed_pdf_id": unique_id,      # Unique ID for this processing run
            "content": structured_content,
            "metadata": extracted_metadata      # This is already the "metadata" object from M2
        }

        output_json_filename = f"{doc_name_for_metadata}_{unique_id}_processed.json"
        output_json_path = OUTPUT_DIR / output_json_filename
        
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(final_output_data, f, indent=2)
        logger.info(f"Combined output saved to: {output_json_path}")

        return JSONResponse(
            content={
                "message": "PDF processed successfully.",
                "original_filename": file.filename,
                "output_file_path": str(output_json_path.relative_to(BASE_DIR)) # Return relative path
            },
            status_code=status.HTTP_200_OK
        )

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