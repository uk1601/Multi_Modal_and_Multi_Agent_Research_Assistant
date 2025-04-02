import json
import logging
import time
from pathlib import Path
import tempfile
from io import BytesIO
import os
from dotenv import load_dotenv
from typing import List, Dict, Tuple

from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.settings import settings
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem
from AWS_utils import S3Handler
from scraper import scrape_publications,close_driver

# Load environment variables
load_dotenv(override=True)
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("./logs/parsing.log"),
        logging.StreamHandler()
    ]
)
_log = logging.getLogger(__name__)

class S3DocumentParser:
    def __init__(self, bucket_name: str):
        """Initialize the S3DocumentParser with bucket configuration"""
        self.s3_handler = S3Handler(bucket_name)
        self.temp_dir = Path(tempfile.mkdtemp())
        self.setup_document_converter()
        _log.info(f"Initialized S3DocumentParser with bucket: {bucket_name}")
        _log.info(f"Using temporary directory: {self.temp_dir}")

    def setup_document_converter(self):
        """Configure the document converter with appropriate settings"""
        pipeline_options = PdfPipelineOptions()
        pipeline_options.images_scale = 2.0
        pipeline_options.generate_page_images = False
        pipeline_options.generate_table_images = True
        pipeline_options.generate_picture_images = True

        self.doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        _log.info("Document converter setup completed")

    def download_pdfs_from_s3(self) -> List[Path]:
        """Download PDFs from S3 to temporary directory"""
        local_paths = []
        pdf_files = self.s3_handler.list_files(self.s3_handler.pdf_prefix)
        _log.info(f"Found {len(pdf_files)} PDF files in S3")

        for pdf_key in pdf_files:
            try:
                local_pdf_path = self.temp_dir / Path(pdf_key).name
                if self.s3_handler.download_file(pdf_key, str(local_pdf_path)):
                    local_paths.append(local_pdf_path)
                    _log.info(f"Successfully downloaded: {pdf_key}")
            except Exception as e:
                _log.error(f"Error downloading {pdf_key}: {str(e)}")

        return local_paths

    def process_document_images(self, conv_res: ConversionResult, doc_filename: str) -> Dict[str, List[str]]:
        """Process and upload images from the document"""
        image_paths = {"tables": [], "pictures": []}
        table_counter = picture_counter = 0

        for element, _ in conv_res.document.iterate_items():
            try:
                if isinstance(element, (TableItem, PictureItem)):
                    is_table = isinstance(element, TableItem)
                    counter = table_counter if is_table else picture_counter
                    element_type = "table" if is_table else "picture"
                    
                    # Create image filename
                    img_filename = f"{doc_filename}-{element_type}-{counter + 1}.png"
                    
                    # Save image to BytesIO
                    img_buffer = BytesIO()
                    element.image.pil_image.save(img_buffer, format="PNG")
                    img_buffer.seek(0)
                    
                    # Upload to S3
                    s3_path = self.s3_handler.save_parsed_image(img_filename, img_buffer.getvalue())
                    
                    if is_table:
                        image_paths["tables"].append(s3_path)
                        table_counter += 1
                    else:
                        image_paths["pictures"].append(s3_path)
                        picture_counter += 1

                    _log.info(f"Processed and uploaded {element_type} image: {img_filename}")
                    
            except Exception as e:
                _log.error(f"Error processing image in document {doc_filename}: {str(e)}")

        return image_paths

    def export_document_content(self, conv_res: ConversionResult, doc_filename: str) -> Tuple[str, str]:
        """Export document content to markdown and JSON"""
        try:
            # Generate markdown content
            content_md = conv_res.document.export_to_markdown(image_mode=ImageRefMode.EMBEDDED)
            md_filename = f"{doc_filename}-with-images.md"
            md_path = self.s3_handler.save_parsed_json(md_filename, content_md)
            
            # Generate JSON content
            json_content = json.dumps(conv_res.document.export_to_dict())
            json_filename = f"{doc_filename}-with-images.json"
            json_path = self.s3_handler.save_parsed_json(json_filename, json_content)
            
            _log.info(f"Exported document content for {doc_filename}")
            return md_path, json_path
            
        except Exception as e:
            _log.error(f"Error exporting document content for {doc_filename}: {str(e)}")
            return "", ""

    def process_s3_pdfs(self) -> Tuple[int, int, int]:
        """Main method to process all PDFs from S3"""
        success_count = partial_success_count = failure_count = 0
        
        try:
            # Download PDFs from S3
            local_pdf_paths = self.download_pdfs_from_s3()
            
            if not local_pdf_paths:
                _log.error("No PDFs found in S3 bucket")
                return 0, 0, 0

            # Convert PDFs
            conv_results = self.doc_converter.convert_all(local_pdf_paths, raises_on_error=False)
            
            for conv_res in conv_results:
                try:
                    if conv_res.status == ConversionStatus.SUCCESS:
                        doc_filename = conv_res.input.file.stem
                        _log.info(f"Processing document: {doc_filename}")

                        # Process images
                        image_paths = self.process_document_images(conv_res, doc_filename)
                        
                        # Export content
                        md_path, json_path = self.export_document_content(conv_res, doc_filename)
                        
                        if md_path and json_path:
                            success_count += 1
                            _log.info(f"Successfully processed document: {doc_filename}")
                        else:
                            partial_success_count += 1
                            _log.warning(f"Partially processed document: {doc_filename}")

                    elif conv_res.status == ConversionStatus.PARTIAL_SUCCESS:
                        _log.warning(f"Document {conv_res.input.file} was partially converted with errors:")
                        for item in conv_res.errors:
                            _log.warning(f"\t{item.error_message}")
                        partial_success_count += 1
                        
                    else:
                        _log.error(f"Document {conv_res.input.file} failed to convert")
                        failure_count += 1

                except Exception as e:
                    _log.error(f"Error processing conversion result: {str(e)}")
                    failure_count += 1

        except Exception as e:
            _log.error(f"Error in main processing loop: {str(e)}")
        
        finally:
            # Cleanup temporary directory
            try:
                for file in self.temp_dir.glob("*"):
                    file.unlink()
                self.temp_dir.rmdir()
                _log.info("Cleaned up temporary directory")
            except Exception as e:
                _log.error(f"Error cleaning up temporary directory: {str(e)}")

        _log.info(
            f"Processing complete. Success: {success_count}, "
            f"Partial: {partial_success_count}, Failed: {failure_count}"
        )
        
        return success_count, partial_success_count, failure_count

def main():
    """Main execution function"""
    start_time = time.time()
    
    try:
        if not AWS_BUCKET_NAME:
            raise ValueError("AWS_BUCKET_NAME environment variable not set")
        
        parser = S3DocumentParser(AWS_BUCKET_NAME)
        success, partial, failed = parser.process_s3_pdfs()
        
        end_time = time.time() - start_time
        _log.info(f"Document conversion completed for {success} documents in {end_time:.2f} seconds")
        
        if failed > 0:
            _log.error(f"The conversion failed for {failed} documents")
            return 1
            
        return 0
        
    except Exception as e:
        _log.error(f"Critical error in main function: {str(e)}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)