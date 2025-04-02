import json
import logging
import time
from pathlib import Path
from typing import Iterable


from docling.datamodel.base_models import ConversionStatus,InputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.settings import settings
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem
#from scraper import scrape_publications,close_driver
# Set up logging to a file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("./logs/parsing.log"),  # Log to a file
        logging.StreamHandler()  # Optional: log to console as well
    ]
)
_log = logging.getLogger(__name__)

IMAGE_RESOLUTION_SCALE = 2.0
USE_V2 = True

def export_documents(
    conv_results: Iterable[ConversionResult],
    output_dir: Path,
):
    # Define paths for markdown and images
    markdown_output_dir = output_dir
    images_output_dir = output_dir / "images"

    # Ensure directories exist
    markdown_output_dir.mkdir(parents=True, exist_ok=True)
    images_output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    failure_count = 0
    partial_success_count = 0

    for conv_res in conv_results:
        if conv_res.status == ConversionStatus.SUCCESS:
            success_count += 1
            doc_filename = conv_res.input.file.stem

            # Export JSON, YAML, doctags, markdown, and text
            if USE_V2:
            #     with (output_dir / f"{doc_filename}.json").open("w") as fp:
            #         fp.write(json.dumps(conv_res.document.export_to_dict()))

            #     # with (output_dir / f"{doc_filename}.yaml").open("w") as fp:
            #     #     fp.write(yaml.safe_dump(conv_res.document.export_to_dict()))

            #     # with (output_dir / f"{doc_filename}.doctags.txt").open("w") as fp:
            #     #     fp.write(conv_res.document.export_to_document_tokens())

                with (output_dir / f"{doc_filename}.md").open("w") as fp:
                    fp.write(conv_res.document.export_to_markdown())

            #     # with (output_dir / f"{doc_filename}.txt").open("w") as fp:
            #     #     fp.write(conv_res.document.export_to_markdown(strict_text=True))

            # Export images for each page, table, and figure
            # for page_no, page in conv_res.document.pages.items():
            #     page_image_filename = output_dir / f"{doc_filename}-page-{page_no}.png"
            #     with page_image_filename.open("wb") as fp:
            #         page.image.pil_image.save(fp, format="PNG")

            table_counter = 0
            picture_counter = 0
            for element, _ in conv_res.document.iterate_items():
                if isinstance(element, TableItem):
                    table_counter += 1
                    element_image_filename = (
                        images_output_dir / f"{doc_filename}-table-{table_counter}.png"
                    )
                    with element_image_filename.open("wb") as fp:
                        element.image.pil_image.save(fp, "PNG")

                if isinstance(element, PictureItem):
                    picture_counter += 1
                    element_image_filename = (
                        images_output_dir / f"{doc_filename}-picture-{picture_counter}.png"
                    )
                    with element_image_filename.open("wb") as fp:
                        element.image.pil_image.save(fp, "PNG")

            # Export markdown with embedded images
            content_md = conv_res.document.export_to_markdown(image_mode=ImageRefMode.EMBEDDED)
            #md_filename = markdown_output_dir / f"{doc_filename}-with-images.md"
            #json_filename=markdown_output_dir / f"{doc_filename}-with-images.json"
            # with md_filename.open("w") as fp:
            #     fp.write(content_md)
            # with json_filename.open("w") as fp:
            #     fp.write(json.dumps(conv_res.document.export_to_dict()))
                

        elif conv_res.status == ConversionStatus.PARTIAL_SUCCESS:
            _log.info(f"Document {conv_res.input.file} was partially converted with errors:")
            for item in conv_res.errors:
                _log.info(f"\t{item.error_message}")
            partial_success_count += 1
        else:
            _log.info(f"Document {conv_res.input.file} failed to convert.")
            failure_count += 1

    _log.info(
        f"Processed {success_count + partial_success_count + failure_count} docs, "
        f"of which {failure_count} failed and {partial_success_count} were partially converted."
    )
    return success_count, partial_success_count, failure_count

def main():
    logging.basicConfig(level=logging.INFO)

    #df=scrape_publications()
    #close_driver()

    input_doc_paths = list(Path("./data/pdfs").glob("*.pdf"))

    # Setup PDF pipeline options for image scaling and generation
    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
    pipeline_options.generate_page_images = False
    pipeline_options.generate_table_images = True
    pipeline_options.generate_picture_images = True

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    start_time = time.time()

    # Convert all documents
    conv_results = doc_converter.convert_all(input_doc_paths, raises_on_error=False)
    success_count, partial_success_count, failure_count = export_documents(
        conv_results, output_dir=Path("./data/parsed")
    )

    end_time = time.time() - start_time
    _log.info(f"Document conversion complete in {end_time:.2f} seconds.")

    if failure_count > 0:
        raise RuntimeError(
            f"The conversion failed for {failure_count} out of {len(input_doc_paths)} documents."
        )
    
if __name__ == "__main__":
    main()