import os
from pathlib import Path
import json
import numpy as np
from typing import Dict, List, Optional
import pinecone
from openai import OpenAI
from tqdm import tqdm
import base64
from PIL import Image
from io import BytesIO
import logging
from time import sleep
from dotenv import load_dotenv
import pinecone 
import openai
from pinecone import Pinecone, ServerlessSpec
load_dotenv(override=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./logs/vectorization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CFADocumentVectorizer:
    def __init__(
        self,
        openai_api_key: str,
        pinecone_api_key: str,
        index_name: str = "cfa-research1",
        namespace: str = "investment_research"
    ):
        # Initialize OpenAI client
        self.client = OpenAI(api_key=openai_api_key)
        self.embedding_model = "text-embedding-3-small"
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=pinecone_api_key)
        try:
            # Check if index exists
            index_list = self.pc.list_indexes()
            logger.info(f"Existing indexes: {index_list}")

            # Check if index exists in the list
            existing_indexes = index_list.get('indexes', [])
            index_exists = any(idx.get('name') == index_name for idx in existing_indexes)
            
            if index_exists:
                logger.info(f"Deleting existing index: {index_name}")
                self.pc.delete_index(index_name)
                # Wait a bit after deletion
                sleep(5)
                logger.info(f"Deleted index: {index_name}")

            # Create new index
            logger.info(f"Creating new index: {index_name}")
            self.pc.create_index(
                name=index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            
            # Wait for index to be ready
            retry_count = 0
            max_retries = 60  # Maximum 1 minute wait
            while retry_count < max_retries:
                try:
                    index_info = self.pc.describe_index(index_name)
                    if index_info.get('status', {}).get('ready', False):
                        break
                except Exception as e:
                    logger.warning(f"Waiting for index to be ready... {str(e)}")
                sleep(1)
                retry_count += 1
                
            if retry_count >= max_retries:
                raise TimeoutError("Index creation timed out")
                
            logger.info(f"Index {index_name} is ready")
            
            self.index = self.pc.Index(index_name)
            self.namespace = namespace
            
            # Set up paths
            self.output_dir = Path("./data/parsed")
            self.images_dir = self.output_dir / "images"
            
        except Exception as e:
            logger.error(f"Error initializing Pinecone: {str(e)}")
            raise


    def load_document(self, json_path: Path) -> Dict:
        """Load and validate JSON document"""
        try:
            with open(json_path) as f:
                doc_data = json.load(f)
            logger.info(f"Successfully loaded document: {json_path.name}")
            return doc_data
        except Exception as e:
            logger.error(f"Error loading document {json_path}: {e}")
            raise

    def create_embedding(self, text: str) -> List[float]:
        """Create embedding with rate limiting and retries"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    input=text,
                    model=self.embedding_model
                )
                return response.data[0].embedding
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to create embedding after {max_retries} attempts: {e}")
                    raise
                sleep(2 ** attempt)  # Exponential backoff
                continue

    def process_text_block(self, text_block: Dict, doc_name: str, current_section: str) -> Dict:
        """Process a single text block with section context"""
        text_content = text_block['text']
        label = text_block['label']
        prov = text_block['prov'][0]
        prov = text_block['prov'][0]  # Access first element of prov list
        page_no = prov['page_no']
        bbox = prov['bbox']
        # page_no = text_block['prov']['0']['page_no']
        # bbox = text_block['prov']['0']['bbox']
        # Convert bbox to string format for Pinecone metadata
        bbox_str = f"{bbox['l']:.2f},{bbox['t']:.2f},{bbox['r']:.2f},{bbox['b']:.2f}"
        try:
            embedding = self.create_embedding(text_content)
            
            return {
                'id': f"{doc_name}_text_{text_block['self_ref'].split('/')[-1]}",
                'values': embedding,
                'metadata': {
                    'doc_name': doc_name,
                    'content_type': 'text',
                    'label': label,
                    'page_no': page_no,
                    'text': text_content[:1000],  ##CHECK IF TO USE OR NOT # Limit metadata text length
                    'bbox': bbox_str,
                    'section': current_section
                }
            }
        except Exception as e:
            logger.error(f"Error processing text block: {e}")
            logger.error(f"Text block structure: {text_block}")
            return None
    def process_table(self, table: Dict, doc_name: str) -> List[Dict]:
        """Process table data and corresponding image"""
        vectors = []
        table_id = table['self_ref'].split('/')[-1]
        prov = table['prov'][0]  # Access first element of prov list
        page_no = prov['page_no']
        bbox = prov['bbox']  # Get bbox from prov
        bbox_str = f"{bbox['l']:.2f},{bbox['t']:.2f},{bbox['r']:.2f},{bbox['b']:.2f}"
        
        # Get table image path
        table_image_path = self.images_dir / f"{doc_name}-table-{int(table_id)+1}.png"
        
        # Process structured table data
        if 'data' in table and 'table_cells' in table['data']:
            table_text = []
            for cell in table['data']['grid']:
                if 'text' in cell:
                    table_text.append(cell['text'])
            
            if table_text:
                try:
                    text_embedding = self.create_embedding(" ".join(table_text))
                    
                    vectors.append({
                        'id': f"{doc_name}_table_{table_id}_text",
                        'values': text_embedding,
                        'metadata': {
                            'doc_name': doc_name,
                            'content_type': 'table_text',
                            'page_no': page_no,
                            'text': " ".join(table_text)[:1000],
                            'bbox': bbox_str,  # Add bbox to metadata
                            'table_structure': {
                                'num_rows': table['data']['table_cells']['num_rows'],
                                'num_cols': table['data']['table_cells']['num_cols']
                            },
                            'image_path': str(table_image_path)
                        }
                    })
                except Exception as e:
                    logger.error(f"Error processing table text: {e}")

        # Process table image if it exists
        if table_image_path.exists():
            try:
                with open(table_image_path, 'rb') as img_file:
                    image_data = img_file.read()
                    
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Describe this table's content and structure in detail."},
                                {"type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{base64.b64encode(image_data).decode()}"}}
                            ]
                        }
                    ],
                    max_tokens=300
                )
                
                image_description = response.choices[0].message.content
                image_embedding = self.create_embedding(image_description)
                
                vectors.append({
                    'id': f"{doc_name}_table_{table_id}_image",
                    'values': image_embedding,
                    'metadata': {
                        'doc_name': doc_name,
                        'content_type': 'table_image',
                        'page_no': page_no,
                        'bbox': bbox_str,  # Add bbox to metadata
                        'description': image_description,
                        'image_path': str(table_image_path)
                    }
                })
            except Exception as e:
                logger.error(f"Error processing table image {table_image_path}: {e}")

        return vectors

    def process_picture(self, picture: Dict, doc_name: str) -> Optional[Dict]:
        """Process picture using the extracted image file"""
        try:
            picture_id = picture['self_ref'].split('/')[-1]
            prov = picture['prov'][0]  # Access first element of prov list
            page_no = prov['page_no']
            bbox = prov['bbox']  # Get bbox from prov
            bbox_str = f"{bbox['l']:.2f},{bbox['t']:.2f},{bbox['r']:.2f},{bbox['b']:.2f}"
            
            picture_path = self.images_dir / f"{doc_name}-picture-{int(picture_id)+1}.png"
            
            if not picture_path.exists():
                logger.warning(f"Picture file not found: {picture_path}")
                return None
                
            with open(picture_path, 'rb') as img_file:
                image_data = img_file.read()
                
            # Get image description using GPT-4 Vision
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this image in detail, including any text or diagrams visible."},
                            {"type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64.b64encode(image_data).decode()}"}}
                        ]
                    }
                ],
                max_tokens=300
            )
            
            image_description = response.choices[0].message.content
            
            # Create embedding for the description
            embedding = self.create_embedding(image_description)
            
            return {
                'id': f"{doc_name}_picture_{picture_id}",
                'values': embedding,
                'metadata': {
                    'doc_name': doc_name,
                    'content_type': 'picture',
                    'page_no': page_no,
                    'bbox': bbox_str,  # Add bbox to metadata
                    'description': image_description,
                    'image_path': str(picture_path)
                }
            }
        except Exception as e:
            logger.error(f"Error processing picture {picture_id}: {str(e)}")
            logger.error(f"Picture structure: {picture}")
            return None

    def process_document(self, json_path: Path) -> None:
        """Process complete document including all its components"""
        doc_data = self.load_document(json_path)
        doc_name = doc_data['name']
        current_section = "introduction"
        vectors_to_upsert = []
        
        logger.info(f"Processing document: {doc_name}")
        
        # Process text blocks
        logger.info("Processing text blocks...")
        for text_block in tqdm(doc_data['texts'], desc="Text blocks"):
            # Update section if section header
            if text_block['label'] == 'section_header':
                current_section = text_block['text']
                
            vector = self.process_text_block(text_block, doc_name, current_section)
            if vector:
                vectors_to_upsert.append(vector)
                
                if len(vectors_to_upsert) >= 100:
                    self.index.upsert(vectors=vectors_to_upsert, namespace=self.namespace)
                    vectors_to_upsert = []
                    sleep(1)  # Rate limiting
        
        # Process tables
        logger.info("Processing tables...")
        for table in tqdm(doc_data['tables'], desc="Tables"):
            table_vectors = self.process_table(table, doc_name)
            vectors_to_upsert.extend(table_vectors)
            
            if len(vectors_to_upsert) >= 100:
                self.index.upsert(vectors=vectors_to_upsert, namespace=self.namespace)
                vectors_to_upsert = []
                sleep(1)  # Rate limiting
        
        # Process pictures
        logger.info("Processing pictures...")
        for picture in tqdm(doc_data['pictures'], desc="Pictures"):
            vector = self.process_picture(picture, doc_name)
            if vector:
                vectors_to_upsert.append(vector)
                
                if len(vectors_to_upsert) >= 100:
                    self.index.upsert(vectors=vectors_to_upsert, namespace=self.namespace)
                    vectors_to_upsert = []
                    sleep(1)  # Rate limiting
        
        # Upsert any remaining vectors
        if vectors_to_upsert:
            self.index.upsert(vectors=vectors_to_upsert, namespace=self.namespace)
        
        logger.info(f"Completed processing document: {doc_name}")

def main():
    """Process all documents in the output directory"""
    try:
        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv()
        
        openai_api_key = os.getenv("OPENAI_API_KEY")
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        #pinecone_env = os.getenv("PINECONE_ENV")
        
        if not all([openai_api_key, pinecone_api_key]):
            raise ValueError("Missing required environment variables")
        
        # Initialize vectorizer
        vectorizer = CFADocumentVectorizer(
            openai_api_key=openai_api_key,
            pinecone_api_key=pinecone_api_key
        )
        
        # Process all JSON files in parsed directory
        json_dir = Path("./data/parsed")
        json_files = list(json_dir.glob("*-with-images.json"))
        
        logger.info(f"Found {len(json_files)} documents to process")
        
        for json_file in json_files:
            try:
                logger.info(f"\nProcessing {json_file.name}")
                vectorizer.process_document(json_file)
            except Exception as e:
                logger.error(f"Error processing document {json_file.name}: {e}")
                continue
        
        logger.info("Completed processing all documents")
        
    except Exception as e:
        logger.error(f"Error in main process: {e}")
        raise

if __name__ == "__main__":
    main()