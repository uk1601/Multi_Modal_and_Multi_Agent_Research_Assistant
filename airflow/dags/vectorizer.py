import os
import sys
from pathlib import Path
import json
import numpy as np
from typing import Dict, List, Optional, Tuple
import pinecone
from openai import OpenAI
from tqdm import tqdm
import base64
from PIL import Image
from io import BytesIO
import logging
from time import sleep
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from AWS_utils import S3Handler

# Load environment variables
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
        aws_bucket_name: str,
        aws_region: str = "us-east-1",
        index_name: str = "researchagent",
        namespace: str = "investment_research"
    ):
        """Initialize the vectorizer with necessary credentials and configurations"""
        # Initialize OpenAI client
        self.client = OpenAI(api_key=openai_api_key)
        self.embedding_model = "text-embedding-3-small"
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.setup_pinecone_index(index_name)
        self.index = self.pc.Index(index_name)
        self.namespace = namespace
        
        # Initialize S3 handler with region
        self.s3_handler = S3Handler(aws_bucket_name, aws_region)
        
        logger.info("Vectorizer initialized successfully")
    
    def setup_pinecone_index(self, index_name: str):
        """Set up Pinecone index with proper configuration"""
        try:
            # Check if index exists
            index_list = self.pc.list_indexes()
            logger.info(f"Existing indexes: {index_list}")

            existing_indexes = index_list.get('indexes', [])
            index_exists = any(idx.get('name') == index_name for idx in existing_indexes)
            
            if index_exists:
                logger.info(f"Deleting existing index: {index_name}")
                self.pc.delete_index(index_name)
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
            self._wait_for_index_ready(index_name)
            
        except Exception as e:
            logger.error(f"Error initializing Pinecone: {str(e)}")
            raise

    def _wait_for_index_ready(self, index_name: str, max_retries: int = 60):
        """Wait for Pinecone index to be ready"""
        retry_count = 0
        while retry_count < max_retries:
            try:
                index_info = self.pc.describe_index(index_name)
                if index_info.get('status', {}).get('ready', False):
                    logger.info(f"Index {index_name} is ready")
                    return True
            except Exception as e:
                logger.warning(f"Waiting for index to be ready... {str(e)}")
            sleep(1)
            retry_count += 1
        
        raise TimeoutError("Index creation timed out")

    def load_document_from_s3(self, json_key: str) -> Dict:
        """Load JSON document from S3"""
        try:
            json_obj = self.s3_handler.download_fileobj(json_key)
            if json_obj:
                doc_data = json.loads(json_obj.read().decode('utf-8'))
                logger.info(f"Successfully loaded document: {json_key}")
                return doc_data
            else:
                raise ValueError(f"Could not download {json_key} from S3")
        except Exception as e:
            logger.error(f"Error loading document {json_key}: {e}")
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
        try:
            text_content = text_block['text']
            label = text_block['label']
            prov = text_block['prov'][0]
            page_no = prov['page_no']
            bbox = prov['bbox']
            bbox_str = f"{bbox['l']:.2f},{bbox['t']:.2f},{bbox['r']:.2f},{bbox['b']:.2f}"
            
            embedding = self.create_embedding(text_content)
            
            return {
                'id': f"{doc_name}_text_{text_block['self_ref'].split('/')[-1]}",
                'values': embedding,
                'metadata': {
                    'doc_name': doc_name,
                    'content_type': 'text',
                    'label': label,
                    'page_no': page_no,
                    'text': text_content[:1000],
                    'bbox': bbox_str,
                    'section': current_section,
                    's3_source': self.s3_handler.get_s3_url(f"{self.s3_handler.parsed_prefix}{doc_name}")
                }
            }
        except Exception as e:
            logger.error(f"Error processing text block: {e}")
            logger.error(f"Text block structure: {text_block}")
            return None

    def process_table(self, table: Dict, doc_name: str) -> List[Dict]:
        """Process table data and corresponding image from S3"""
        vectors = []
        table_id = table['self_ref'].split('/')[-1]
        prov = table['prov'][0]
        page_no = prov['page_no']
        bbox = prov['bbox']
        bbox_str = f"{bbox['l']:.2f},{bbox['t']:.2f},{bbox['r']:.2f},{bbox['b']:.2f}"
        
        # Get S3 path for table image
        table_image_key = f"{self.s3_handler.parsed_images_prefix}{doc_name}-table-{int(table_id)+1}.png"
        s3_image_url = self.s3_handler.get_s3_url(table_image_key)
        
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
                            'bbox': bbox_str,
                            'table_structure': {
                                'num_rows': table['data']['table_cells']['num_rows'],
                                'num_cols': table['data']['table_cells']['num_cols']
                            },
                            'image_path': s3_image_url,
                            's3_source': self.s3_handler.get_s3_url(f"{self.s3_handler.parsed_prefix}{doc_name}")
                        }
                    })
                except Exception as e:
                    logger.error(f"Error processing table text: {e}")

        # Process table image if it exists in S3
        try:
            image_data = self.s3_handler.download_fileobj(table_image_key)
            if image_data:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Describe this table's content and structure in detail."},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{base64.b64encode(image_data.getvalue()).decode()}"
                                    }
                                }
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
                        'bbox': bbox_str,
                        'description': image_description,
                        'image_path': s3_image_url,
                        's3_source': self.s3_handler.get_s3_url(f"{self.s3_handler.parsed_prefix}{doc_name}")
                    }
                })
        except Exception as e:
            logger.error(f"Error processing table image {s3_image_url}: {e}")

        return vectors

    def process_picture(self, picture: Dict, doc_name: str) -> Optional[Dict]:
        """Process picture using the image from S3"""
        try:
            picture_id = picture['self_ref'].split('/')[-1]
            prov = picture['prov'][0]
            page_no = prov['page_no']
            bbox = prov['bbox']
            bbox_str = f"{bbox['l']:.2f},{bbox['t']:.2f},{bbox['r']:.2f},{bbox['b']:.2f}"
            
            # Get S3 path for picture
            picture_key = f"{self.s3_handler.parsed_images_prefix}{doc_name}-picture-{int(picture_id)+1}.png"
            s3_image_url = self.s3_handler.get_s3_url(picture_key)
            
            image_data = self.s3_handler.download_fileobj(picture_key)
            if not image_data:
                logger.warning(f"Picture file not found in S3: {picture_key}")
                return None
                
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this image in detail, including any text or diagrams visible."},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64.b64encode(image_data.getvalue()).decode()}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300
            )
            
            image_description = response.choices[0].message.content
            embedding = self.create_embedding(image_description)
            
            return {
                'id': f"{doc_name}_picture_{picture_id}",
                'values': embedding,
                'metadata': {
                    'doc_name': doc_name,
                    'content_type': 'picture',
                    'page_no': page_no,
                    'bbox': bbox_str,
                    'description': image_description,
                    'image_path': s3_image_url,
                    's3_source': self.s3_handler.get_s3_url(f"{self.s3_handler.parsed_prefix}{doc_name}")
                }
            }
        except Exception as e:
            logger.error(f"Error processing picture {picture_id}: {str(e)}")
            return None

    def process_document(self, json_key: str) -> None:
        """Process complete document including all its components"""
        doc_data = self.load_document_from_s3(json_key)
        doc_name = doc_data['name']
        current_section = "introduction"
        vectors_to_upsert = []
        
        logger.info(f"Processing document: {doc_name}")
        
        # Process text blocks
        logger.info("Processing text blocks...")
        for text_block in tqdm(doc_data['texts'], desc="Text blocks"):
            if text_block['label'] == 'section_header':
                current_section = text_block['text']
                
            vector = self.process_text_block(text_block, doc_name, current_section)
            if vector:
                vectors_to_upsert.append(vector)
                
                if len(vectors_to_upsert) >= 100:
                    self.index.upsert(vectors=vectors_to_upsert, namespace=self.namespace)
                    vectors_to_upsert = []
                    sleep(1)
        
        # Process tables
        logger.info("Processing tables...")
        for table in tqdm(doc_data['tables'], desc="Tables"):
            table_vectors = self.process_table(table, doc_name)
            vectors_to_upsert.extend(table_vectors)
            
            if len(vectors_to_upsert) >= 100:
                self.index.upsert(vectors=vectors_to_upsert, namespace=self.namespace)
                vectors_to_upsert = []
                sleep(1)
        
        # Process pictures
        logger.info("Processing pictures...")
        for picture in tqdm(doc_data['pictures'], desc="Pictures"):
            vector = self.process_picture(picture, doc_name)
            if vector:
                vectors_to_upsert.append(vector)
                
                if len(vectors_to_upsert) >= 100:
                    self.index.upsert(vectors=vectors_to_upsert, namespace=self.namespace)
                    vectors_to_upsert = []
                    sleep(1)
        
        # Upsert any remaining vectors
        if vectors_to_upsert:
            self.index.upsert(vectors=vectors_to_upsert, namespace=self.namespace)
        
        logger.info(f"Completed processing document: {doc_name}")

def main():
    """Main execution function"""
    try:
        # Load environment variables
        openai_api_key = os.getenv("OPENAI_API_KEY")
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        aws_bucket_name = os.getenv("AWS_BUCKET_NAME")
        aws_region = os.getenv("AWS_REGION", "us-east-1")
        
        if not all([openai_api_key, pinecone_api_key, aws_bucket_name,aws_region]):
            raise ValueError("Missing required environment variables")
        
        # Initialize vectorizer
        vectorizer = CFADocumentVectorizer(
            openai_api_key=openai_api_key,
            pinecone_api_key=pinecone_api_key,
            aws_bucket_name=aws_bucket_name,
            aws_region=aws_region
        )
        
        # Process all JSON files from S3
        json_files = vectorizer.s3_handler.list_files(vectorizer.s3_handler.parsed_prefix)
        json_files = [f for f in json_files if f.endswith('-with-images.json')]
        
        logger.info(f"Found {len(json_files)} documents to process")
        
        for json_file in json_files:
            try:
                logger.info(f"\nProcessing {json_file}")
                vectorizer.process_document(json_file)
            except Exception as e:
                logger.error(f"Error processing document {json_file}: {e}")
                continue
        
        logger.info("Completed processing all documents")
        
    except Exception as e:
        logger.error(f"Error in main process: {e}")
        raise

def create_vector_search(
    openai_api_key: str,
    pinecone_api_key: str,
    aws_bucket_name: str,
    query: str,
    namespace: str = "investment_research",
    top_k: int = 5
) -> List[Dict]:
    """
    Utility function to search vectors using a query
    Args:
        openai_api_key: OpenAI API key
        pinecone_api_key: Pinecone API key
        aws_bucket_name: AWS bucket name
        query: Search query
        namespace: Pinecone namespace
        top_k: Number of results to return
    Returns:
        List of matching results with metadata
    """
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=openai_api_key)
        
        # Create query embedding
        response = client.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        )
        query_embedding = response.data[0].embedding
        
        # Initialize Pinecone
        pc = Pinecone(api_key=pinecone_api_key)
        index = pc.Index("cfa-research1")
        
        # Search vectors
        search_response = index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=namespace,
            include_metadata=True
        )
        
        # Format results
        results = []
        for match in search_response['matches']:
            result = {
                'score': match['score'],
                'metadata': match['metadata']
            }
            results.append(result)
            
        return results
        
    except Exception as e:
        logger.error(f"Error in vector search: {e}")
        raise

def cleanup_indexes():
    """Utility function to cleanup Pinecone indexes"""
    try:
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            raise ValueError("Pinecone API key not found")
            
        pc = Pinecone(api_key=pinecone_api_key)
        indexes = pc.list_indexes()
        
        for index in indexes.get('indexes', []):
            index_name = index.get('name')
            if index_name:
                logger.info(f"Deleting index: {index_name}")
                pc.delete_index(index_name)
                sleep(1)
        
        logger.info("Successfully cleaned up all indexes")
        
    except Exception as e:
        logger.error(f"Error cleaning up indexes: {e}")
        raise

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Critical error: {e}")
        sys.exit(1)