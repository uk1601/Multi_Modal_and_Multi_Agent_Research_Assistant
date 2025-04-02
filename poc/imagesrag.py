
import os
import boto3
import random
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
import logging
from typing import Dict, List, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ResearchQueryHandler:
    def __init__(self):
        load_dotenv()
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.s3_client = boto3.client('s3')
        self.index = self.pc.Index("researchagent")
        self.namespace = "investment_research"
        self.bucket_name = "researchagent-bigdata"
        self.pdfs_prefix = "raw/pdfs/"

    def list_index_stats(self):
        try:
            stats = self.index.describe_index_stats()
            logger.info(f"Index stats: {stats}")
            return stats
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            raise

    def get_random_pdf(self) -> str:
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=self.pdfs_prefix
            )
            
            pdf_files = [obj['Key'].split('/')[-1] for obj in response['Contents'] 
                        if obj['Key'].endswith('.pdf')]
            
            if not pdf_files:
                raise ValueError("No PDF files found in S3 bucket")
            
            doc_name = random.choice(pdf_files)
            logger.info(f"Selected document: {doc_name}")
            return doc_name
        except Exception as e:
            logger.error(f"Error getting random PDF: {e}")
            raise

    def get_document_vectors(self, doc_name: str, query: str, filter_visuals: bool = True) -> List[Dict]:
        try:
            doc_name = doc_name.replace('.pdf', '')
            
            # Generate query embedding
            query_embedding = self.openai_client.embeddings.create(
                input=query,
                model="text-embedding-3-small"
            ).data[0].embedding

            # Build filter
            if filter_visuals:
                filter_expression = {
                    "$and": [
                        {"doc_name": {"$eq": doc_name}},
                        {"content_type": {"$in": [ "picture"]}}
                    ]
                }
            else:
                filter_expression = {"doc_name": {"$eq": doc_name}}

            # Execute query
            logger.info(f"Using filter: {filter_expression}")
            result = self.index.query(
                vector=query_embedding,
                filter=filter_expression,
                top_k=10,
                include_metadata=True,
                namespace=self.namespace
            )
            
            logger.info(f"Retrieved {len(result.matches)} vectors")
            if result.matches:
                for idx, match in enumerate(result.matches[:2]):
                    logger.info(f"Match {idx} - ID: {match.id}")
                    logger.info(f"Match {idx} - Metadata: {match.metadata}")
            
            return result.matches
            
        except Exception as e:
            logger.error(f"Error getting document vectors: {e}")
            raise

    def _prepare_context(self, vectors: List[Dict]) -> str:
        try:
            context_parts = []
            for vector in vectors:
                if not hasattr(vector, 'metadata') or not vector.metadata:
                    continue
                    
                metadata = vector.metadata
                if 'content_type' not in metadata:
                    continue
                
                if metadata['content_type'] == 'text' and 'text' in metadata:
                    context_parts.append(metadata['text'])
                elif metadata['content_type'] == 'table_text' and 'text' in metadata:
                    context_parts.append(f"Table content: {metadata['text']}")
                elif metadata['content_type'] in ['table_image', 'picture'] and 'image_path' in metadata:
                    desc = metadata.get('description', 'Visual content')
                    context_parts.append(f"Visual element ({metadata['content_type']}): {desc}")
                    context_parts.append(f"![{desc}]({metadata['image_path']})")
            
            return "\n\n".join(context_parts)
        except Exception as e:
            logger.error(f"Error preparing context: {e}")
            raise

    def _prepare_visual_context(self, vectors: List[Dict]) -> str:
        try:
            visual_parts = []
            for vector in vectors:
                if not hasattr(vector, 'metadata') or not vector.metadata:
                    continue
                    
                metadata = vector.metadata
                if metadata['content_type'] in ['table_image', 'picture'] and 'image_path' in metadata:
                    desc = metadata.get('description', 'Visual content')
                    visual_parts.append(f"![{desc}]({metadata['image_path']})")
            
            return "\n\n".join(visual_parts)
        except Exception as e:
            logger.error(f"Error preparing visual context: {e}")
            raise

    def generate_response(self, query: str, doc_name: str) -> str:
        try:
            regular_vectors = self.get_document_vectors(doc_name, query, filter_visuals=False)
            visual_vectors = self.get_document_vectors(doc_name, query, filter_visuals=True)
            
            if not regular_vectors and not visual_vectors:
                raise ValueError(f"No vectors found for document: {doc_name}")
            
            context = self._prepare_context(regular_vectors)
            visual_context = self._prepare_visual_context(visual_vectors)
            
            system_prompt = """You are a research analyst. Create a response in markdown format. 
            When referencing any tables or figures, you MUST include them using markdown image syntax: 
            ![description](url). Every visual element you mention must be included with its URL."""
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Query: {query}\n\nDocument Content: {context}\n\n" + 
                     f"Available Visual Elements:\n{visual_context}"}
                ]
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    def save_response_to_markdown(self, response: str, query: str, doc_name: str) -> str:
        try:
            output_dir = "research_outputs"
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{output_dir}/research_response_{timestamp}.md"
            
            markdown_content = f"""
ID: {doc_name}
# Research Query Response
## Document: {doc_name}
## Query: {query}
## Research Response:

{response}"""
            
            with open(filename, 'w') as f:
                f.write(markdown_content)
            logger.info(f"Saved response to: {filename}")
            return filename
        except Exception as e:
            logger.error(f"Error saving markdown file: {e}")
            raise

def main():
    try:
        handler = ResearchQueryHandler()
        doc_name = handler.get_random_pdf()
        query = "What are the key findings and insights from this document? Include and reference any relevant tables or figures in your response."
        
        response = handler.generate_response(query, doc_name)
        output_file = handler.save_response_to_markdown(response, query, doc_name)
        print(f"Generated response for {doc_name} and saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Error in main process: {e}")
        raise

if __name__ == "__main__":
    main()
