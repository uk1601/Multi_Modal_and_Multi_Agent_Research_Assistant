import os
from dotenv import load_dotenv
from openai import OpenAI
import pinecone
from pinecone import Pinecone
import random
import boto3
import logging
from typing import Dict, List, Optional
import json
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ResearchSummaryGenerator:
    def __init__(self):
        load_dotenv()
        
        # Initialize clients
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.s3_client = boto3.client('s3')
        
        # Initialize Pinecone index
        self.index = self.pc.Index("researchagent")
        self.namespace = "investment_research"
        
        # S3 configuration
        self.bucket_name = "researchagent-bigdata"
        self.pdfs_prefix = "raw/pdfs/"

    def get_random_pdf(self) -> str:
        """Get a random PDF from S3 bucket"""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=self.pdfs_prefix
            )
            
            pdf_files = [obj['Key'] for obj in response['Contents'] 
                        if obj['Key'].endswith('.pdf')]
            
            if not pdf_files:
                raise ValueError("No PDF files found in S3 bucket")
            
            selected_pdf = random.choice(pdf_files)
            logger.info(f"Selected PDF: {selected_pdf}")
            return selected_pdf
            
        except Exception as e:
            logger.error(f"Error getting random PDF: {e}")
            raise

    def get_document_vectors(self, doc_name: str) -> List[Dict]:
        """Get all vectors for a specific document from Pinecone"""
        try:
            # Query Pinecone for all vectors related to this document
            query_response = self.index.query(
                vector=[0] * 1536,  # Dummy vector for metadata filtering
                filter={
                    "doc_name": doc_name
                },
                top_k=10000,  # Get all vectors
                include_metadata=True
            )
            
            return query_response.matches
            
        except Exception as e:
            logger.error(f"Error getting document vectors: {e}")
            raise

    def generate_document_summary(self, doc_vectors: List[Dict]) -> Dict:
        """Generate a comprehensive summary using OpenAI"""
        try:
            # Collect all text content
            text_content = []
            tables = []
            images = []
            
            for vector in doc_vectors:
                metadata = vector.metadata
                if metadata['content_type'] == 'text':
                    text_content.append(metadata['text'])
                elif metadata['content_type'] == 'table_text':
                    tables.append(metadata)
                elif metadata['content_type'] == 'picture':
                    images.append(metadata)

            # Generate main summary
            main_summary = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a research analyst creating a comprehensive document summary."},
                    {"role": "user", "content": f"Create a detailed summary of this research document. Focus on key findings, methodology, and conclusions: {' '.join(text_content[:5000])}"}
                ]
            )

            # Generate key points
            key_points = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Extract the main points and findings from this research."},
                    {"role": "user", "content": f"List the key points and findings from this document: {' '.join(text_content[:3000])}"}
                ]
            )

            return {
                "summary": main_summary.choices[0].message.content,
                "key_points": key_points.choices[0].message.content,
                "tables": tables,
                "images": images
            }
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            raise

    def format_codelab_document(self, doc_name: str, summary_data: Dict) -> str:
        """Format the summary in Google Codelabs format"""
        try:
            # Clean the document name
            clean_doc_name = doc_name.replace('_', ' ').replace('.pdf', '')
            
            codelab_content = f"""author: Research Agent
summary: Analysis of {clean_doc_name}
id: {doc_name.lower().replace('.pdf', '').replace('_', '-')}
categories: Research Analysis
environments: Web
status: Published
feedback link: 
analytics account: 

# {clean_doc_name}
<!-- ------------------------ -->
## Overview 
Duration: 5

### Document Analysis
This document provides a comprehensive analysis of the research paper "{clean_doc_name}".

### Key Findings
{summary_data['key_points']}

<!-- ------------------------ -->
## Executive Summary
Duration: 10

{summary_data['summary']}

<!-- ------------------------ -->
## Visual Elements
Duration: 5

### Tables and Figures
"""
            # Add tables
            for i, table in enumerate(summary_data['tables'], 1):
                codelab_content += f"\n#### Table {i}\n"
                codelab_content += f"Location: Page {table['page_no']}\n"
                codelab_content += f"Description: {table.get('description', 'Table content')}\n"
                codelab_content += f"[View Table]({table['image_path']})\n"

            # Add images
            for i, image in enumerate(summary_data['images'], 1):
                codelab_content += f"\n#### Figure {i}\n"
                codelab_content += f"Location: Page {image['page_no']}\n"
                codelab_content += f"Description: {image['description']}\n"
                codelab_content += f"[View Image]({image['image_path']})\n"

            return codelab_content
            
        except Exception as e:
            logger.error(f"Error formatting codelab document: {e}")
            raise

    def generate_research_codelab(self) -> str:
        """Main function to generate research codelab"""
        try:
            # Get random PDF
            pdf_key = self.get_random_pdf()
            doc_name = pdf_key.split('/')[-1]
            
            # Get document vectors
            vectors = self.get_document_vectors(doc_name)
            
            # Generate summary
            summary_data = self.generate_document_summary(vectors)
            
            # Format as codelab
            codelab_content = self.format_codelab_document(doc_name, summary_data)
            
            # Save to file
            output_dir = "abc"
            os.makedirs(output_dir, exist_ok=True)
            
            output_file = os.path.join(
                output_dir, 
                f"research_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            )
            
            with open(output_file, 'w') as f:
                f.write(codelab_content)
                
            logger.info(f"Generated codelab document: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error generating research codelab: {e}")
            raise

def main():
    try:
        generator = ResearchSummaryGenerator()
        output_file = generator.generate_research_codelab()
        print(f"Successfully generated codelab document: {output_file}")
        
    except Exception as e:
        logger.error(f"Error in main process: {e}")
        raise

if __name__ == "__main__":
    main()