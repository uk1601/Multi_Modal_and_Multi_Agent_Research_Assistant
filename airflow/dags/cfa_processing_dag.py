from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
import os
from pathlib import Path
from dotenv import load_dotenv

# Import your processing functions
from scraper import scrape_publications, close_driver
from parsing import main as parse_documents
from vectorisation import CFADocumentVectorizer
load_dotenv(override=True)


# Define default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(minutes=30),
    'start_date': days_ago(1)
}


# def scrape_task():
#     """Task to scrape documents"""
#     df = scrape_publications()
#     close_driver()

def parse_task():
    """Task to parse documents"""
    parse_documents()

def vectorize_task(**context):
    """Task to vectorize documents"""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    
    # Initialize vectorizer
    vectorizer = CFADocumentVectorizer(
        openai_api_key=openai_api_key,
        pinecone_api_key=pinecone_api_key
    )
    
    json_dir = Path("./data/parsed")
    json_files = list(json_dir.glob("*-with-images.json"))
    
    for json_file in json_files:
        vectorizer.process_document(json_file)

with DAG(
    'cfa_document_processing',
    default_args=default_args,
    description='Process CFA research documents',
    schedule_interval=timedelta(days=1),
    catchup=False,
    dagrun_timeout=timedelta(hours=2)

) as dag:

    # scrape = PythonOperator(
    #     task_id='scrape_documents',
    #     python_callable=scrape_task,
    #     provide_context=True,
    #     execution_timeout=timedelta(minutes=10),
    #     retries=2,
    #     retry_delay=timedelta(minutes=2),
    #     pool='default_pool'
    # )

    parse = PythonOperator(
        task_id='parse_documents',
        python_callable=parse_task,
        provide_context=True,
        execution_timeout=timedelta(minutes=15),
        retries=1,
        pool='default_pool'
    )

    vectorize = PythonOperator(
        task_id='vectorize_documents',
        python_callable=vectorize_task,
        provide_context=True,
        execution_timeout=timedelta(minutes=30),
        retries=1,
        pool='default_pool'
    )

    # Set task dependencies
    # scrape >> parse >> vectorize
    parse >> vectorize