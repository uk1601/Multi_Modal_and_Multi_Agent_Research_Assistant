import requests
from bs4 import BeautifulSoup
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from dotenv import load_dotenv
import os
from AWS_utils import S3Handler

load_dotenv(override=True)
pages = int(os.getenv("PAGES"))
number_of_pub = int(os.getenv("PUBLICATIONS"))
aws_bucket = os.getenv("AWS_BUCKET_NAME")

# Initialize S3 handler
s3_handler = S3Handler(aws_bucket)

# Base URL for the first page
base_url = "https://rpc.cfainstitute.org/en/research-foundation/publications"
# Base domain to construct full URLs for PDFs and images
base_domain = "https://rpc.cfainstitute.org"
# Default alternative image URL
alternative_image_url = "https://media.istockphoto.com/id/1352945762/vector/no-image-available-like-missing-picture.jpg?s=612x612&w=0&k=20&c=4X-znbt02a8EIdxwDFaxfmKvUhTnLvLMv1i1f3bToog="


# Initialize Selenium
chrome_options = Options()
chrome_options.add_argument("--headless")
# Connect to the remote Selenium server
# driver = webdriver.Chrome(
#     options=chrome_options
# )
driver = webdriver.Remote(
    command_executor="http://selenium-chrome:4444/wd/hub",
    options=chrome_options
)

global_id_counter = 1

def get_publications_from_page(page_url):
    global global_id_counter  # Declare as global to update across function calls

    # List to store publication data
    publications_data = []
    # Load the page using Selenium
    driver.get(page_url)
    time.sleep(5)  # Wait for the page to fully load

    # Get the page content and pass it to BeautifulSoup
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    publications = soup.find_all('div', class_='coveo-list-layout CoveoResult')
    publications=publications[:number_of_pub]

    for publication in publications:
        # Get the title and link
        title_tag = publication.find('h4', class_='coveo-title').find('a', class_='CoveoResultLink')
        if not title_tag:
            print(f"No title available for publication: {len(publications_data)+1}")
            continue

        title = title_tag.text.strip()
        publication_link = title_tag['href']
        
        # Extract the summary text
        summary_tag = publication.find('div', class_='result-body')
        summary = summary_tag.text.strip() if summary_tag else "No summary available"

        # Handle image
        result_link_div = publication.find('div', class_='result-link')
        image_url = alternative_image_url
        if result_link_div:
            image_tag = result_link_div.find('img', class_='coveo-result-image')
            if image_tag:
                image_url = base_domain + image_tag['src']
        
        # Save image to S3
        image_path = s3_handler.save_image(title, image_url)

        # Get PDF URL and save to S3
        driver.get(publication_link)
        time.sleep(3)
        publication_soup = BeautifulSoup(driver.page_source, 'html.parser')
        pdf_link_tag = publication_soup.find('a', href=lambda href: href and href.endswith('.pdf'))
        pdf_url = base_domain + pdf_link_tag['href'] if pdf_link_tag else None
        pdf_path = s3_handler.save_pdf(title, pdf_url) if pdf_url else "No PDF Found!!!"

        # Append data to publications_data
        publications_data.append({
            "ID": global_id_counter,
            "Title": title,
            "Summary": summary,
            "Image Path": image_path,
            "PDF Path": pdf_path
        })

        # Output the extracted information
        print(f"ID: {global_id_counter}")
        print(f"Title: {title}")
        print(f"Summary: {summary}")
        print(f"Image Path: {image_path}")
        print(f"PDF Path: {pdf_path}")
        print("-" * 100)
        
        global_id_counter += 1

    return pd.DataFrame(publications_data)

def scrape_publications():
    all_publications_df = pd.DataFrame(columns=["ID", "Title", "Summary", "Image Path", "PDF Path"])
    
    for page_number in range(1, pages + 1):
        page_url = f"{base_url}#first={(page_number - 1) * 10}"
        print(f"\n{'-'*100}\nScraping page {page_number}: {page_url}\n{'-'*100}\n")
        page_df = get_publications_from_page(page_url)
        all_publications_df = pd.concat([all_publications_df, page_df], ignore_index=True)

    return all_publications_df

def close_driver():
    driver.quit()
df=scrape_publications()
close_driver()
