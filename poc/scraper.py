import requests
from bs4 import BeautifulSoup
import time
import pandas as pd
#from aws_s3 import save_image, download_pdf
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from dotenv import load_dotenv
import os

load_dotenv(override=True)
pages=int(os.getenv("PAGES"))
number_of_pub=int(os.getenv("PUBLICATIONS"))

# Base URL for the first page
base_url = "https://rpc.cfainstitute.org/en/research-foundation/publications"

# Base domain to construct full URLs for PDFs and images
base_domain = "https://rpc.cfainstitute.org"

# Default alternative image URL
alternative_image_url = "https://media.istockphoto.com/id/1352945762/vector/no-image-available-like-missing-picture.jpg?s=612x612&w=0&k=20&c=4X-znbt02a8EIdxwDFaxfmKvUhTnLvLMv1i1f3bToog="

# Initialize the Selenium WebDriver with remote URL
chrome_options = Options()
chrome_options.add_argument("--headless")
# chrome_options.add_argument("--no-sandbox")
# chrome_options.add_argument("--disable-dev-shm-usage")

# Connect to the remote Selenium server
driver = webdriver.Chrome(
    options=chrome_options
)

# Connect to the remote Selenium server (selenium-chrome service)
# driver = webdriver.Remote(
#     command_executor="http://selenium-chrome:4444/wd/hub",
#     options=chrome_options
# )

# Define a global ID counter outside the functions
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
        if title_tag:
            title = title_tag.text.strip()
            publication_link = title_tag['href']
        else:
            print(f"No title available for publication: {len(publications_data)+1}")
            continue

        # Extract the summary text
        summary_tag = publication.find('div', class_='result-body')
        summary = summary_tag.text.strip() if summary_tag else "No summary available"

        # Extract the image URL
        result_link_div = publication.find('div', class_='result-link')
        if result_link_div:
            image_tag = result_link_div.find('img', class_='coveo-result-image')
            image_url = base_domain + image_tag['src'] if image_tag else alternative_image_url
            image_path = save_image(title, image_url)
        else:
            image_url = alternative_image_url
            image_path = save_image(title, image_url)

        # Visit the publication page to extract the PDF link
        driver.get(publication_link)
        time.sleep(3)
        publication_soup = BeautifulSoup(driver.page_source, 'html.parser')
        pdf_link_tag = publication_soup.find('a', href=lambda href: href and href.endswith('.pdf'))
        pdf_url = base_domain + pdf_link_tag['href'] if pdf_link_tag else "No PDF found"
        pdf_path = download_pdf(title, pdf_url)

        # Append data to publications_data
        publications_data.append({
            "ID": global_id_counter,
            "Title": title,
            "Summary": summary,
            "Image Path": image_path,
            "PDF Path": pdf_path
        })

        # Output the extracted information
        print(f"ID:{global_id_counter}")
        print(f"Title: {title}")
        print(f"Summary: {summary}")
        print(f"Image Path: {image_path}")
        print(f"Publication Link: {publication_link}")
        print(f"PDF Path: {pdf_path}")
        print("-" * 100)
        global_id_counter += 1  # Increment the global ID counter
    # Convert the publications_data list to a DataFrame
    publications_df = pd.DataFrame(publications_data, columns=["ID", "Title", "Summary", "Image Path", "PDF Path"])
    return publications_df



import os
import requests

def download_pdf(title, pdf_url):
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()

        if response.content:  # Ensure response has content
            # Define local path for saving the PDF
            local_path = f"./data/pdfs/{sanitize_filename(title)}.pdf"
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Write content to a local PDF file
            with open(local_path, "wb") as pdf_file:
                pdf_file.write(response.content)
                
            print(f"PDF saved locally as {local_path}")
            return local_path
        else:
            print(f"No content in PDF for {title}.")
            return ""
    except Exception as e:
        print(f"Failed to download PDF for {title}. Error: {e}")
        return ""

def save_image(title, image_url):
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        
        if response.content:  # Ensure response has content
            # Define local path for saving the image
            local_path = f"./data/images/{sanitize_filename(title)}.jpg"
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Write content to a local image file
            with open(local_path, "wb") as image_file:
                image_file.write(response.content)
                
            print(f"Image saved locally as {local_path}")
            return local_path
        else:
            print(f"No content in image for {title}.")
            return ""
    except Exception as e:
        print(f"Failed to save image for {title}. Error: {e}")
        return ""

def sanitize_filename(name):
    return "".join([c if c.isalnum() or c in " ._-()" else "_" for c in name])

def scrape_publications():
    total_pages = pages  # Adjust the number of pages
    all_publications_df = pd.DataFrame(columns=["ID", "Title", "Summary", "Image Path", "PDF Path"])
    for page_number in range(1, total_pages + 1):
        page_url = f"{base_url}#first={(page_number - 1) * 10}"
        print(f"\n{'-'*100}\nScraping page {page_number}: {page_url}\n{'-'*100}\n")
        # Get the DataFrame for each page and append it to the main DataFrame
        page_df = get_publications_from_page(page_url)
        all_publications_df = pd.concat([all_publications_df, page_df], ignore_index=True)

    # all_publications_df.to_csv("publications_data.csv", index=False)
    # print("Data saved to publications_data.csv")
    return all_publications_df

# Close the Selenium browser after scraping
def close_driver():
    driver.quit()
import pathlib
print(pathlib.Path.cwd())
df= scrape_publications()
print(df.head())
close_driver()
# df