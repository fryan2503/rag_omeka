from bs4 import BeautifulSoup
import requests
import json
import os

def generate_page_urls(base_url, total_pages):
    """
    Generate a list of URLs for paginated Omeka browse pages.

    Args:
        base_url (str): The base URL for the first page (no `?page=1`).
        total_pages (int): The total number of pages to scrape.

    Returns:
        list[str]: A list of all paginated URLs.
    """
    urls = [base_url]  # Page 1 URL
    for i in range(2, total_pages + 1):
        urls.append(f"{base_url}?page={i}")
    return urls

def get_item_links(page_url):
    """
    Extract item detail page links from a browse page.

    Args:
        page_url (str): The URL of the current browse page.

    Returns:
        list[str]: List of full item detail page URLs.
    """
    response = requests.get(page_url)
    if response.status_code != 200:
        print(f"Failed to retrieve {page_url}. Status code: {response.status_code}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    links = []
    for a_tag in soup.find_all('a', href=True):
        if a_tag['href'].startswith('/items/show/'):
            # Construct the absolute URL
            full_url = "https://miamiuniversityartmuseum.omeka.net" + a_tag['href']
            links.append(full_url)
    print(links)
    return links
    
    


def fetch_soup(url):
    """
    Fetches a webpage and returns its parsed BeautifulSoup object.

    Args:
        url (str): The webpage URL.

    Returns:
        BeautifulSoup | None: Parsed HTML or None if failed.
    """
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to retrieve {url}. Status code: {response.status_code}")
        return None
    return BeautifulSoup(response.text, 'html.parser')

def extract_data_from_soup(soup, item_url):
    """
    Extracts metadata fields and tags from an Omeka item page soup.

    Args:
        soup (BeautifulSoup): Parsed HTML of an Omeka item detail page.
        item_url (str): The URL of the item page.

    Returns:
        dict: Extracted data dictionary.
    """
    data = {}
    # Map Omeka div IDs to output field names
    fields = {
        'dublin-core-title': 'Title',
        'dublin-core-identifier': 'Identifier',
        'dublin-core-subject': 'Subject',
        'dublin-core-description': 'Description',
        'dublin-core-creator': 'Creator',
        'dublin-core-format': 'Format',
        'dublin-core-date': 'Date',
        'dublin-core-medium': 'Medium',
        'physical-object-item-type-metadata-donor': 'Donor',
        'item-citation': 'Citation'
    }

    for field_id, field_name in fields.items():
        element = soup.find('div', {'id': field_id})
        if element:
            text_element = element.find('div', class_='element-text')
            if text_element:
                data[field_name] = text_element.get_text(strip=True)

    # Extract tags from the item
    data['Tags'] = [tag.get_text(strip=True) for tag in soup.find_all('a', {'rel': 'tag'})]
    # Add the item URL
    data['Collection Link'] = item_url
    return data

def append_to_json(data, output_path):
    """
    Appends extracted data to a JSON file. Ensures the output file is a list.

    Args:
        data (list[dict] or dict): New data to append.
        output_path (str): Path to the JSON output file.

    Side effects:
        Writes/updates the output JSON file with new data.
    """
    if not isinstance(data, list):
        data = [data]

    # Read existing data if the file exists
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            try:
                existing_data = json.load(f)
                if not isinstance(existing_data, list):
                    existing_data = [existing_data]
            except json.JSONDecodeError:
                existing_data = []
    else:
        existing_data = []

    existing_data.extend(data)

    # Write all data back to the file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=4)

    print(f"Appended {len(data)} records to {output_path}")

def scrape_omeka(base_url, total_pages, output_path):
    """
    Main scraping routine: iterates over paginated browse pages,
    extracts item links, fetches details, and saves the results.

    Args:
        base_url (str): The main browse URL.
        total_pages (int): Total number of paginated browse pages.
        output_path (str): Output file path for JSON data.
    """
    page_urls = generate_page_urls(base_url, total_pages)

    for page_url in page_urls:
        print(f"Processing page: {page_url}")
        page_data = []
        item_links = get_item_links(page_url) # page_url is the url of the individual artwork
        for item_url in item_links:
            soup = fetch_soup(item_url)
            if soup:
                data = extract_data_from_soup(soup,item_url)
                page_data.append(data)

        append_to_json(page_data, output_path)

def main():
    base_url = "https://miamiuniversityartmuseum.omeka.net/items/browse"
    total_pages = 1
    output_path = "data/extracted_data.json"
    scrape_omeka(base_url, total_pages, output_path)

if __name__ == "__main__":
    main()
