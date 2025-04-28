from bs4 import BeautifulSoup
import requests
import json
import os

def generate_page_urls(base_url, total_pages):
    urls = [base_url]  # Page 1
    for i in range(2, total_pages + 1):  # Pages 2 onwards
        urls.append(f"{base_url}?page={i}")
    return urls

def get_item_links(page_url):
    response = requests.get(page_url)
    if response.status_code != 200:
        print(f"Failed to retrieve {page_url}. Status code: {response.status_code}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    links = []
    for a_tag in soup.find_all('a', href=True):
        if a_tag['href'].startswith('/items/show/'):
            full_url = "https://miamiuniversityartmuseum.omeka.net" + a_tag['href']
            links.append(full_url)
    return links

def fetch_soup(url):
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to retrieve {url}. Status code: {response.status_code}")
        return None
    return BeautifulSoup(response.text, 'html.parser')

def extract_data_from_soup(soup):
    data = {}
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

    # Extract tags
    data['Tags'] = [tag.get_text(strip=True) for tag in soup.find_all('a', {'rel': 'tag'})]

    # Extract collection link
    collection_element = soup.find('div', id='collection')
    if collection_element:
        link = collection_element.find('a')
        if link:
            data['Collection Link'] = link['href']

    return data

def append_to_json(data, output_path):
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
    else:
        existing_data = []

    existing_data.extend(data)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=4)

    print(f"Appended {len(data)} records to {output_path}")

def scrape_omeka(base_url, total_pages, output_path):
    page_urls = generate_page_urls(base_url, total_pages)

    for page_url in page_urls:
        print(f"Processing page: {page_url}")
        page_data = []
        item_links = get_item_links(page_url)
        for item_url in item_links:
            soup = fetch_soup(item_url)
            if soup:
                data = extract_data_from_soup(soup)
                page_data.append(data)

        append_to_json(page_data, output_path)

def main():
    base_url = "https://miamiuniversityartmuseum.omeka.net/items/browse"
    total_pages = 2
    output_path = "data/extracted_data.json"
    scrape_omeka(base_url, total_pages, output_path)

if __name__ == "__main__":
    main()
