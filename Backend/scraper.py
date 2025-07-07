import requests
from bs4 import BeautifulSoup
import pandas as pd
import time # To not overwhelm the server
import re
import csv
import openpyxl

print("Starting the scraping process...")

# The main page with the list of all State of the Union addresses
BASE_URL = "https://www.presidency.ucsb.edu"
SOTU_LIST_URL = BASE_URL + "/documents/presidential-documents-archive-guidebook/annual-messages-congress-the-state-the-union"

# Step 1: Get the main page content

# Identifying the user-agent
HEADERS = {
    'User-Agent': 'SOTU Scraper Bot (Educational Project) - Contact: <EMAIL>'
}


try:
    response = requests.get(SOTU_LIST_URL, headers=HEADERS)
    response.raise_for_status() # This will raise an error for bad responses (4xx or 5xx)
except requests.exceptions.RequestException as e:
    print(f"Error fetching list page: {e}")
    exit()

soup = BeautifulSoup(response.content, 'html.parser')

# Step 2: Find all the links to individual speeches
# By inspecting the website's HTML, we see speech links are in <td> tags with class 'views-field-title'
sotu_table = soup.find('table', class_='table-responsive')

if not sotu_table:
    print("Error: Could not find the main SOTU table on the page. The website structure may have changed.")
    exit()


speech_links = []
# Find all 'a' (anchor) tags within that specific table
all_links_in_table = sotu_table.find_all('a')
# Find all table rows, because each row corresponds to one speech
table_rows = soup.find_all('tr')

for row in table_rows:
    # Look for the cell containing the title and link
    title_cells = row.find_all('td')
    for title_cell in title_cells:
        # Looking for links in this cell:
        link_tag = title_cell.find('a')
        if link_tag and 'href' in link_tag.attrs:
            link = link_tag['href']
            # If link is relative, prepend BASE url.
            if link.startswith('/'):
                speech_links.append(BASE_URL + link)
            else:
                speech_links.append(link)


print(f"Found {len(speech_links)} speech links.")

# Step 3 & 4: Visit each speech link and extract the data
all_speeches_data = []
for i, url in enumerate(speech_links):
    # Give a status update
    print(f"Scraping speech {i + 1}/{len(speech_links)}: {url}")

    try:
        speech_response = requests.get(url)
        speech_response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"  -> Could not fetch {url}. Error: {e}. Skipping.")
        continue

    speech_soup = BeautifulSoup(speech_response.content, 'html.parser')

    # --- MODIFICATION START ---
    # By inspecting the HTML, all relevant content is inside <div class="col-sm-8">
    # We will search within this container to make our selectors more reliable.
    main_content = speech_soup.find('div', class_='col-sm-8')

    if not main_content:
        print(f"  -> Could not find main content block on {url}. Skipping.")
        continue

    try:
        # The president's name is in an h3 tag with class 'diet-title' within the main content
        president_tag = main_content.find('h3', class_='diet-title')
        if president_tag and president_tag.find('a'):
            president_name = president_tag.find('a').text.strip()
        else:
            # Fallback for pages that might have a different structure
            president_name_container = main_content.find('div', class_='field-docs-person')
            if president_name_container:
                 president_name = president_name_container.text.strip()
            else:
                 president_name = "Unknown"

        # The date is in a specific div within the main content
        date_div = main_content.find('div', class_='field-docs-start-date-time')
        if date_div and date_div.find('span', class_='date-display-single'):
            date_str = date_div.find('span', class_='date-display-single').text.strip()
        else:
            date_str = "Unknown"

        # The speech text is in a div with class 'field-docs-content' within the main content
        speech_content_div = main_content.find('div', class_='field-docs-content')
        if speech_content_div:
            # --- MODIFICATION START ---
            # Get the raw text first
            raw_text = speech_content_div.get_text(separator='\n', strip=True)

            # 1. Remove all bracketed annotations like [Applause], [Laughter], etc.
            #    The re.DOTALL flag ensures that this works even if brackets span multiple lines.
            cleaned_text = re.sub(r'\[.*?\]', '', raw_text, flags=re.DOTALL)

            # 2. Remove speaker attribution lines like "The President.", "Audience members.", etc.
            #    This looks for lines that start with common attribution words.
            #    The re.MULTILINE flag makes ^ and $ match the start/end of lines, not just the whole string.
            cleaned_text = re.sub(r'^\s*.*?(?:President|Speaker|Representative|Audience members?|Rep\.|Senator)\s*\..*$\n?', '', cleaned_text, flags=re.MULTILINE)

            # 3. Clean up any excess newlines that might result from the above steps
            speech_text = re.sub(r'\n\s*\n', '\n', cleaned_text).strip()
        else:
            speech_text = "Content not found."

        all_speeches_data.append({
            'president': president_name,
            'date': date_str,
            'url': url,
            'speech_text': speech_text
        })

    except AttributeError as e:
        print(f"  -> Could not parse data from {url}. Maybe the page structure is different. Error: {e}. Skipping.")
        continue  # Skip this speech if we can't parse it
    # --- MODIFICATION END ---

    # Be polite: wait for a short time between requests
    time.sleep(0.5)

# Step 5: Save to a DataFrame and then a CSV file
df = pd.DataFrame(all_speeches_data)

df['president'] = df['president'].str.replace(r'\s*\d+.*President of the United States.*', '', regex=True).str.strip()

# Convert the 'date' column to a proper datetime format
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Sort by date to have a chronological record
df = df.sort_values(by='date').reset_index(drop=True)

# Save the data so we don't have to scrape again!
df.to_csv('sotu_speeches.csv', index=False, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)
df.to_excel("sotu_speeches.xlsx", index=False, engine='openpyxl')

print("\nScraping complete!")
print(f"Successfully saved {len(df)} speeches to sotu_speeches.csv")
print("\nHere's a sample of the data:")
print(df.head())
print(df.tail())