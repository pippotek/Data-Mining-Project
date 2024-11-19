import requests
import json
import time
from setup import load_config

# Load the configuration
config = load_config("src/config.yaml")

# Access specific values
if config:
    api_key = config.get('api_key', None)    

# Set the headers for authorization
headers = {'x-api-key': api_key}

# URL for the base API call (replace 'base_url' with the actual API endpoint)
base_url = 'https://api.newscatcherapi.com/v2/search'

# List of parameters to make multiple queries
params = [
    {
        'q': "Trump",
        'lang': "en",
        'to_rank': 10000,
        'page_size': 100,
        'page': 1,
    }
]

# Variable to store all found news articles
all_news_articles_mp = []

# Loop through each set of parameters to fetch results page by page
for separated_param in params:
    print(f'Query in use => {str(separated_param)}')

    # Wait for 1 second between each call to avoid hitting API rate limits
    time.sleep(1)

    # GET request to fetch the articles based on the current parameters
    response = requests.get(base_url, headers=headers, params=separated_param)
    results = response.json()

    if response.status_code == 200:
        print(f'Done for page number => {separated_param["page"]}')

        # If there are no articles in the response, continue to the next set of parameters
        if 'articles' not in results or not results['articles']:
            print(f"No articles found for page number => {separated_param['page']}")
            continue

        # Adding your parameters to each result to be able to explore afterward
        for i in results['articles']:
            i['used_params'] = str(separated_param)

        # Storing all found articles
        all_news_articles_mp.extend(results['articles'])

        print(f'Number of articles extracted for page {separated_param["page"]}: {len(results["articles"])}')

    else:
        # Print the error response if the API call fails
        print(results)
        print(f'ERROR: API call failed for page number => {separated_param["page"]}')

# Print the total number of extracted articles
print(f'Total number of extracted articles => {str(len(all_news_articles_mp))}')

# Save the content of all extracted articles to a JSON file
with open('trump_articles.json', 'w') as json_file:
    json.dump(all_news_articles_mp, json_file, indent=4)

print("Data has been saved to bitcoin_articles.json")
