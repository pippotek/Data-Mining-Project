import requests
import json
import time
from datetime import datetime, timedelta
from collections import defaultdict
from setup import load_config


### FETCHING SCRIPT, TO PERFECT

def load_api_key(config_path="src/config.yaml"):
    """Load API key from configuration file."""
    config = load_config(config_path)
    if config:
        return config.get('api_key', None)
    return None

def generate_time_intervals(start_date, end_date, num_intervals):
    """Generate equal time intervals between two dates."""
    total_seconds = int((end_date - start_date).total_seconds())
    interval_seconds = total_seconds // num_intervals
    intervals = []
    for i in range(num_intervals):
        interval_start = start_date + timedelta(seconds=i * interval_seconds)
        if i == num_intervals - 1:
            interval_end = end_date
        else:
            interval_end = start_date + timedelta(seconds=(i + 1) * interval_seconds - 1)
        intervals.append((
            interval_start.strftime('%Y-%m-%d %H:%M:%S'),
            interval_end.strftime('%Y-%m-%d %H:%M:%S')
        ))
    return intervals

def fetch_articles(api_key, intervals, topics, max_articles_per_source_per_interval):
    """Fetch articles from API over the date range for the specified topics with interval-based source exclusion."""
    headers = {'x-api-key': api_key}
    base_url = 'https://api.newscatcherapi.com/v2/search'
    all_articles = []

    for topic in topics:
        for interval_index, (interval_start_str, interval_end_str) in enumerate(intervals):
            print(f"\nFetching articles for topic '{topic}' in interval {interval_index + 1}/{len(intervals)}")
            # Initialize counts and exclusions for this interval
            source_counts = defaultdict(int)
            excluded_sources = set(['blogspot.com','yahoo.com', 'amazonaws.com'])  # Initial exclusions
            page = 11

            while page <= 15:
                # Update not_sources parameter
                not_sources_param = ','.join(excluded_sources)
                # Prepare parameters
                params = {
                    "q": "NOT fjaojaocjaofjwwada",
                    "lang": "en",
                    "countries": "US,GB,CA,AU",
                    "from": interval_start_str,
                    "to": interval_end_str,
                    "ranked_only": "True",
                    "to_rank": 5000,
                    "sort_by": "relevancy",
                    "page": page,
                    "page_size": 100,
                    "not_sources": not_sources_param,
                    "topic": topic
                }

                print(f'Processing: Topic => {topic}, Interval => {interval_index + 1}, Page => {page}')
                # Wait for 1 second between each call to avoid hitting API rate limits
                time.sleep(1)

                # GET request to fetch the articles based on the current parameters
                response = requests.get(base_url, headers=headers, params=params)
                if response.status_code != 200:
                    # Handle API error
                    print(f'API request failed with status code {response.status_code}')
                    break

                results = response.json()

                # If there are no articles in the response, break the loop
                if 'articles' not in results or not results['articles']:
                    print(f"No articles found for page number => {page}")
                    break  # No more pages to fetch

                articles_added = 0

                # Process each article
                for article in results['articles']:
                    source = article.get('clean_url')
                    if source:
                        source_counts[source] += 1
                        all_articles.append(article)
                        articles_added += 1
                    else:
                        # If no source info, just add the article
                        all_articles.append(article)
                        articles_added += 1

                print(f'Number of articles extracted for page {page}: {articles_added}')

                # After processing the page, check for sources that exceeded the limit
                for source, count in source_counts.items():
                    if count > max_articles_per_source_per_interval and source not in excluded_sources:
                        print(f"Source {source} exceeded limit in interval {interval_index + 1}. Adding to excluded_sources.")
                        excluded_sources.add(source)

                # If less than page_size articles were returned, no need to proceed to next page
                if len(results['articles']) < 100:
                    break  # No more pages

                page += 1  # Move to the next page

    return all_articles

def save_articles_to_json(articles, filename='sample_articles3.json'):
    """Save articles to a JSON file."""
    with open(filename, 'w') as json_file:
        json.dump(articles, json_file, indent=4)
    print(f"\nData has been saved to {filename}")

def main():
    # Load API key
    api_key = load_api_key()

    # Check if API key is available
    if not api_key:
        print("API key not found. Please check your configuration.")
        return

    # Define start and end dates
    start_date = datetime(2024, 1, 1)
    end_date = datetime.now()

    # Generate intervals (e.g., 10 equal intervals)
    num_intervals = 10
    intervals = generate_time_intervals(start_date, end_date, num_intervals)

    # Define topics
    topics = ['world', 'politics']

    # Define maximum articles per source per interval
    max_articles_per_source_per_interval = 80  # Limit to 100 articles per source per interval

    # Fetch articles
    all_articles = fetch_articles(api_key, intervals, topics, max_articles_per_source_per_interval)

    # Print total articles fetched
    print(f'\nTotal number of extracted articles => {len(all_articles)}')

    # Save collected articles to JSON
    save_articles_to_json(all_articles)

if __name__ == '__main__':
    main()
