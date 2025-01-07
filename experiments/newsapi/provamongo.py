from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from pymongo.errors import BulkWriteError
import json
from setup import load_config

### TRIAL SCRIPT TO LOAD DATA INTO MONGO (Batch Processing)

def batch_insert(collection, data, batch_size=100):
    """Insert data into MongoDB in batches."""
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        try:
            result = collection.insert_many(batch, ordered=False)
            print(f"Inserted {len(result.inserted_ids)} documents successfully in batch {i // batch_size + 1}.")
        except BulkWriteError as bwe:
            print(f"Error inserting documents in batch {i // batch_size + 1}: {bwe.details}")

# Load the configuration
config = load_config("src/config.yaml")

# Access specific values
if config:
    uri = config.get('db_connection_string', None)
    print(f"Database Connection String: {uri}")

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

# Load the data from the JSON file
with open('deduplicated_data.json', 'r', encoding='utf-8') as file:
    data = json.load(file)  # Load JSON data as a Python list

# Filter the data to keep only the specified fields
filtered_data = [
    {
        "_id": article["_id"],  # Renaming `id` to `_id` to use as MongoDB's primary key if it exists
        "title": article.get("title"),
        "author": article.get("author"),
        "link": article.get("link"),
        "summary": article.get("summary"),
        "excerpt": article.get("excerpt"),
        "published_date" : article.get("published_date")
    }
    for article in data
]

# Connect to MongoDB
db = client['datamining']  # Replace with your database name
articles_collection = db['news']

# Insert data in batches
batch_insert(articles_collection, filtered_data, batch_size=100)
