from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import json
from setup import load_config

### TRIAL SCRIPT TO LOAD DATA INTO MONGO

# Load the configuration
config = load_config("src/config.yaml")

# Access specific values
if config:
    uri = config.get('db_connection_string', None)
    print(f"Database Connection String: {uri}")

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

with open('/home/filippo/Desktop/Uni/Data-Mining-Project/bitcoin_articles.json', 'r', encoding='utf-8') as file:
    data = json.load(file)  # Load JSON data as a Python list

# Filter the data to keep only the specified fields
filtered_data = [
    {
        "_id": article["_id"],  # Renaming `id` to `_id` to use as MongoDB's primary key if it exists
        "title": article.get("title"),
        "author": article.get("author"),
        "link": article.get("link"),
        "summary": article.get("summary"),
        "excerpt": article.get("excerpt")
    }
    for article in data
]

# Connect to MongoDB  # Use your MongoDB URI if it's different
db = client['NEWS']  # Replace with your database name

# Insert data into the 'articles' collection
articles_collection = db['articles']
result = articles_collection.insert_many(filtered_data)  # Insert the list of articles

# Print out the inserted IDs to verify
print("Inserted document IDs:", result.inserted_ids)