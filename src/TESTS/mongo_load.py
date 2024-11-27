from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from pymongo.errors import BulkWriteError
import pandas as pd
from setup import load_config

### FUNCTIONS ###

def batch_insert(collection, data, batch_size=100):
    """Insert data into MongoDB in batches."""
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        try:
            result = collection.insert_many(batch, ordered=False)
            print(f"Inserted {len(result.inserted_ids)} documents successfully in batch {i // batch_size + 1}.")
        except BulkWriteError as bwe:
            print(f"Error inserting documents in batch {i // batch_size + 1}: {bwe.details}")

def load_tsv_to_dataframe(filepath, column_names):
    """Load a TSV file into a Pandas DataFrame."""
    return pd.read_csv(filepath, sep='\t', names=column_names, header=None)

### SCRIPT ###

# Load the configuration
config = load_config("src/config.yaml")

# Access specific values
if config:
    uri = config.get('db_connection_string', None)
    print(f"Database Connection String: {uri}")

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

# Define file paths and column mappings
interactions_file = 'interactions.tsv'
news_file = 'news.tsv'

# Define MongoDB database and collections
db = client['datamining']  # Replace with your database name
interactions_collection = db['interactions']
news_collection = db['news']

### Load and Process Interactions Data ###
interactions_columns = ['session_id', 'userid', 'timestamp', 'articleids', 'additional_data']
interactions_df = load_tsv_to_dataframe(interactions_file, interactions_columns)

# Group by userid and aggregate articleids
interactions_grouped = interactions_df.groupby('userid').agg({
    'articleids': lambda x: list(set(" ".join(x.dropna()).split()))  # Aggregate all article IDs for each user
}).reset_index()

# Convert the grouped DataFrame to a list of dictionaries
interactions_data_filtered = [
    {
        "_id": row["userid"],  # Use userid as the MongoDB primary key
        "articleids": row["articleids"]
    }
    for _, row in interactions_grouped.iterrows()
]

# Insert grouped interactions data
print("Inserting grouped interactions data...")
batch_insert(interactions_collection, interactions_data_filtered, batch_size=100)

### Load and Process News Data ###
news_columns = ['articleid', 'topic', 'detailed_topic', 'title', 'description', 'content', 'link', 'entities', 'additional_data']
news_df = load_tsv_to_dataframe(news_file, news_columns)

# Simplify news data
news_data_filtered = [
    {
        "_id": row["articleid"],  # Use articleid as the MongoDB primary key
        "title": row["title"],
        "content": row["content"]
    }
    for _, row in news_df.iterrows()
]

# Insert news data
print("Inserting simplified news data...")
batch_insert(news_collection, news_data_filtered, batch_size=100)

print("Data import completed successfully!")
