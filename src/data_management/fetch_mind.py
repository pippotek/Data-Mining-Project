import os
import pandas as pd
from pymongo import MongoClient
from recommenders.datasets.mind import download_mind
from recommenders.datasets.download_utils import unzip_file
from tempfile import TemporaryDirectory

# MongoDB connection details
MONGO_URI = "mongodb://root:example@mongodb:27017"
DB_NAME = "mind_news"
BEHAVIORS_TRAIN_COLLECTION = "behaviors_train"
BEHAVIORS_VALID_COLLECTION = "behaviors_valid"
NEWS_TRAIN_COLLECTION = "news_train"
NEWS_VALID_COLLECTION = "news_valid"

# MIND dataset parameters
mind_type = "demo"  # "demo", "small", or "large"
tmpdir = TemporaryDirectory()
data_path = tmpdir.name

# Define headers for the TSV files
BEHAVIORS_HEADERS = ["impression_id", "user_id", "time", "history", "impressions"]
NEWS_HEADERS = ["news_id", "category", "subcategory", "title", "abstract", "url", "title_entities", "text_entities"]

def connect_to_mongo(uri, db_name):
    """Connect to MongoDB and return the database object."""
    client = MongoClient(uri)
    db = client[db_name]
    return db

def load_tsv_to_mongo(db, collection_name, tsv_file, headers):
    """Load a TSV file into a MongoDB collection with headers."""
    collection = db[collection_name]
    
    # Read the TSV file into a pandas DataFrame with headers
    print(f"Reading {tsv_file} for collection {collection_name}...")
    df = pd.read_csv(tsv_file, sep="\t", names=headers, quoting=3)
    
    # Convert DataFrame to dictionary records
    records = df.to_dict(orient="records")
    
    # Insert records into MongoDB
    if records:
        collection.delete_many({})  # Clear existing data
        result = collection.insert_many(records)
        print(f"Inserted {len(result.inserted_ids)} records into '{collection_name}' collection.")
    else:
        print(f"No records found in {tsv_file} for {collection_name}.")

def main():
    # Download and unzip the MIND dataset
    print("Downloading and extracting MIND dataset...")
    train_zip, valid_zip = download_mind(size=mind_type, dest_path=data_path)
    train_dir = os.path.join(data_path, 'train')
    valid_dir = os.path.join(data_path, 'valid')
    unzip_file(train_zip, train_dir, clean_zip_file=False)
    unzip_file(valid_zip, valid_dir, clean_zip_file=False)

    # Paths to training and validation TSV files
    behaviors_train_tsv = os.path.join(train_dir, "behaviors.tsv")
    news_train_tsv = os.path.join(train_dir, "news.tsv")

    behaviors_valid_tsv = os.path.join(valid_dir, "behaviors.tsv")
    news_valid_tsv = os.path.join(valid_dir, "news.tsv")

    # Connect to MongoDB
    db = connect_to_mongo(MONGO_URI, DB_NAME)

    # Load TSV files into MongoDB with headers
    load_tsv_to_mongo(db, BEHAVIORS_TRAIN_COLLECTION, behaviors_train_tsv, BEHAVIORS_HEADERS)
    load_tsv_to_mongo(db, NEWS_TRAIN_COLLECTION, news_train_tsv, NEWS_HEADERS)

    load_tsv_to_mongo(db, BEHAVIORS_VALID_COLLECTION, behaviors_valid_tsv, BEHAVIORS_HEADERS)
    load_tsv_to_mongo(db, NEWS_VALID_COLLECTION, news_valid_tsv, NEWS_HEADERS)

if __name__ == "__main__":
    main()
