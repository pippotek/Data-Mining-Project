import os
import pandas as pd
from pymongo import MongoClient
from src.data_management.mind import download_mind
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
mind_type = "small"  # "demo", "small", or "large"
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
    """Load a TSV file into a MongoDB collection with headers if not already loaded."""
    collection = db[collection_name]

    # Check if the collection already has data
    if collection.estimated_document_count() > 0:
        print(f"Collection '{collection_name}' already contains data. Skipping load.")
        return

    # Read the TSV file into a pandas DataFrame with headers
    print(f"Reading {tsv_file} for collection {collection_name}...")
    df = pd.read_csv(tsv_file, sep="\t", names=headers, quoting=3)

    # Convert DataFrame to dictionary records
    records = df.to_dict(orient="records")

    # Insert records into MongoDB
    if records:
        collection.delete_many({})  # Clear existing data if required (optional)
        result = collection.insert_many(records)
        print(f"Inserted {len(result.inserted_ids)} records into '{collection_name}' collection.")
    else:
        print(f"No records found in {tsv_file} for {collection_name}.")


def main():
    # Connect to MongoDB
    db = connect_to_mongo(MONGO_URI, DB_NAME)

    # Check if data already exists in MongoDB
    if db[BEHAVIORS_TRAIN_COLLECTION].estimated_document_count() > 0 and db[NEWS_TRAIN_COLLECTION].estimated_document_count() > 0:
        print("Training data already exists in MongoDB. Skipping download and loading for training data.")
    else:
        print("Downloading and extracting MIND training dataset...")
        train_zip, _ = download_mind(size=mind_type, dest_path=data_path)
        train_dir = os.path.join(data_path, 'train')
        unzip_file(train_zip, train_dir, clean_zip_file=False)

        # Paths to training TSV files
        behaviors_train_tsv = os.path.join(train_dir, "behaviors.tsv")
        news_train_tsv = os.path.join(train_dir, "news.tsv")

        # Load training data into MongoDB
        load_tsv_to_mongo(db, BEHAVIORS_TRAIN_COLLECTION, behaviors_train_tsv, BEHAVIORS_HEADERS)
        load_tsv_to_mongo(db, NEWS_TRAIN_COLLECTION, news_train_tsv, NEWS_HEADERS)

    if db[BEHAVIORS_VALID_COLLECTION].estimated_document_count() > 0 and db[NEWS_VALID_COLLECTION].estimated_document_count() > 0:
        print("Validation data already exists in MongoDB. Skipping download and loading for validation data.")
    else:
        print("Downloading and extracting MIND validation dataset...")
        _, valid_zip = download_mind(size=mind_type, dest_path=data_path)
        valid_dir = os.path.join(data_path, 'valid')
        unzip_file(valid_zip, valid_dir, clean_zip_file=False)

        # Paths to validation TSV files
        behaviors_valid_tsv = os.path.join(valid_dir, "behaviors.tsv")
        news_valid_tsv = os.path.join(valid_dir, "news.tsv")

        # Load validation data into MongoDB
        load_tsv_to_mongo(db, BEHAVIORS_VALID_COLLECTION, behaviors_valid_tsv, BEHAVIORS_HEADERS)
        load_tsv_to_mongo(db, NEWS_VALID_COLLECTION, news_valid_tsv, NEWS_HEADERS)

if __name__ == "__main__":
    main()