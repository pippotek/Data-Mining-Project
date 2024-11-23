## News Article Processing and Embedding Pipeline

## Overview

The pipeline processes news articles through the following stages:

1. **Fetching Articles** : Fetch articles from an API using topic filters and time-based intervals.
2. **Cleaning and Deduplication** : Merge multiple datasets, remove duplicates, and identify duplicate entries.
3. **Storage in MongoDB** : Load cleaned articles into MongoDB for efficient access and further processing.
4. **Embedding with BERT** : Generate vector representations of articles using BERT embeddings through Spark NLP.

### Article document example:

{

    **"_id"**:{"$oid":"6741ec0c1f2d0c67050da7f0"},

    **"title"**:"Heavy rains trigger deadly landslide in south Peru",

    **"author"**:"",

    **"link"**:"https://sg.news.yahoo.com/heavy-rains-trigger-deadly-landslide-042511587.html",

    **"summary"**:"STORY: The landslide hit the district of Marcabamba in the Ayacucho region, the National Emergency Operations Centre (COEN) reported, leaving a trail of destruction. Officials said the two 				victims were an elderly woman and her daughter, who were buried in mud in their home.\nIn central Peru, floodwaters entered homes, destroying belongings and knocking down walls, according to eyewitnesses.\nA woman in Chanchamayo pleaded for help in an interview with local media.\"The water is coming from above and is coming all at once,\" she said. \"Look what has happened, all the walls of the neighboring house have fallen down.\"\nWhile in Apurimac, heavy rains caused the Chalhuanca River to overflow, cutting off access to the province's capital Aymaraes after the collapse of a highway.",

    **"excerpt"**:"STORY: The landslide hit the district of Marcabamba in the Ayacucho region, the National Emergency Operations Centre (COEN) reported, leaving a trail of destruction. Officials said the two 	victims…",

    **"published_date"**:"2024-01-27 04:25:11"

}

## Workflow

1. **Article Fetching** : Fetch raw data using  `prova_fetching.py`.
2. **Data Cleaning** : Process the raw data using  `cleaning.py` to remove duplicates.
3. **Data Storage** : Load cleaned data into MongoDB using  `provamongo.py`.
4. **Embedding** : Generate embeddings with** **`clean_embed.py` and update MongoDB.

## Files

### 1. `prova_fetching.py`

 **Purpose** : Fetches articles from a news API based on specified topics and time intervals.

* Implements rate-limiting and handles source exclusions to avoid overloading.
* Articles are saved in JSON format for further processing.

### 2.` cleaning.py`

 **Purpose** : Cleans and deduplicates articles from JSON files.

* Merges multiple JSON files into one dataset.
* Identifies and logs duplicate articles by their IDs.
* Produces a deduplicated dataset saved as** **`deduplicated_data.json`.

### 3. `provamongo.py`

 **Purpose** : Loads cleaned articles into MongoDB.

* Connects to MongoDB using credentials from the configuration file.
* Inserts articles in batches for better performance.
* Maps fields in the JSON to MongoDB schema, using** **`_id` as the primary key.

### 4. `clean_embed.py`

 **Purpose** : Processes and embeds articles using Spark NLP.

* Reads articles from MongoDB.
* Preprocesses the text (e.g., cleaning, tokenizing).
* Uses a pre-trained BERT model to generate vector embeddings.
* Outputs embeddings back to MongoDB for downstream tasks.

### 5. `setup.py`

 **Purpose** : Provides utility functions for loading configuration.

* Reads sensitive data like API keys and database connection strings from a YAML file.
* Used across the pipeline for consistent configuration management.
