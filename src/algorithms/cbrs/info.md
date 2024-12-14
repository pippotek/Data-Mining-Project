To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
spark-cbrs  | INFO:__main__:Spark Session initialized successfully.
spark-cbrs  | INFO:__main__:Loading data from MongoDB...
INFO:__main__:Train records: 51282
spark-cbrs  | INFO:__main__:Validation records: 42416
spark-cbrs  | INFO:__main__:Combining train and validation data...
spark-cbrs  | INFO:__main__:Dropping duplicate news articles based on 'news_id'...
spark-cbrs  | INFO:__main__:Unique records after deduplication: 65238
spark-cbrs  | INFO:__main__:Preprocessing data...
spark-cbrs  | INFO:__main__:Setting up Spark NLP pipeline...
spark-cbrs  | small_bert_L2_768 download started this may take some time.

# Content-Based Recommendation System

## Overview

**Content-Based Recommendation Systems** are a specific type of recommendation system that focuses on the features of items and user profiles. These systems suggest items similar to those the user has interacted with in the past by analyzing the content or attributes of the items (in our case news).

### Example Use Cases:

* Suggesting similar news articles based on the content of articles a user has already read.
* Recommending products with descriptions or categories matching previously purchased items.

## Clean_embed.py

The script performs: 

### 1.** ****Data Loading and Combination**:

* Loads news data from two MongoDB collections:** **`<span>news_train</span>` and** **`<span>news_valid</span>`.
* Combines the data into a single dataset.
* Drops duplicate articles based on the unique** **`<span>news_id</span>` field.

### 2.** ****Preprocessing**:

* Combines the** **`<span>title</span>` and** **`<span>abstract</span>` fields into a single** **`<span>combined_text</span>` column.
* Cleans the** **`<span>combined_text</span>` by removing special characters and converting text to lowercase.

### 3.** ****Embedding Generation**:

* Uses the Spark NLP library to generate** ****BERT-based embeddings** for each news article.
* Tokenizes the cleaned text, removes stopwords, and computes embeddings for each article using BERT.
* Aggregates the token-level embeddings into a single** ****sentence embedding** using an averaging strategy.

### 4.** ****Batch Processing**:

* Processes the articles in batches to handle large datasets efficiently.
* Saves the embeddings for each article into a MongoDB collection (`<span>news_combined_embeddings</span>`).

---

## Script Details

### **1. Spark Initialization**

The script initializes a Spark session with appropriate configurations for memory and MongoDB connectivity.

```
spark = SparkSession.builder \
    .appName("Combine News and Generate Embeddings") \
    .master("local[*]") \
    .config("spark.driver.memory", "16G") \
    .config("spark.executor.memory", "16G") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .config("spark.mongodb.read.connection.uri", MONGO_URI) \
    .config("spark.mongodb.write.connection.uri", MONGO_URI) \
    .getOrCreate()
```

### **2. Data Loading and Combination**

* Reads news data from the** **`<span>news_train</span>` and** **`<span>news_valid</span>` collections in MongoDB.
* Combines the two datasets using** **`<span>union()</span>`.
* Removes duplicate rows based on the** **`<span>news_id</span>` field.

```
train_df = spark.read.format("mongodb").option("database", DATABASE_NAME).option("collection", TRAIN_COLLECTION).load()
valid_df = spark.read.format("mongodb").option("database", DATABASE_NAME).option("collection", VALID_COLLECTION).load()
combined_df = train_df.union(valid_df).dropDuplicates(["news_id"])
```

### **3. Preprocessing**

* Combines the** **`<span>title</span>` and** **`<span>abstract</span>` fields into a single** **`<span>combined_text</span>` field.
* Cleans the combined text by removing special characters and converting it to lowercase.

```
combined_df = combined_df.withColumn("combined_text", concat_ws(" ", col("title"), col("abstract")))
combined_df = combined_df.withColumn("clean_text", lower(regexp_replace(col("combined_text"), "[^a-zA-Z0-9 ]", "")))
```

### **4. NLP Pipeline for Embedding Generation**

The script sets up an NLP pipeline using Spark NLP to compute embeddings for each article:

1. **Text Processing**:
   * Tokenization
   * Stopword removal
2. **BERT Embeddings**:
   * Computes token-level embeddings using a pre-trained BERT model (`<span>small_bert_L2_768</span>`).
   * Aggregates token-level embeddings into a single sentence embedding using the "average" pooling strategy.

```
nlp_pipeline = Pipeline(stages=[
    DocumentAssembler().setInputCol("clean_text").setOutputCol("document"),
    Tokenizer().setInputCols(["document"]).setOutputCol("token"),
    StopWordsCleaner().setInputCols(["token"]).setOutputCol("clean_tokens"),
    BertEmbeddings.pretrained("small_bert_L2_768", "en")
        .setInputCols(["document", "clean_tokens"])
        .setOutputCol("embeddings"),
    SentenceEmbeddings()
        .setInputCols(["document", "embeddings"])
        .setOutputCol("sentence_embeddings")
        .setPoolingStrategy("AVERAGE"),
    EmbeddingsFinisher()
        .setInputCols(["sentence_embeddings"])
        .setOutputCols(["embedding"])
        .setOutputAsVector(False)
])
```

### **5. Batch Processing**

* Processes the dataset in batches to handle large data efficiently.
* Generates and saves embeddings for each batch into the** **`<span>news_combined_embeddings</span>` collection in MongoDB.

```
for start in range(0, total_records, BATCH_SIZE):
    batch_df = combined_df.filter((col("row_num") > start) & (col("row_num") <= end))
    processed_batch_df = nlp_model.transform(batch_df)
    news_embeddings_df = processed_batch_df.select("news_id", "embedding")
    news_embeddings_df.write \
        .format("mongodb") \
        .option("database", DATABASE_NAME) \
        .option("collection", EMBEDDINGS_COLLECTION) \
        .mode("append") \
        .save()
```

---

## To-Do: Next Steps

### **1. Behaviors Preprocessing**

* Load user behavior data from the** **`<span>behaviors</span>` collection.
* Preprocess user click and impression logs.
* Create user profiles based on the articles they have interacted with.

### **2. Model Building**

* Build a similarity-based model using the computed embeddings.
* Compute similarity between articles using cosine similarity.
* Rank and recommend articles based on similarity scores.

### **3. Evaluation**

* Evaluate the recommendation system using metrics like Precision, Recall, and NDCG (Normalized Discounted Cumulative Gain).

---

## Conclusion

This script provides the foundation for a content-based recommendation system by processing and embedding news articles. Future steps include behavior preprocessing, model development, and evaluation to complete the recommendation pipeline.
