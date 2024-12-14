import os
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, concat_ws, regexp_replace, lower, flatten, row_number
from pyspark.sql.window import Window
from pyspark.sql.types import ArrayType
from pyspark.ml import Pipeline
import sparknlp
from sparknlp.base import DocumentAssembler, EmbeddingsFinisher
from sparknlp.annotator import Tokenizer, BertEmbeddings, StopWordsCleaner
from sparknlp.annotator import SentenceEmbeddings

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MongoDB Configuration
MONGO_URI = "mongodb://root:example@mongodb:27017/admin"
DATABASE_NAME = "mind_news"
TRAIN_COLLECTION = "news_train"
VALID_COLLECTION = "news_valid"
EMBEDDINGS_COLLECTION = "news_combined_embeddings"

BATCH_SIZE = 5000  # Adjust based on available resources

# Initialize Spark Session
logger.info("Initializing Spark Session...")
spark = SparkSession.builder \
    .appName("Combine News and Generate Embeddings") \
    .master("local[*]") \
    .config("spark.driver.memory", "32G") \
    .config("spark.executor.memory", "32G") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .config("spark.kryoserializer.buffer.max", "2000M") \
    .config("spark.driver.maxResultSize", "0") \
    .config("spark.jars.packages",
            "com.johnsnowlabs.nlp:spark-nlp_2.12:5.5.1,"
            "org.mongodb.spark:mongo-spark-connector_2.12:10.2.0") \
    .config("spark.mongodb.read.connection.uri", MONGO_URI) \
    .config("spark.mongodb.write.connection.uri", MONGO_URI) \
    .config("spark.mongodb.output.writeConcern.w", "1") \
    .getOrCreate()

logger.info("Spark Session initialized successfully.")

# Load data from MongoDB
logger.info("Loading data from MongoDB...")
train_df = spark.read \
    .format("mongodb") \
    .option("uri", MONGO_URI) \
    .option("database", DATABASE_NAME) \
    .option("collection", TRAIN_COLLECTION) \
    .load()

valid_df = spark.read \
    .format("mongodb") \
    .option("uri", MONGO_URI) \
    .option("database", DATABASE_NAME) \
    .option("collection", VALID_COLLECTION) \
    .load()

logger.info("Train records: %d", train_df.count())
logger.info("Validation records: %d", valid_df.count())

# Combine train and valid data
logger.info("Combining train and validation data...")
combined_df = train_df.union(valid_df)

# Drop duplicate news based on 'news_id'
logger.info("Dropping duplicate news articles based on 'news_id'...")
combined_df = combined_df.dropDuplicates(["news_id"])
logger.info("Unique records after deduplication: %d", combined_df.count())

# Preprocessing
logger.info("Preprocessing data...")
combined_df = combined_df.withColumn(
    "combined_text", concat_ws(" ", col("title"), col("abstract"))
).withColumn(
    "clean_text",
    lower(regexp_replace(col("combined_text"), "[^a-zA-Z0-9 ]", ""))
)

# Setup NLP pipeline
logger.info("Setting up Spark NLP pipeline...")
document_assembler = DocumentAssembler().setInputCol("clean_text").setOutputCol("document")
tokenizer = Tokenizer().setInputCols(["document"]).setOutputCol("token")
stop_words_cleaner = StopWordsCleaner().setInputCols(["token"]).setOutputCol("clean_tokens")
bert_embeddings = BertEmbeddings.pretrained("small_bert_L2_768", "en") \
    .setInputCols(["document", "clean_tokens"]) \
    .setOutputCol("embeddings") \
    .setCaseSensitive(False)
sentence_embeddings = SentenceEmbeddings() \
    .setInputCols(["document", "embeddings"]) \
    .setOutputCol("sentence_embeddings") \
    .setPoolingStrategy("AVERAGE")
embeddings_finisher = EmbeddingsFinisher() \
    .setInputCols(["sentence_embeddings"]) \
    .setOutputCols(["embedding"]) \
    .setOutputAsVector(False)

nlp_pipeline = Pipeline(stages=[
    document_assembler,
    tokenizer,
    stop_words_cleaner,
    bert_embeddings,
    sentence_embeddings,
    embeddings_finisher
])

# Fit the pipeline
nlp_model = nlp_pipeline.fit(combined_df)
logger.info("NLP pipeline model fitted on the combined dataset.")

# Add a row number column for batching
window_spec = Window.orderBy("news_id")
combined_df = combined_df.withColumn("row_num", row_number().over(window_spec))

total_records = combined_df.count()
logger.info("Total unique records: %d. Processing in batches of %d.", total_records, BATCH_SIZE)

# Process in batches
for start in range(0, total_records, BATCH_SIZE):
    end = start + BATCH_SIZE
    logger.info("Processing batch from %d to %d", start + 1, end)

    # Filter batch
    batch_df = combined_df.filter((col("row_num") > start) & (col("row_num") <= end))

    # Transform the batch
    processed_batch_df = nlp_model.transform(batch_df)

    # Flatten embedding if nested
    embedding_schema = processed_batch_df.schema["embedding"].dataType
    if isinstance(embedding_schema, ArrayType) and isinstance(embedding_schema.elementType, ArrayType):
        processed_batch_df = processed_batch_df.withColumn("embedding", flatten(col("embedding")))

    # Convert embedding array to string
    processed_batch_df = processed_batch_df.withColumn(
        "embedding_string", concat_ws(",", col("embedding"))
    )

    # Select only required columns
    news_embeddings_df = processed_batch_df.select("news_id", "embedding_string")

    # Write this batch to MongoDB
    news_embeddings_df.write \
        .format("mongodb") \
        .option("uri", MONGO_URI) \
        .option("database", DATABASE_NAME) \
        .option("collection", EMBEDDINGS_COLLECTION) \
        .mode("append") \
        .save()

    logger.info("Batch %d to %d processed and saved.", start + 1, end)

logger.info("All batches processed successfully.")

# Stop Spark session
logger.info("Stopping Spark Session...")
spark.stop()
logger.info("Spark Session stopped.")
