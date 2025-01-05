import logging
from typing import List

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, concat_ws, regexp_replace, lower, flatten, row_number
from pyspark.sql.window import Window
from pyspark.sql.types import ArrayType
from pyspark.ml import Pipeline
from src.utilities.data_utils import wait_for_data, fetch_data_from_mongo, write_to_mongodb
from sparknlp.base import DocumentAssembler, EmbeddingsFinisher
from sparknlp.annotator import Tokenizer, BertEmbeddings, StopWordsCleaner, SentenceEmbeddings

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


def init_spark_session() -> SparkSession:
    """
    Initialize and return a SparkSession configured for the job.
    """
    logger.info("Initializing Spark Session...")
    spark = (SparkSession.builder
             .appName("Combine News and Generate Embeddings")
             .master("local[*]")
             .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
             .config("spark.kryoserializer.buffer.max", "2000M")
             .config("spark.driver.maxResultSize", "0")
             .config("spark.jars.packages",
                     "com.johnsnowlabs.nlp:spark-nlp_2.12:5.5.1,"
                     "org.mongodb.spark:mongo-spark-connector_2.12:10.2.0")
             .config("spark.mongodb.read.connection.uri", MONGO_URI)
             .config("spark.mongodb.write.connection.uri", MONGO_URI)
             .config("spark.mongodb.output.writeConcern.w", "1")
             .getOrCreate())
    logger.info("Spark Session initialized successfully.")
    spark.conf.set("spark.sql.debug.maxToStringFields", 50000)  # Replace 1000 with your desired limit

    return spark


def preprocess_data(combined_df: DataFrame, already_processed: DataFrame) -> DataFrame:
    """
    Combine the train and validation dataframes, filter already processed IDs, and preprocess.
    
    :param combined_df: DataFrame with combined train and valid data
    :param already_processed: DataFrame with already processed IDs
    :return: Preprocessed DataFrame ready for embedding
    """
    logger.info("Dropping duplicate news articles based on 'news_id'...")
    combined_df = combined_df.dropDuplicates(["news_id"])

    logger.info("Filtering out already processed articles using a left_anti join...")
    if already_processed.count() > 0:
        filtered_df = combined_df.join(already_processed.select("news_id"), on="news_id", how="left_anti")
    else:
        filtered_df = combined_df

    count_after_filter = filtered_df.count()
    logger.info("Records to process after filtering: %d", count_after_filter)
    if count_after_filter == 0:
        logger.info("All articles already processed successfully.")
        return filtered_df

    logger.info("Preprocessing data (cleaning text)...")
    filtered_df = filtered_df.withColumn(
        "combined_text", concat_ws(" ", col("title"), col("abstract"))
    ).withColumn(
        "clean_text",
        lower(regexp_replace(col("combined_text"), "[^a-zA-Z0-9 ]", ""))
    )
    return filtered_df


def build_nlp_pipeline() -> Pipeline:
    """
    Build and return a Spark NLP pipeline for generating BERT embeddings.
    
    :return: Spark ML Pipeline for NLP tasks
    """
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
    return nlp_pipeline


def process_batches(spark: SparkSession, df: DataFrame, nlp_model, batch_size: int):
    """
    Process the filtered DataFrame in batches to generate embeddings and write them to MongoDB.
    
    :param spark: SparkSession
    :param df: Preprocessed DataFrame to process
    :param nlp_model: Fitted NLP pipeline model
    :param batch_size: Size of each batch
    """
    logger.info("Preparing for batch processing...")
    window_spec = Window.orderBy("news_id")
    df = df.withColumn("row_num", row_number().over(window_spec))

    total_records = df.count()
    logger.info("Total records to process: %d. Processing in batches of %d.", total_records, batch_size)

    for start in range(0, total_records, batch_size):
        end = start + batch_size
        logger.info("Processing batch from %d to %d", start + 1, end)

        batch_df = df.filter((col("row_num") > start) & (col("row_num") <= end))

        processed_batch_df = nlp_model.transform(batch_df)

        # Flatten embedding if nested
        embedding_schema = processed_batch_df.schema["embedding"].dataType
        if isinstance(embedding_schema, ArrayType) and isinstance(embedding_schema.elementType, ArrayType):
            processed_batch_df = processed_batch_df.withColumn("embedding", flatten(col("embedding")))

        # Convert embedding array to string
        processed_batch_df = processed_batch_df.withColumn(
            "embedding_string", concat_ws(",", col("embedding"))
        )

        news_embeddings_df = processed_batch_df.select("news_id", "embedding_string")

        # Write this batch to MongoDB
        write_to_mongodb(news_embeddings_df, MONGO_URI, DATABASE_NAME, EMBEDDINGS_COLLECTION)

        logger.info("Batch %d to %d processed and saved.", start + 1, end)


def main_embedding(spark):
    wait_for_data(
            uri=MONGO_URI,
            db_name="mind_news",
            collection_names=["behaviors_train", "news_train", "behaviors_valid", "news_valid"],
            check_field= "_id"
            )
    try:
        # Load data
        train_df = fetch_data_from_mongo(spark, MONGO_URI, DATABASE_NAME, TRAIN_COLLECTION)
        valid_df = fetch_data_from_mongo(spark, MONGO_URI, DATABASE_NAME, VALID_COLLECTION)
        combined_df = train_df.union(valid_df)

        # Get processed IDs as a DataFrame
        already_processed = fetch_data_from_mongo(spark, MONGO_URI, DATABASE_NAME, EMBEDDINGS_COLLECTION)


        # Preprocess data using a left_anti join
        filtered_df = preprocess_data(combined_df, already_processed)
        if filtered_df.count() == 0:
            # Nothing to process
            return

        # Build and fit the NLP pipeline
        nlp_pipeline = build_nlp_pipeline()
        nlp_model = nlp_pipeline.fit(filtered_df)
        logger.info("NLP pipeline model fitted on the filtered dataset.")

        # Process data in batches
        process_batches(spark, filtered_df, nlp_model, BATCH_SIZE)

        logger.info("All batches processed successfully.")
    except Exception as e:

        logging.error(f"Error generating embeddings: {e}")
    
    spark.stop()



