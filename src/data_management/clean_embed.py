from pyspark.sql import SparkSession
from setup import load_config
from pyspark.sql.functions import concat_ws, col, regexp_replace, lower
from sparknlp.base import DocumentAssembler, EmbeddingsFinisher
from sparknlp.annotator import Tokenizer as SparkNLPTokenizer, BertEmbeddings, StopWordsCleaner
from pyspark.ml import Pipeline  
import sparknlp

# --- Configuration Loading ---
config = load_config("src/config.yaml")

if config:
    uri = config.get('db_connection_string', None)
    if not uri:
        raise ValueError("Database connection string is missing in the configuration.")
    print(f"Database Connection String: {uri}")
else:
    raise ValueError("Configuration could not be loaded.")

# --- Spark Session Initialization ---
spark = SparkSession.builder \
    .appName("Text Embedding") \
    .master("local[*]") \
    .config("spark.driver.memory", "16G") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .config("spark.kryoserializer.buffer.max", "2000M") \
    .config("spark.driver.maxResultSize", "0") \
    .config("spark.jars.packages", 
            "com.johnsnowlabs.nlp:spark-nlp_2.12:5.5.1,"
            "org.mongodb.spark:mongo-spark-connector_2.12:10.2.0") \
    .config("spark.mongodb.read.connection.uri", uri) \
    .config("spark.mongodb.write.connection.uri", uri) \
    .getOrCreate()

# --- Verify Spark NLP Installation ---
print("Spark NLP version:", sparknlp.version())

# Set log level to reduce verbosity (optional)
spark.sparkContext.setLogLevel("WARN")

# --- Data Loading ---
df = spark.read \
    .format("mongodb") \
    .option("database", "datamining") \
    .option("collection", "news") \
    .load()

df = df.select("_id", "title", "summary", "excerpt")

# --- Data Preprocessing ---
df = df.withColumn("combined_text", concat_ws(" ", col("title"), col("summary"), col("excerpt"))) \
       .withColumn("clean_text", lower(regexp_replace(col("combined_text"), "[^a-zA-Z0-9 ]", "")))

# --- Spark NLP Pipeline Setup ---
document_assembler = DocumentAssembler() \
    .setInputCol("clean_text") \
    .setOutputCol("document")

tokenizer = SparkNLPTokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token")

stop_words_cleaner = StopWordsCleaner() \
    .setInputCols(["token"]) \
    .setOutputCol("clean_tokens")

# Initialize BERT Embeddings
bert_embeddings = BertEmbeddings.pretrained("small_bert_L2_768", "en") \
    .setInputCols(["document", "clean_tokens"]) \
    .setOutputCol("embeddings")

# Initialize EmbeddingsFinisher with renamed output column
embeddings_finisher = EmbeddingsFinisher() \
    .setInputCols(["embeddings"]) \
    .setOutputCols(["embedding"]) \
    .setOutputAsVector(False)  # Ensures embeddings are output as arrays

# Assemble the pipeline
nlp_pipeline = Pipeline(stages=[
    document_assembler,
    tokenizer,
    stop_words_cleaner,
    bert_embeddings,
    embeddings_finisher  # Use EmbeddingsFinisher instead of Finisher
])

# --- Pipeline Fitting and Transformation ---
nlp_model = nlp_pipeline.fit(df)
processed_df = nlp_model.transform(df)

# --- Result Selection ---
# Select only '_id' and 'embedding' for the update
result_df = processed_df.select("_id", "embedding")
result_df.show()


# # --- Writing Results Back to MongoDB with Partial Updates ---
# result_df.write \
#     .format("mongodb") \
#     .option("database", "datamining") \
#     .option("collection", "news_embeddings") \
#     .mode("append") \
#     .save()

# --- Graceful Shutdown ---
spark.stop()
