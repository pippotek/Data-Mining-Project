from src.algorithms.cbrs.cbrs_utils import * 
from src.utilities.data_utils import * 
from src.algorithms.cbrs.clean_embed import main_embedding
from cbrs_utils import *
import logging
from pyspark.sql import SparkSession



logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def main():

    # MongoDB Configuration
    MONGO_URI = "mongodb://root:example@mongodb:27017"
    DATABASE_NAME = "mind_news"
    news_embeddings_collection = "news_combined_embeddings"
    behaviors_train_collection = "behaviors_train"
    behaviors_test_collection = "behaviors_valid"
    RECOMMENDATIONS_COLLECTION = "cbrs_recommendations"

        # Initialize Spark Session with Optimized Configurations
    spark = (SparkSession.builder
                .appName("CBRS")
                .config("spark.master", "local[*]")
                .config("spark.ui.port", "4040")  
                .config("spark.jars.packages",
                        "com.johnsnowlabs.nlp:spark-nlp_2.12:5.5.1,"
                        "org.mongodb.spark:mongo-spark-connector_2.12:10.2.0")
                .config("spark.mongodb.read.connection.uri", MONGO_URI)
                .config("spark.mongodb.write.connection.uri", MONGO_URI)
                .config("spark.mongodb.output.writeConcern.w", "1")
                .getOrCreate())
    logger.info("Spark Session initialized.")
        
    # If you have a main_embedding function, ensure it's correctly defined and called
    main_embedding(spark)   
    
    
    try:
        # 1. Load data
        news_embeddings_df, behaviors_train_df, behaviors_test_df = load_data(
            spark, MONGO_URI, DATABASE_NAME, news_embeddings_collection, behaviors_train_collection, behaviors_test_collection
        )
        
        # Check if any DataFrame is empty
        if news_embeddings_df.rdd.isEmpty():
            logger.error("News Embeddings DataFrame is empty. Exiting pipeline.")
            return
        if behaviors_train_df.rdd.isEmpty():
            logger.error("Behaviors Train DataFrame is empty. Exiting pipeline.")
            return
        if behaviors_test_df.rdd.isEmpty():
            logger.error("Behaviors Test DataFrame is empty. Exiting pipeline.")
            return
        print(news_embeddings_df.count())
        print(behaviors_test_df.count()) 
#         # 2. Preprocess news embeddings
#         news_embeddings_df = preprocess_news_embeddings(news_embeddings_df)
        
#         # 3. Create user profiles
#         user_profiles_df = create_user_profiles_with_pandas_udaf(behaviors_train_df, news_embeddings_df)
        
#         # Check if user_profiles_df is empty
#         if user_profiles_df.rdd.isEmpty():
#             logger.error("User Profiles DataFrame is empty after aggregation. Exiting pipeline.")
#             return
        
#         # 4. Generate recommendations via Approximate Nearest Neighbors
#         recommendations_df = compute_recommendations_ann(
#             user_profiles_df=user_profiles_df,
#             news_embeddings_df=news_embeddings_df,
#             top_k=10
#         )
        
#         # 5. Check recommendations before saving
#         recommendations_count = recommendations_df.count()
#         logger.info(f"Recommendations Count: {recommendations_count}")
#         recommendations_df.show(5, truncate=False)
        
#         if recommendations_count == 0:
#             logger.warning("Recommendations DataFrame is empty. Skipping save operation.")
#         else:
#             # 6. Save the recommendations to MongoDB
#             logger.info("Saving recommendations to MongoDB...")
#             recommendations_df.write \
#                 .format("com.mongodb.spark.sql.DefaultSource") \
#                 .mode("append") \
#                 .option("spark.mongodb.output.uri", "mongodb://localhost:27017/news_db.news_recommendations") \
#                 .save()
            
#             # Force Spark to materialize tasks before stopping
#             recommendations_df.rdd.count()
#             logger.info("Recommendations computed and persisted.")
        
#         # 7. Optionally, evaluate recommendations
#         # metrics = evaluate_recommendations(recommendations_df, behaviors_test_df)
#         # logger.info("Evaluation Metrics: %s", metrics)
    
    except Exception as e:
        logger.exception("An error occurred during the CBRS pipeline: %s", e)
    
    finally:
        spark.stop()
        logger.info("Spark session stopped.")

if __name__ == "__main__":
    main()