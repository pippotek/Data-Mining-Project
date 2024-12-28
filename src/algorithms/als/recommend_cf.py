import sys
import os

# Dynamically add 'src' to sys.path
src_path = os.path.abspath(os.path.join(os.getcwd(), "../../../"))
sys.path.append(src_path)
from utilities.data_utils import write_to_mongodb

from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel
from pyspark.sql.functions import explode, col


#MongoDB configuration

MONGO_URI = "mongodb://localhost:27017"  
DATABASE_NAME = "mind_news"
COLLECTION_NAME = "als_recommendations" 

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("ALSModelLoad") \
    .config("spark.master", "local[*]") \
    .config("spark.executor.memory", "2g") \
    .config("spark.mongodb.input.uri", MONGO_URI) \
    .config("spark.mongodb.output.uri", MONGO_URI) \
    .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:10.1.1") \
    .getOrCreate()

current_working_directory = os.getcwd()
base_directory = os.path.abspath(os.path.join(current_working_directory, "../../../"))
model_path = os.path.join(base_directory, "saved_models", "als_model")

# Load the ALS model
als_model = ALSModel.load(model_path)  

# Perform recommendations
recommendations = als_model.recommendForAllUsers(numItems=10)
recommendations.show()

# Explode the 'recommendations' column
exploded_recommendations = recommendations.select(
    col("userId"),
    explode(col("recommendations")).alias("recommendation")
)

# Extract 'itemId' and 'rating' from the 'recommendation' column
final_recommendations = exploded_recommendations.select(
    col("userId"),
    col("recommendation.newsId").alias("recommendation"),
    col("recommendation.rating").alias("rating")
)

# Show the transformed DataFrame
final_recommendations.show(truncate=False)


### TO DO:

# Upload recommendations to MongoDB
#write_to_mongodb(final_recommendations, MONGO_URI, DATABASE_NAME, COLLECTION_NAME)

#print("Recommendations uploaded to MongoDB successfully!")  