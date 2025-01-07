import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel
import wandb
from training.evaluation_metrics import compute_ranking_metrics 
from src.configs.setup import load_config

config = load_config('src/configs/config.yaml')

wandb.login(key=config.get('wandb_key'))
wandb.init(project="MIND-RS", entity="MIND-RS", name="ALS_Evaluation")  # MIND-RS project should be set up as your default location on wandb

def evaluate_model(spark: SparkSession, als_model_path: str, test_data_path: str, k: int = 10):
    print(f"Loading ALS model from: {als_model_path}")
    als_model = ALSModel.load(als_model_path) 

    print("Generating recommendations...")
    user_recommendations = als_model.recommendForAllUsers(k)

    print("Formatting recommendations...")
    exploded_recommendations = user_recommendations.withColumn(
        "recommendations", F.explode(F.col("recommendations"))
    ).select(
        F.col("userId"),
        F.col("recommendations.itemId").alias("itemId"),
        F.col("recommendations.rating").alias("rating"),
    )

    print("Evaluating the model...")
    ranking_metrics = compute_ranking_metrics(exploded_recommendations, k)

    wandb.log(ranking_metrics)

    for metric, value in ranking_metrics.items():
        print(f"{metric}: {value}")

    print("Evaluation completed and metrics logged to WandB.")
