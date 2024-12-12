# NOTE: SparkRankingEvaluation used in this script assumes a specific format for 'predictions_and_labels'. 
# Make sure that this format is correct beyond the initial tests.

import pyspark.sql.functions as F
from pyspark.ml.evaluation import RegressionEvaluator
from recommenders.evaluation.spark_evaluation import SparkRankingEvaluation
from src.algorithms.als.als_configs import EVAL_CONFIG


# Compute ranking-based metrics for the ALS model.
def compute_ranking_metrics(predictions, top_k=None):
    if top_k is None:
        top_k = EVAL_CONFIG["k"]

    prediction_and_labels = predictions \
        .groupBy("userId") \
        .agg(F.collect_list(F.struct("prediction", "newsId")).alias("predictions")) \
        .select("userId", "predictions")

    # Instantiate SparkRankingEvaluation
    ranking_eval = SparkRankingEvaluation(prediction_and_labels, k=k)

    precision_at_k = ranking_eval.precision_at_k()
    recall_at_k = ranking_eval.recall_at_k()
    ndcg_at_k = ranking_eval.ndcg_at_k()
    mean_avg_precision = ranking_eval.map()

    return {
        "Precision@K": precision_at_k,
        "Recall@K": recall_at_k,
        "NDCG@K": ndcg_at_k,
        "Mean Average Precision": mean_avg_precision,
    }


# Compute regression-based metrics for the ALS model.
def compute_regression_metrics(predictions):
    # Initialize evaluators
    rmse_evaluator = RegressionEvaluator(metricName="rmse", labelCol="clicked", predictionCol="prediction")
    mae_evaluator = RegressionEvaluator(metricName="mae", labelCol="clicked", predictionCol="prediction")

    rmse = rmse_evaluator.evaluate(predictions)
    mae = mae_evaluator.evaluate(predictions)

    return {
        "RMSE": rmse,
        "MAE": mae,
    }
