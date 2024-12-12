import os
from pyspark.ml.recommendation import ALS
from pyspark.ml.recommendation import ALSModel
from src.algorithms.als.als_configs import ALS_CONFIG

def create_als_model():
    als = ALS(
        rank=ALS_CONFIG["rank"],
        maxIter=ALS_CONFIG["max_iter"],
        regParam=ALS_CONFIG["reg_param"],
        implicitPrefs=ALS_CONFIG["implicit_prefs"],
        alpha=ALS_CONFIG["alpha"],
        userCol="userId", 
        itemCol="newsId", 
        ratingCol="clicked",
        coldStartStrategy=ALS_CONFIG["cold_start_strategy"] 
    )
    return als

def save_model(model: ALSModel, model_save_path: str):
    directory = os.path.dirname(model_save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)  
    try:
        model.write().overwrite().save(model_save_path)
        print(f"Model saved successfully at {model_save_path}")
    except Exception as e:
        print(f"Error saving the model: {e}")

def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No model found at {path}")
    model = ALSModel.load(path)
    print(f"Model loaded from {path}")
    return model

def make_predictions(model, dataset):
    predictions = model.transform(dataset)
    return predictions