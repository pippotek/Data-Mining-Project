import os
from pyspark.ml.recommendation import ALS
from pyspark.ml.recommendation import ALSModel
from configs.als_configs import ALS_CONFIG

def create_als_model():
    als = ALS(
        rank=ALS_CONFIG["rank"],
        maxIter=ALS_CONFIG["max_iter"],
        regParam=ALS_CONFIG["reg_param"],
        implicitPrefs=ALS_CONFIG["implicit_prefs"],
        alpha=ALS_CONFIG["alpha"],
        coldStartStrategy=ALS_CONFIG["cold_start_strategy"],
        userCol="userId", 
        itemCol="newsId", 
        ratingCol="clicked" # Make sure this is the final name of our binary column 
    )
    return als

def save_model(model, path):
    if not os.path.exists(path):
        os.makedirs(path)
    model.save(path)
    print(f"Model saved at {path}")

def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No model found at {path}")
    model = ALSModel.load(path)
    print(f"Model loaded from {path}")
    return model

def make_predictions(model, dataset):
    predictions = model.transform(dataset)
    return predictions