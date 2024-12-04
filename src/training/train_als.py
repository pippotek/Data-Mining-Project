import wandb
from algorithms.als_utils import create_als_model, save_model, make_predictions
from configs.als_configs import ALS_CONFIG
from training.evaluation import evaluate_model
from data_management.data_utils import load_and_prepare_mind_dataset, preprocess_behaviors_mind
from utilities.logger import get_logger
from training.evaluation_metrics import compute_regression_metrics
#from recommenders.datasets.mind import download_mind, extract_mind


logger = get_logger(name="ALS_Training", log_file="logs/train_als.log")

#Load training and validation data based on the selected data source.
def load_training_data(spark,
                       data_source = "recommenders",  # "db", "recommenders", or "csv"
                       **kwargs):
    if data_source == "recommenders":
        
        logger.info("Loading and preprocessing MIND dataset...")
        
        train_path, valid_path = load_and_prepare_mind_dataset(
            size="demo", 
            dest_path="./data/mind"
        )
        
        training_data, validation_data = preprocess_behaviors_mind(
            spark=spark,
            train_path=train_path,
            valid_path=valid_path,
            npratio=4  # Adjust this ratio based on class imbalance requirements
        )
        logger.info("MIND dataset preprocessed successfully.")
        
    elif data_source == "db":
        from data_management.data_utils import load_data_split
        config = kwargs.get("config")
        query = kwargs.get("query")
        training_data, validation_data = load_data_split(spark, config=config, query=query)
    
    elif data_source == "csv":
        file_path = kwargs.get("file_path", "./data/csv")
        logger.info(f"Loading training data from CSV: {file_path}/training_data.csv")
        training_data = spark.read.csv(f"{file_path}/training_data.csv", header=True)
        logger.info(f"Loading validation data from CSV: {file_path}/validation_data.csv")
        validation_data = spark.read.csv(f"{file_path}/validation_data.csv", header=True)
    
    else:
        raise ValueError(f"Unsupported data source: {data_source}")

    return training_data, validation_data


def train_als_model(training_data, validation_data, model_save_path):
    wandb.init(
        project="MIND-RS",
        name=f"als_rank_{ALS_CONFIG['rank']}_reg_{ALS_CONFIG['reg_param']}",
        config=ALS_CONFIG
    )

    logger.info("Starting ALS model training...")
    als = create_als_model()
    model = als.fit(training_data)
    logger.info("ALS model training completed.")

    logger.info("Generating predictions for validation data...")
    predictions = make_predictions(model, validation_data)

    logger.info("Evaluating the ALS model...")
    regression_metrics = compute_regression_metrics(predictions)
    rmse = regression_metrics["RMSE"]
    mae = regression_metrics["MAE"]
    logger.info(f"Validation RMSE: {rmse}")

    wandb.log({
        "RMSE": rmse,
        "rank": ALS_CONFIG["rank"],
        "maxIter": ALS_CONFIG["max_iter"],
        "regParam": ALS_CONFIG["reg_param"],
        "alpha": ALS_CONFIG["alpha"],
    })
    logger.info("Metrics logged to WandB.")

    logger.info("Saving the trained ALS model...")
    save_model(model, model_save_path)
    logger.info(f"ALS model saved successfully to {model_save_path}.")

    wandb.finish()
    logger.info("Training process completed and WandB session closed.")
