import wandb
from algorithms.als_utils import create_als_model, save_model, make_predictions
from configs.als_configs import ALS_CONFIG, DATASET_CONFIG, EVAL_CONFIG
from training.evaluation import evaluate_model
from data_management.data_utils import load_and_prepare_mind_dataset, preprocess_behaviors_mind
from utilities.logger import get_logger
from training.evaluation_metrics import compute_regression_metrics, compute_ranking_metrics
#from recommenders.datasets.mind import download_mind, extract_mind


logger = get_logger(name="ALS_Training", log_file="logs/train_als.log")

#Load training and validation data based on the selected data source.
def load_training_data(spark,
                       data_source = "recommenders",  # "db", "recommenders", or "csv"
                       **kwargs):
    if data_source == "recommenders":
        
        logger.info("Loading and preprocessing MIND dataset...")
        
        train_path, valid_path = load_and_prepare_mind_dataset(
            size=DATASET_CONFIG["size"], 
            dest_path=DATASET_CONFIG["data_path"]
        )
        
        training_data, validation_data = preprocess_behaviors_mind(
            spark=spark,
            train_path=train_path,
            valid_path=valid_path,
            npratio=DATASET_CONFIG["npratio"]  
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
    
    # COMMENTED FOR DEBUGGING, WANDB/LOGGER MIGHT BE THE CAUSE OF THE BROKEN TRAINING LOOP
    wandb.init(
        project="MIND-RS",
        name=f"als_rank_{ALS_CONFIG['rank']}_reg_{ALS_CONFIG['reg_param']}",
        config={**ALS_CONFIG, **DATASET_CONFIG, **EVAL_CONFIG}
    )
    
    logger.info("Starting ALS model training...")
    als = create_als_model()
    
    for iteration in range(ALS_CONFIG["max_iter"]):
        
        als.setMaxIter(iteration + 1)
        model = als.fit(training_data)

        predictions = make_predictions(model, validation_data)

        regression_metrics = compute_regression_metrics(predictions)
        rmse = regression_metrics["RMSE"]
        logger.info(f"Training ALS model - Iteration {iteration + 1}/{ALS_CONFIG['max_iter']}")
        

        wandb.log({
            "RMSE": rmse,
            "Iteration": iteration + 1
        })

    # Final predictions and metrics
    predictions = make_predictions(model, validation_data)
    regression_metrics = compute_regression_metrics(predictions)
    ranking_metrics = compute_ranking_metrics(predictions, top_k=EVAL_CONFIG["k"])
    rmse = regression_metrics["RMSE"]
    
    wandb.log({
        "Final RMSE": rmse
    })
    logger.info(f"Final RMSE: {rmse}")    

    logger.info("Saving the trained ALS model...")
    save_model(model, ALS_CONFIG["model_save_path"])
    
    logger.info(f"ALS model saved successfully to {model_save_path}.")

    wandb.finish()
    logger.info("Training process completed and WandB session closed.")
