import wandb
from src.algorithms.als.als_utils import create_als_model, save_model, make_predictions
from src.algorithms.als.als_configs import ALS_CONFIG
from src.training.evaluation_metrics import compute_regression_metrics, compute_ranking_metrics
from src.utilities.logger import get_logger
from src.configs.setup import load_config
from src.algorithms.als.als_configs import ALS_CONFIG, EVAL_CONFIG
#from recommenders.datasets.mind import download_mind, extract_mind


logger = get_logger(name="ALS_Training", log_file="logs/train_als.log")


def train_als_model(training_data, validation_data, model_save_path):
    config = load_config('src/configs/config.yaml')
    wandb.login(key=config.get('wandb_key'))
    wandb.init(
        project="MIND-RS",
        name=f"als_rank_{ALS_CONFIG['rank']}_reg_{ALS_CONFIG['reg_param']}",
        config=ALS_CONFIG,

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
