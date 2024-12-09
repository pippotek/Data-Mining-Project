from src.utils import create_spark_session
from src.data_loading import load_news_data, load_behaviors_data
from src.preprocessing import (
    preprocess_news_data,
    preprocess_behaviors_data,
    explode_impressions,
    join_with_news_features,
    filter_clicked_news
)
from src.tokenization import tokenize_and_remove_stopwords
from src.embeddings import compute_tfidf, add_text_column, compute_bert_embeddings
# If you need similarity or user profiles, import from similarity.py as well:
# from src.similarity import build_category_user_profiles

def main():
    spark = create_spark_session("Content Based Recommender System")

    # Load and preprocess news data
    news_path = "data/mind/MINDsmall_train/news.tsv"
    news_df = load_news_data(spark, news_path)
    news_df = news_df.toDF("NewsID", "Category", "Subcategory", "Title", "Abstract", "URL", "TitleEntities", "AbstractEntities")
    news_df = preprocess_news_data(news_df)
    news_df = tokenize_and_remove_stopwords(news_df)
    news_df = compute_tfidf(news_df)
    news_df = add_text_column(news_df)
    news_df = compute_bert_embeddings(news_df)

    # Example: Show processed embeddings
    news_df.select("NewsID", "sentence_embedding").show(5, truncate=False)

    # Load and preprocess behaviors data
    behaviors_path = "data/mind/MINDsmall_train/behaviors.tsv"
    behaviors_df = load_behaviors_data(spark, behaviors_path)
    behaviors_df = preprocess_behaviors_data(behaviors_df)
    impressions_exploded = explode_impressions(behaviors_df)

    # Join impressions with news features (assuming news_df has TFIDFeatures and Category)
    # Extract a subset of news_df as news_features_df if needed:
    news_features_df = news_df.select("NewsID", "Category", "TFIDFeatures")

    impressions_with_features = join_with_news_features(impressions_exploded, news_features_df)
    clicked_news_df = filter_clicked_news(impressions_with_features)

    # If you want to build user profiles:
    # user_profiles = build_category_user_profiles(clicked_news_df, news_features_df)
    # user_profiles.show(5, truncate=False)

if __name__ == "__main__":
    main()
