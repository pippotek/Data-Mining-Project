from pyspark.sql.functions import udf, col, split, explode, regexp_extract
from pyspark.sql.types import StringType, IntegerType

def clean_text(text):
    """
    Clean the text by converting to lowercase and removing newline and tab characters.

    Parameters
    ----------
    text : str or None
        The input text.

    Returns
    -------
    str or None
        The cleaned text.
    """
    if text:
        return text.lower().replace("\n", " ").replace("\t", " ")
    return None

def preprocess_news_data(news_df):
    """
    Preprocess the news DataFrame by dropping rows without Title or Abstract and cleaning the text.

    Parameters
    ----------
    news_df : DataFrame
        The news DataFrame.

    Returns
    -------
    DataFrame
        The preprocessed news DataFrame.
    """
    # Drop rows where Title or Abstract are missing
    news_df = news_df.na.drop(subset=["Title", "Abstract"])

    # Register the UDF for cleaning text
    clean_text_udf = udf(clean_text, StringType())

    # Apply text cleaning to Title and Abstract
    news_df = news_df.withColumn("CleanTitle", clean_text_udf(col("Title")))
    news_df = news_df.withColumn("CleanAbstract", clean_text_udf(col("Abstract")))

    return news_df

def preprocess_behaviors_data(behaviors_df):
    """
    Preprocess the behaviors DataFrame by splitting History and Impressions into arrays.

    Parameters
    ----------
    behaviors_df : DataFrame
        The behaviors DataFrame.

    Returns
    -------
    DataFrame
        The preprocessed behaviors DataFrame.
    """
    behaviors_df = behaviors_df.withColumn("HistoryList", split(col("History"), " ")).drop("History")
    behaviors_df = behaviors_df.withColumn("ImpressionsList", split(col("Impressions"), " ")).drop("Impressions")

    return behaviors_df

def explode_impressions(behaviors_df):
    """
    Explode the ImpressionsList into individual impressions and extract CandidateNewsID and ClickLabel.

    Parameters
    ----------
    behaviors_df : DataFrame
        The preprocessed behaviors DataFrame.

    Returns
    -------
    DataFrame
        A DataFrame with exploded impressions and extracted CandidateNewsID and ClickLabel.
    """
    impressions_exploded = behaviors_df.select(
        "ImpressionID",
        "UserID",
        "Time",
        "HistoryList",
        explode("ImpressionsList").alias("ImpressionItem")
    )

    impressions_exploded = impressions_exploded \
        .withColumn("CandidateNewsID", regexp_extract(col("ImpressionItem"), r"^(N\d+)-\d+$", 1)) \
        .withColumn("ClickLabel", regexp_extract(col("ImpressionItem"), r"^N\d+-(\d+)$", 1).cast(IntegerType())) \
        .drop("ImpressionItem")

    return impressions_exploded

def join_with_news_features(impressions_exploded, news_features_df):
    """
    Join the exploded impressions DataFrame with the news features DataFrame.

    Parameters
    ----------
    impressions_exploded : DataFrame
        The DataFrame with impressions data.
    news_features_df : DataFrame
        The DataFrame with news features (e.g., TF-IDF features).

    Returns
    -------
    DataFrame
        A DataFrame enriched with news features.
    """
    impressions_with_features = impressions_exploded.join(
        news_features_df,
        impressions_exploded.CandidateNewsID == news_features_df.NewsID,
        how="left"
    ).drop(news_features_df.NewsID)  # Drop duplicate NewsID column if present

    return impressions_with_features

def filter_clicked_news(impressions_with_features):
    """
    Filter the DataFrame to only include clicked news articles (ClickLabel == 1).

    Parameters
    ----------
    impressions_with_features : DataFrame
        The DataFrame with impressions and news features.

    Returns
    -------
    DataFrame
        A DataFrame containing only clicked news.
    """
    clicked_news_df = impressions_with_features.filter(col("ClickLabel") == 1)
    return clicked_news_df
