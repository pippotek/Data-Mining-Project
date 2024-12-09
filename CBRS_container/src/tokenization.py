from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF
from pyspark.sql.functions import col, udf, concat_ws, split
from pyspark.sql.types import ArrayType, StringType

def tokenize_and_remove_stopwords(news_df):
    # Tokenize CleanTitle and CleanAbstract
    tokenizer_title = Tokenizer(inputCol="CleanTitle", outputCol="TitleTokens")
    tokenizer_abstract = Tokenizer(inputCol="CleanAbstract", outputCol="AbstractTokens")
    news_df = tokenizer_title.transform(news_df)
    news_df = tokenizer_abstract.transform(news_df)

    # Remove stopwords
    stopword_remover_title = StopWordsRemover(inputCol="TitleTokens", outputCol="FilteredTitleTokens")
    stopword_remover_abstract = StopWordsRemover(inputCol="AbstractTokens", outputCol="FilteredAbstractTokens")
    news_df = stopword_remover_title.transform(news_df)
    news_df = stopword_remover_abstract.transform(news_df)

    # Clean tokens (remove commas)
    def clean_tokens(tokens):
        if tokens:
            return [token.replace(",", "") for token in tokens]
        return tokens

    clean_tokens_udf = udf(clean_tokens, ArrayType(StringType()))
    news_df = news_df.withColumn("FilteredTitleTokens", clean_tokens_udf(col("FilteredTitleTokens")))
    news_df = news_df.withColumn("FilteredAbstractTokens", clean_tokens_udf(col("FilteredAbstractTokens")))

    # Combine tokens
    news_df = news_df.withColumn("CombinedTokens", concat_ws(" ", "FilteredTitleTokens", "FilteredAbstractTokens"))
    news_df = news_df.withColumn("CombinedWords", split(news_df["CombinedTokens"], " "))

    return news_df


