from sparknlp.base import DocumentAssembler
from sparknlp.annotator import Tokenizer as NLPTokenizer, BertEmbeddings, EmbeddingsFinisher
from pyspark.ml import Pipeline
from pyspark.ml.feature import CountVectorizer, IDF
from pyspark.sql.functions import col, concat_ws, expr

def compute_tfidf(news_df, vocab_size=10000, min_df=2):
    # Compute TF
    cv = CountVectorizer(inputCol="CombinedWords", outputCol="RawFeatures", vocabSize=vocab_size, minDF=min_df)
    cv_model = cv.fit(news_df)
    news_df = cv_model.transform(news_df)

    # Compute IDF
    idf = IDF(inputCol="RawFeatures", outputCol="TFIDFeatures")
    idf_model = idf.fit(news_df)
    news_df = idf_model.transform(news_df)

    return news_df

def add_text_column(news_df):
    # Create a text column by concatenating CombinedWords
    news_df = news_df.withColumn("text", concat_ws(" ", col("CombinedWords")))
    return news_df

def compute_bert_embeddings(news_df):
    documentAssembler = DocumentAssembler() \
        .setInputCol("text") \
        .setOutputCol("document")

    tokenizer = NLPTokenizer() \
        .setInputCols(["document"]) \
        .setOutputCol("token")

    bert = BertEmbeddings.pretrained("bert_base_uncased", "en") \
        .setInputCols(["document", "token"]) \
        .setOutputCol("embeddings")

    embeddingsFinisher = EmbeddingsFinisher() \
        .setInputCols(["embeddings"]) \
        .setOutputCols(["finished_embeddings"]) \
        .setOutputAsVector(True) \
        .setCleanAnnotations(False)

    pipeline = Pipeline(stages=[documentAssembler, tokenizer, bert, embeddingsFinisher])
    bert_model = pipeline.fit(news_df)
    news_df = bert_model.transform(news_df)

    # Average embeddings to get a sentence-level representation
    news_df = news_df.withColumn(
        "sentence_embedding",
        expr("transform(finished_embeddings, x -> aggregate(x, 0D, (acc, y) -> acc + y) / size(x))")
    )

    return news_df

