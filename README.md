# News Recommendation System Using MIND Dataset

This project implements and compares two personalized news recommendation systems based on the **Microsoft News Dataset (MIND)**. The two approaches used are:

1. **Collaborative Filtering (ALS)** - Implemented using the Alternating Least Squares algorithm in Apache Spark.
2. **Content-Based Filtering (FAISS)** - Utilizes BERT embeddings and FAISS for efficient vector similarity search.

The project explores key challenges such as scalability, implicit feedback handling, and embedding-based recommendations, and provides insights into the trade-offs between the two approaches.

## Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Content-Based Filtering](#Content-Based-Filtering)
- [Clustering](#Clustering)
- [Collaborative Filtering](#Collaborative-Filtering)
- [Setup Instructions](#setup-instructions)
- [Results](#results)
- [Acknowledgements](#acknowledgements)

## Introduction

The goal of this project is to compare the strengths and weaknesses of **Collaborative Filtering** and **Content-Based Filtering** in the context of personalized news recommendations. The collaborative approach models user-item interactions through matrix factorization, while the content-based approach leverages high-dimensional embeddings to recommend articles based on similarity.

The **Microsoft News Dataset (MIND)** provides the foundation for this analysis, containing:
- News articles categorized by topic and subtopic.
- User behavior logs including article clicks, impressions, and interaction histories.

### Key Highlights
- **ALS Model**: Handles large-scale, sparse user-item matrices with Spark's distributed architecture.
- **FAISS Integration**: Employs BERT embeddings and FAISS for scalable nearest-neighbor search, addressing the cold-start problem effectively.

## Project Structure

The repository is organized as follows:

```
.
├── Dockerfile
├── EDA.ipynb
├── LICENSE
├── MIND_Recommender_Results.pbix
├── README.md
├── docker-compose.yml
├── experiments
│   ├── cbrs_spark.py
│   └── newsapi
│       ├── cleaning.py
│       ├── embed.py
│       ├── prova_fetching.py
│       └── provamongo.py
├── requirements
│   ├── requirements_als.txt
│   ├── requirements_cbrs.txt
│   ├── requirements_clustering.txt
│   └── requirements_fetching.txt
├── requirements.txt
├── saved_models
├── src
│   ├── __init__.py
│   ├── algorithms
│   │   ├── als
│   │   │   ├── als_utils.py
│   │   │   ├── run_train_als.py
│   │   │   └── train_als.py
│   │   ├── cbrs
│   │   │   ├── __init__.py
│   │   │   ├── cbrs_utils_pandas.py
│   │   │   ├── clean_embed.py
│   │   │   ├── info.md
│   │   │   └── run_cbrs_pandas.py
│   │   └── clustering
│   │       └── clustering.py
│   ├── configs
│   │   ├── config.yaml
│   │   └── setup.py
│   ├── data_management
│   │   ├── __init__.py
│   │   ├── fetch_mind.py
│   │   └── mind.py
│   ├── training
│   │   ├── ALS_hyperparam_optimization.ipynb
│   │   ├── __init__.py
│   │   ├── evaluation.py
│   │   └── evaluation_metrics.py
│   └── utilities
│       ├── __init__.py
│       └── data_utils.py
└── start.sh
```


### Key Directories
- `src/`: Core codebase including algorithms, data management, and utilities.
- `experiments/`: Exploratory scripts for embedding generation and news API fetching.
- `outputs/`: Contains visualizations and analysis outputs (e.g., cluster visualizations).
- `requirements/`: Separate requirements files for different modules (ALS, FAISS, etc.).

## Content-Based Filtering

The **Content-Based Filtering (CBR)** model uses article embeddings to recommend news based on similarity to previously clicked articles. Key highlights:

- **Text Processing**: News articles are represented using BERT embeddings that capture their semantic meaning.
- **Vector Similarity Search**: FAISS (Facebook AI Similarity Search) is employed to compute approximate nearest neighbors efficiently in the embedding space.
- **Cold-Start Problem Handling**: Unlike collaborative filtering, CBR does not require user-item interaction history, making it well-suited for new users or articles.

Steps:
1. **Embedding Generation**: News articles are converted into vector embeddings using a pretrained BERT model.
2. **FAISS Indexing**: The embeddings are indexed with FAISS for efficient similarity search.
3. **Recommendation Generation**: Articles similar to a user’s reading history are retrieved based on cosine similarity.

Advantages:
- Handles cold-start scenarios effectively.
- Highly interpretable and adaptable for dynamic datasets like news.

Limitations:
- Requires high-quality embeddings to perform well.
- Content-based recommendations may lack diversity.

---

## Clustering

To validate the embeddings generated for content-based filtering, **K-means clustering** was performed on the news article embeddings. The goal was to ensure that similar articles were grouped together.

### Methodology
- **Number of Clusters (k=3)**: Based on the EDA, the two largest categories (news and sports) formed distinct clusters, while minor categories were grouped into the third cluster.
- **Visualization**:
  - PCA (Principal Component Analysis) was used to reduce the dimensionality of embeddings for visualization.
  - A heatmap was generated to show the distribution of categories within each cluster.

### Observations
- Clusters effectively captured the semantic structure of the dataset, with "news" and "sports" forming distinct groups.
- Minor categories were grouped into the third cluster, aligning with expectations from the EDA.
  
---

## Collaborative Filtering

The **Collaborative Filtering (ALS)** model uses matrix factorization to recommend news based on user-item interaction data. Key highlights:

- **ALS Algorithm**: The Alternating Least Squares (ALS) algorithm is implemented using Apache Spark, leveraging its distributed computing capabilities.
- **Latent Factors**: The model learns latent factors for both users and articles, capturing hidden relationships in the interaction matrix.
- **Interaction Matrix**: Implicit feedback (e.g., clicks) is used to construct a user-item matrix.

Steps:
1. **Data Preprocessing**: The user-item interaction matrix is constructed using implicit feedback from the MIND dataset.
2. **Model Training**: The ALS algorithm trains on the interaction matrix, minimizing reconstruction error.
3. **Recommendations**: The trained model predicts user preferences for articles not yet interacted with.

Advantages:
- Highly accurate for dense interaction matrices.
- Scales well to large datasets with Spark.

Limitations:
- Struggles with cold-start problems for new users or articles.
- Requires significant computational resources for matrix factorization.


---

## Setup Instructions

### Prerequisites
- Docker and Docker Compose
- Python 3.8+
- MongoDB

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/Data-Mining-Project.git
   cd Data-Mining-Project
   ```
2. **Start the App**:
   ```
   bash start.sh
   ```

