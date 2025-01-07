# News Recommendation System Using MIND Dataset

This project implements and compares two personalized news recommendation systems based on the **Microsoft News Dataset (MIND)**. The two approaches used are:

1. **Collaborative Filtering (ALS)** - Implemented using the Alternating Least Squares algorithm in Apache Spark.
2. **Content-Based Filtering (FAISS)** - Utilizes BERT embeddings and FAISS for efficient vector similarity search.

The project explores key challenges such as scalability, implicit feedback handling, and embedding-based recommendations, and provides insights into the trade-offs between the two approaches.

## Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Features](#features)
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
├── outputs
│   ├── cluster_category_distribution.csv
│   ├── cluster_category_distribution.png
│   ├── cluster_category_distribution_heatmap.png
│   ├── cluster_category_pivot_table.csv
│   ├── clusters_visualization.png
│   ├── clusters_visualization_3.png
│   └── clusters_visualization_4_clusters.png
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

## Features

- **Exploratory Data Analysis (EDA)**: Analyzes the distribution of news categories, subcategories, and user behavior.
- **Collaborative Filtering (ALS)**:
  - Trains a matrix factorization model using implicit feedback.
  - Optimized with hyperparameter tuning for RMSE performance.
- **Content-Based Filtering (FAISS)**:
  - Uses BERT embeddings to represent news articles in high-dimensional space.
  - Efficient similarity search with FAISS for personalized recommendations.
- **Clustering**:
  - Validates the semantic structure of embeddings using PCA and K-means.
  - Visualizes category distribution across clusters.

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
2. ** Install Dependencies:
   ```bash
   pip install -r requirements.txt
    ```
3. Start the App:
   ``bash
   bash start.sh
   ```

