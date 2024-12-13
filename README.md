# News Recommendation System

## Project Overview
This project involves building a **news recommendation system** using the **MIND dataset** provided by Microsoft. The system employs three recommendation algorithms: 

1. **Collaborative Filtering**: Suggests news based on user similarity and interaction patterns.
2. **Content-Based Filtering**: Recommends news based on the similarity of content features, leveraging BERT embeddings for semantic understanding.
3. **Hybrid Recommendation System**: Combines collaborative and content-based filtering to enhance recommendation accuracy.

The goal is to explore the effectiveness of different recommendation strategies and evaluate their performance on the MIND dataset.

---

## Dataset Description

### **MIND Dataset**

The MIND dataset consists of two key files: `behaviors.tsv` and `news.tsv`.

#### **1. behaviors.tsv**

Contains impression logs and user click histories. This file has the following columns:

| Column          | Description                                                                                  |
|------------------|----------------------------------------------------------------------------------------------|
| **Impression ID** | Unique identifier for an impression.                                                        |
| **User ID**      | Anonymized identifier for a user.                                                            |
| **Time**         | Timestamp of the impression in the format `MM/DD/YYYY HH:MM:SS AM/PM`.                       |
| **History**      | List of news IDs clicked by the user before this impression (space-separated).               |
| **Impressions**  | List of news displayed in the impression, followed by user click behaviors (1 for clicked, 0 for not clicked). |

**Example:**
```
Impression ID: 123
User ID: U131
Time: 11/13/2019 8:36:57 AM
History: N11 N21 N103
Impressions: N4-1 N34-1 N156-0 N207-0 N198-0
```

#### **2. news.tsv**

Contains metadata about the news articles referenced in `behaviors.tsv`. This file has the following columns:

| Column            | Description                                                                                       |
|--------------------|---------------------------------------------------------------------------------------------------|
| **News ID**        | Unique identifier for a news article.                                                            |
| **Category**       | High-level category of the news (e.g., sports, politics).                                         |
| **Subcategory**    | More specific category of the news (e.g., golf, elections).                                       |
| **Title**          | Headline of the news article.                                                                    |
| **Abstract**       | Brief summary of the news article.                                                               |
| **URL**            | Link to the full news article (some URLs may be expired).                                         |
| **Title Entities** | Entities in the news title linked to the Wikidata knowledge graph.                                |
| **Abstract Entities** | Entities in the news abstract linked to the Wikidata knowledge graph.                          |

**Example:**
```
News ID: N37378
Category: sports
Subcategory: golf
Title: PGA Tour winners
Abstract: A gallery of recent winners on the PGA Tour.
URL: https://www.msn.com/en-us/sports/golf/pga-tour-winners/ss-AAjnQjj?ocid=chopendata
Title Entities: [{“Label”: “PGA Tour”, “Type”: “O”, “WikidataId”: “Q910409”, “Confidence”: 1.0, “OccurrenceOffsets”: [0], “SurfaceForms”: [“PGA Tour”]}]
Abstract Entities: [{“Label”: “PGA Tour”, “Type”: “O”, “WikidataId”: “Q910409”, “Confidence”: 1.0, “OccurrenceOffsets”: [35], “SurfaceForms”: [“PGA Tour”]}]
```

---

## Methodology

### **1. Data Preprocessing**

- **User Behavior Data**: Extract user click histories and impression interactions from `behaviors.tsv`.
- **News Content Data**: Extract and preprocess features such as title, abstract, category, and entities from `news.tsv`.

### **2. Embedding Generation**

- Utilize **BERT embeddings** for semantic representation of:
  - News titles
  - News abstracts
  - Entity information (if available)

### **3. Model Building**

- **Collaborative Filtering**:
  - Based on user-item interaction matrix from `behaviors.tsv`.
  - Matrix factorization or neighborhood-based approaches.

- **Content-Based Filtering**:
  - Compute similarity between news articles using BERT embeddings.
  - Recommend articles with the highest similarity to user preferences.

- **Hybrid Model**:
  - Combine collaborative and content-based predictions.
  - Experiment with weighted averages or machine learning-based ensemble methods.

### **4. Evaluation**

---

## Data Storage
 **MongoDB database**.

---

