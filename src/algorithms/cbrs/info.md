## Content-Based Recommendation System (CBRS)

### EMBEDDINGS

The** **`clean_embed.py` script is responsible for preprocessing and embedding news articles. Here’s a step-by-step breakdown of its workflow:

#### **1. Data Loading**

* Reads data from MongoDB:
  * **`news_train`** : Training news data.
  * **`news_valid`** : Validation news data.
  * **`news_combined_embeddings`** : Collection of already processed embeddings.
* Combines training and validation data into a single DataFrame.

#### **2. Preprocessing Data**

* **Removing Duplicates** : Ensures no duplicate** **`news_id` values exist.
* **Filtering Processed Records** : Excludes articles already present in the** **`news_combined_embeddings` collection using a** **`left_anti` join.
* **Cleaning Text** :
* Combines** **`title` and** **`abstract` fields into a** **`combined_text` column.
* Removes non-alphanumeric characters and converts text to lowercase.

#### **3. Building NLP Pipeline**

* Constructs a Spark NLP pipeline with the following stages:
  * **DocumentAssembler** : Converts text into a format suitable for NLP tasks.
  * **Tokenizer** : Splits text into tokens (words).
  * **StopWordsCleaner** : Removes common stop words (e.g., "the", "and").
  * **BERT Embeddings** : Generates word embeddings using a pretrained BERT model.
  * **Sentence Embeddings** : Averages word embeddings to form sentence-level representations.
  * **Embeddings Finisher** : Converts embeddings into a format suitable for output.

#### **4. Batch Processing**

* Processes the data in batches to handle large-scale datasets efficiently:
  * Applies the NLP pipeline to generate embeddings for each batch.
  * Converts embeddings into a string format for storage.
  * Writes the processed embeddings back to MongoDB (`news_combined_embeddings` collection).

---

### CBRS PIPELINE

The** **`run_cbrs.py` script implements the main pipeline for the Content-Based Recommendation System. Here’s how it works:

#### **1. Data Loading**

* Loads:
  * **News Embeddings** : Preprocessed embeddings from** **`news_combined_embeddings`.
  * **Behavioral Data** :
  * **`behaviors_train`** : Training user click behaviors.
  * **`behaviors_valid`** : Validation user click behaviors.

#### **2. Preprocessing**

* **News Embeddings** : Converts the string representation of embeddings back into numerical arrays.
* **User Behavior** : Explodes the** **`history` of user interactions into individual news items for easier processing.

#### **3. User Profile Creation**

* Uses historical user behavior to create profiles:

  * Joins user interaction data with news embeddings.
  * Averages the embeddings of all news articles a user interacted with to create a** ****user embedding** (personalized representation of a user).
  * 

  #### **4. Recommendations**


  * Matches user profiles with news embeddings to generate recommendations:
    * Joins user profiles with news articles.
    * Computes similarity scores using** ** **Cosine Similarity** :
      * Measures the similarity between user embeddings and news embeddings.
    * Ranks news articles for each user based on similarity scores.

  #### **5. Storing Recommendations (TO DO)**

  * Writes top-`k` recommendations for each user to MongoDB in the** **`cbrs_recommendations` collection.

  #### **6. Evaluation (TO DO)**

  * Compares recommendations against user click behavior in** **`behaviors_valid` using metrics like** ** **Precision@k** .

### **Example: Preprocessing and User Profile Creation**

#### **Sample Data** :

| **Column**        | **Content**               |
| ----------------------- | ------------------------------- |
| **Impression ID** | 123                             |
| **User ID**       | U131                            |
| **Time**          | 11/13/2019 8:36:57 AM           |
| **History**       | N11 N21 N103                    |
| **Impressions**   | N4-1 N34-1 N156-0 N207-0 N198-0 |

---

#### **Step 1: Explode the** **`History` Column**

The** **`History` column is exploded to create one row per interaction:

| **User ID** | **History Item** |
| ----------------- | ---------------------- |
| U131              | N11                    |
| U131              | N21                    |
| U131              | N103                   |

---

#### **Step 2: Join with News Embeddings**

Join the exploded** **`History Item` with the** **`news_combined_embeddings` collection to retrieve embeddings:

| **User ID** | **News ID** | **Embedding** |
| ----------------- | ----------------- | ------------------- |
| U131              | N11               | [0.1, 0.2, 0.3]     |
| U131              | N21               | [0.2, 0.3, 0.4]     |
| U131              | N103              | [0.3, 0.4, 0.5]     |

---

#### **Step 3: Compute the Average Embedding**

Average the embeddings for all** **`History Items` associated with the user to create a** ** **user embedding** :

| **User ID** | **User Embedding** |
| ----------------- | ------------------------ |
| U131              | [0.2, 0.3, 0.4]          |

 **Formula for Averaging** :

User Embedding=[0.1,0.2,0.3]+[0.2,0.3,0.4]+[0.3,0.4,0.5]3=[0.2,0.3,0.4]**User Embedding**=**3**[**0.1**,**0.2**,**0.3**]**+**[**0.2**,**0.3**,**0.4**]**+**[**0.3**,**0.4**,**0.5**]=**[**0.2**,**0.3**,**0.4**]**This embedding serves as the** ****personalized representation** of the user's preferences.
