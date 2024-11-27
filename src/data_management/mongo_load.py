import cx_Oracle
import pandas as pd
from utilities import load_config

### FUNCTIONS ###

def batch_insert_oracle(connection, table_name, data, columns, batch_size=100):
    """Insert data into Oracle DB in batches."""
    cursor = connection.cursor()
    placeholders = ", ".join([":" + str(i + 1) for i in range(len(columns))])  # e.g., :1, :2, :3
    column_list = ", ".join(columns)
    insert_query = f"INSERT INTO {table_name} ({column_list}) VALUES ({placeholders})"

    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        try:
            cursor.executemany(insert_query, batch)
            connection.commit()
            print(f"Inserted {len(batch)} rows successfully in batch {i // batch_size + 1}.")
        except cx_Oracle.DatabaseError as e:
            print(f"Error inserting rows in batch {i // batch_size + 1}: {str(e)}")
            connection.rollback()

def load_tsv_to_dataframe(filepath, column_names):
    """Load a TSV file into a Pandas DataFrame."""
    return pd.read_csv(filepath, sep='\t', names=column_names, header=None)

### SCRIPT ###

# Load the configuration
config = load_config("src/config.yaml")

# Access specific values
if config:
    dsn = config.get('db_dsn', None)
    user = config.get('db_user', None)
    password = config.get('db_password', None)
    wallet_location = config.get('wallet_path', None)

    print(f"Connecting to Oracle DB using DSN: {dsn}")

# Establish the Oracle connection
connection = cx_Oracle.connect(
    user=user,
    password=password,
    dsn=dsn,
    config_dir=wallet_location
)

# Define file paths and column mappings
interactions_file = 'interactions.tsv'
news_file = 'news.tsv'

# Define Oracle tables and column mappings
interactions_table = 'INTERACTIONS'  # Replace with your Oracle table name
news_table = 'NEWS'  # Replace with your Oracle table name

### Load and Process Interactions Data ###
interactions_columns = ['session_id', 'userid', 'timestamp', 'articleids', 'additional_data']
interactions_df = load_tsv_to_dataframe(interactions_file, interactions_columns)

# Group by userid and aggregate articleids
interactions_grouped = interactions_df.groupby('userid').agg({
    'articleids': lambda x: list(set(" ".join(x.dropna()).split()))  # Aggregate all article IDs for each user
}).reset_index()

# Prepare grouped interactions data for Oracle
interactions_data_filtered = [
    (row["userid"], ",".join(row["articleids"]))  # Convert articleids list to a comma-separated string
    for _, row in interactions_grouped.iterrows()
]

# Insert grouped interactions data
print("Inserting grouped interactions data...")
batch_insert_oracle(connection, interactions_table, interactions_data_filtered, columns=['USERID', 'ARTICLEIDS'], batch_size=100)

### Load and Process News Data ###
news_columns = ['articleid', 'topic', 'detailed_topic', 'title', 'description', 'content', 'link', 'entities', 'additional_data']
news_df = load_tsv_to_dataframe(news_file, news_columns)

# Prepare simplified news data for Oracle
news_data_filtered = [
    (row["articleid"], row["title"], row["content"])
    for _, row in news_df.iterrows()
]

# Insert news data
print("Inserting simplified news data...")
batch_insert_oracle(connection, news_table, news_data_filtered, columns=['ARTICLEID', 'TITLE', 'CONTENT'], batch_size=100)

# Close the connection
connection.close()
print("Data import completed successfully!")
