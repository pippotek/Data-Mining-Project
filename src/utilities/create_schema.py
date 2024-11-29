import oracledb

def create_schema():
    # Database connection details
    db_dsn = "(description= (retry_count=20)(retry_delay=3)(address=(protocol=tcps)(port=1522)(host=adb.eu-milan-1.oraclecloud.com))(connect_data=(service_name=g126d38945ee7b1_w1d1rk5b8lafcoir_medium.adb.oraclecloud.com))(security=(ssl_server_dn_match=yes)))"  # TNS entry or DSN (e.g., "example_high")
    db_user = "ADMIN"        # Oracle DB user
    db_password = "sLg95NBJc-*:v,A"    # Oracle DB password

    # SQL statements to drop tables if they exist
    drop_interactions_table = "DROP TABLE interactions_val CASCADE CONSTRAINTS"
    drop_news_table = "DROP TABLE news_val CASCADE CONSTRAINTS"

    # SQL statements to create tables
    create_news_table = """
    CREATE TABLE news_test (
        articleid VARCHAR2(100) PRIMARY KEY,       -- Unique article ID
        topic VARCHAR2(100) NOT NULL,       -- Topic of the article
        topic_det VARCHAR2(100),            -- Detailed topic or subtopic
        title CLOB NOT NULL,       -- Title of the article
        content CLOB,                       -- Article content (large text)
        asset CLOB,
        title_entities CLOB,                -- JSON-structured entities extracted from the title
        content_entities CLOB,               -- JSON-structured entities extracted from the content
        embedding CLOB                    -- Article embedding (stored as text or binary)
    )
    """

    create_interactions_table = """
    CREATE TABLE interactions_test (
        interactionid NUMBER PRIMARY KEY,   -- Unique interaction ID
        userid VARCHAR2(100) NOT NULL,             -- User ID
        timestamp TIMESTAMP NOT NULL,       -- Interaction timestamp
        prev_viewed CLOB,                   -- Series of previously viewed article IDs
        displayed CLOB                      -- Series of displayed article IDs with suffixes (-1 or -0)
    )
    """

    try:
        # Connect to the Oracle database
        with oracledb.connect(user=db_user, password=db_password, dsn=db_dsn) as connection:
            with connection.cursor() as cursor:
                # Drop interactions table if it exists
                # try:
                #     print("Dropping interactions table if it exists...")
                #     cursor.execute(drop_interactions_table)
                #     print("Interactions table dropped successfully.")
                # except oracledb.DatabaseError as e:
                #     print("Interactions table does not exist or could not be dropped.")

                # # Drop news table if it exists
                # try:
                #     print("Dropping news table if it exists...")
                #     cursor.execute(drop_news_table)
                #     print("News table dropped successfully.")
                # except oracledb.DatabaseError as e:
                #     print("News table does not exist or could not be dropped.")

                # Create interactions table
                print("Creating interactions table...")
                cursor.execute(create_interactions_table)
                print("Interactions table created successfully.")

                # Create news table
                print("Creating news table...")
                cursor.execute(create_news_table)
                print("News table created successfully.")

            # Commit changes
            connection.commit()

    except oracledb.DatabaseError as e:
        # Handle database errors
        error, = e.args
        print("Error while recreating schema:", error.message)

if __name__ == "__main__":
    create_schema()