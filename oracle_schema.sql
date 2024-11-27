CREATE TABLE news (
    id NUMBER PRIMARY KEY,
    title VARCHAR2(500),
    summary VARCHAR2(2000),
    excerpt VARCHAR2(2000),
    embedding CLOB  -- Add this column if not already present
);

CREATE TABLE interactions (
    userid VARCHAR2(100) PRIMARY KEY,
    articleids CLOB
);