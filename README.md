## Recommendation System

To run the app:

```
docker-compose build
docker-compose up -d
chmod +x start.sh
./start.sh
```

Wait for the data to populate the db.

Wait for the script clean_embed to calculate and load embeddings in the db.

Then monitorate the activity of the containers (cbrs and als).

Monitor the activity of the containers:

* **ALS** : This component is fully operational and can be monitored.
* **CBRS** : Currently under development and not functional yet.

Mongo Local Host: http://localhost:8081
Spark Local Host: http://localhost:4040
