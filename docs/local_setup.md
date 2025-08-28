# Running with local databases

The project normally connects to Google Cloud services.  For offline
development you can switch to lightweight local connectors that use
standard database drivers.

## Configure

Edit `config.ini` and set the connector backend to `local` and provide
connection information for the local databases:

```ini
[CONFIG]
connector_backend = local

[LOCAL]
# PostgreSQL connection string
pg_conn = postgresql+psycopg2://user:pass@localhost:5432/opendataqna
# SQLite database used for the BigQuery and Firestore connectors
sqlite_db = opendataqna.db
```

The SQLite file will be created automatically if it does not exist.

## Launch

After the configuration is updated, launch the application as usual:

```bash
python app.py
```

The application will now use the local PostgreSQL instance and the
SQLite file instead of Google Cloud services.

