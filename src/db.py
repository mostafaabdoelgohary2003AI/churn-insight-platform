"""Prediction logging store. PostgreSQL via DATABASE_URL in compose; SQLite locally."""
import os, sqlite3, time

SQLITE_PATH = os.environ.get("APP_DB", "data/app.db")

def _pg():
    import psycopg2
    return psycopg2.connect(os.environ["DATABASE_URL"])

def get_conn():
    if os.environ.get("DATABASE_URL"):
        return _pg()
    return sqlite3.connect(SQLITE_PATH)

def run_migrations(path=None):
    if path is None:
        path = "migrations/001_init.sql" if os.environ.get("DATABASE_URL") else "migrations/001_init.sqlite.sql"
    with open(path) as f:
        sql = f.read()
    conn = get_conn()
    cur = conn.cursor()
    for stmt in [s.strip() for s in sql.split(";") if s.strip()]:
        cur.execute(stmt)
    conn.commit(); conn.close()

def log_prediction(features: dict, probability: float, prediction: int):
    conn = get_conn()
    cur = conn.cursor()
    q = ("INSERT INTO prediction_log (ts, subscription_length, seconds_of_use, complaints, probability, prediction) "
         "VALUES (%s, %s, %s, %s, %s, %s)" if os.environ.get("DATABASE_URL") else
         "INSERT INTO prediction_log (ts, subscription_length, seconds_of_use, complaints, probability, prediction) "
         "VALUES (?, ?, ?, ?, ?, ?)")
    cur.execute(q, (int(time.time()), float(features.get("subscription_length", 0)),
                    float(features.get("seconds_of_use", 0)), int(features.get("complaints", 0)),
                    probability, prediction))
    conn.commit(); conn.close()

def history(limit=50):
    conn = get_conn()
    cur = conn.cursor()
    q = ("SELECT ts, subscription_length, seconds_of_use, complaints, probability, prediction "
         "FROM prediction_log ORDER BY id DESC LIMIT %s" if os.environ.get("DATABASE_URL") else
         "SELECT ts, subscription_length, seconds_of_use, complaints, probability, prediction "
         "FROM prediction_log ORDER BY id DESC LIMIT ?")
    cur.execute(q, (limit,))
    rows = cur.fetchall(); conn.close()
    return [{"ts": r[0], "subscription_length": r[1], "seconds_of_use": r[2], "complaints": r[3],
             "probability": r[4], "prediction": r[5]} for r in rows]
