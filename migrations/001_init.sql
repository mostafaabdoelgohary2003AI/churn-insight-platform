CREATE TABLE IF NOT EXISTS prediction_log (
    id SERIAL PRIMARY KEY,
    ts BIGINT NOT NULL,
    subscription_length DOUBLE PRECISION,
    seconds_of_use DOUBLE PRECISION,
    complaints INTEGER,
    probability DOUBLE PRECISION,
    prediction INTEGER
);
