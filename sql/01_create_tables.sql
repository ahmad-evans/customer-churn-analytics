-- 01_create_tables.sql
-- Builds the raw table and a cleaned view for the Telco churn data.
-- Works in PostgreSQL and DuckDB.

DROP VIEW IF EXISTS customers_clean;
DROP TABLE IF EXISTS customers_raw;

-- Raw table: one column per CSV column, loaded as-is.
-- total_charges is TEXT because 11 rows are blank in the source file.
CREATE TABLE customers_raw (
    customer_id        TEXT PRIMARY KEY,
    gender             TEXT,
    senior_citizen     INTEGER,
    partner            TEXT,
    dependents         TEXT,
    tenure             INTEGER,
    phone_service      TEXT,
    multiple_lines     TEXT,
    internet_service   TEXT,
    online_security    TEXT,
    online_backup      TEXT,
    device_protection  TEXT,
    tech_support       TEXT,
    streaming_tv       TEXT,
    streaming_movies   TEXT,
    contract           TEXT,
    paperless_billing  TEXT,
    payment_method     TEXT,
    monthly_charges    NUMERIC(8,2),
    total_charges      TEXT,
    churn              TEXT
);

-- PostgreSQL load (run from the repo root in psql):
-- \copy customers_raw FROM 'data/telco_customer_churn.csv' WITH (FORMAT csv, HEADER true);
-- DuckDB load is handled by src/run_sql.py.

-- Cleaned view:
--   * blank total_charges -> 0 (these are brand-new customers, tenure = 0)
--   * churn Yes/No -> 1/0
--   * count of add-on services each customer has
CREATE VIEW customers_clean AS
SELECT
    customer_id,
    gender,
    senior_citizen,
    partner,
    dependents,
    tenure,
    phone_service,
    multiple_lines,
    internet_service,
    online_security,
    online_backup,
    device_protection,
    tech_support,
    streaming_tv,
    streaming_movies,
    contract,
    paperless_billing,
    payment_method,
    monthly_charges,
    COALESCE(CAST(NULLIF(TRIM(total_charges), '') AS NUMERIC(10,2)), 0) AS total_charges,
    CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END AS churned,
      (CASE WHEN online_security   = 'Yes' THEN 1 ELSE 0 END)
    + (CASE WHEN online_backup     = 'Yes' THEN 1 ELSE 0 END)
    + (CASE WHEN device_protection = 'Yes' THEN 1 ELSE 0 END)
    + (CASE WHEN tech_support      = 'Yes' THEN 1 ELSE 0 END)
    + (CASE WHEN streaming_tv      = 'Yes' THEN 1 ELSE 0 END)
    + (CASE WHEN streaming_movies  = 'Yes' THEN 1 ELSE 0 END) AS num_addons
FROM customers_raw;
