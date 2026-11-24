"""
Runs the SQL files against a local DuckDB database and prints each result.
No database server needed:  python src/run_sql.py
(The same SQL also runs in PostgreSQL; see sql/01_create_tables.sql.)
"""
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "telco_customer_churn.csv"
SQL = ROOT / "sql"


def split_statements(sql_text):
    """Split a .sql file into statements, keeping the comment just above each one."""
    return [s.strip() for s in sql_text.split(";") if s.strip() and not all(
        line.strip().startswith("--") or not line.strip() for line in s.splitlines())]


def main():
    con = duckdb.connect(str(ROOT / "churn.duckdb"))

    for stmt in split_statements((SQL / "01_create_tables.sql").read_text()):
        con.execute(stmt)
    con.execute(f"INSERT INTO customers_raw SELECT * FROM read_csv('{DATA.as_posix()}', header=true, all_varchar=true)")

    for stmt in split_statements((SQL / "02_exploratory_queries.sql").read_text()):
        title = next((l.strip("- ").strip() for l in stmt.splitlines() if l.startswith("-- Q")), "Query")
        print(f"\n=== {title}")
        print(con.sql(stmt))


if __name__ == "__main__":
    main()
