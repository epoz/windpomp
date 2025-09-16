import duckdb
import os


def import_from_parquet(pq_path: str, dest_db: str):
    if not os.path.isfile(dest_db):
        print(f"Creating new DuckDB database at {dest_db}")
    if not os.path.isfile(pq_path):
        raise FileNotFoundError(f"Parquet file {pq_path} not found")

    con = duckdb.connect(dest_db)

    con.execute(
        f"""create table if not exists clip_features (path TEXT PRIMARY KEY, features FLOAT[512] )"""
    )

    con.execute(
        f"insert or ignore into clip_features select path, json_extract(features, '$')::FLOAT[] as F from read_parquet('{pq_path}')"
    )
    con.execute("install vss")
    con.execute("load vss")
    con.execute("SET hnsw_enable_experimental_persistence = true;")
    con.execute("CREATE INDEX clip_idx ON clip_features USING HNSW (features);")
    con.close()
