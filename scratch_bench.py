import sqlite3
import time
from pathlib import Path

from src.blocking.union import (
    _add_batch_keys,
    _attach_strategy_indexes,
    _create_database,
    _strategy_instances,
)
from src.data_loader import load_source

source1 = load_source("train", "source1").head(1000)
blockers = _strategy_instances(["name", "phonetic", "tokenset"])
conn = sqlite3.connect(":memory:")
_create_database(conn)
cache_paths = {
    s: next(Path("output/cache/blocking").glob(f"*source2_{s}_*.sqlite"))
    for s in ("name", "phonetic", "tokenset")
}
_attach_strategy_indexes(conn, cache_paths)
_add_batch_keys(conn, source1, blockers)

for strat in ("name", "phonetic", "tokenset"):
    sql = f"""WITH matched AS (
     SELECT DISTINCT k.source1_id, p.entity_id FROM batch_keys k
     JOIN idx_{strat}.eligible_keys e ON e.country = k.country AND e.block_key = k.block_key
     JOIN idx_{strat}.postings p ON p.country = k.country AND p.block_key = k.block_key
     WHERE k.strategy = ?
    ), ranked AS (
     SELECT source1_id, entity_id, ROW_NUMBER() OVER (PARTITION BY source1_id ORDER BY entity_id) AS rank FROM matched
    ) INSERT OR IGNORE INTO candidates SELECT source1_id, entity_id FROM ranked WHERE rank <= 500"""
    t0 = time.time()
    conn.execute(sql, (strat,))
    cnt = conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
    print(f"Time for 1000 rows {strat}: {time.time() - t0:.2f}s, candidates={cnt}")
