import sqlite3
import tempfile
import os

def test_idempotency():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "test.sqlite")
    
    # First pass
    c = sqlite3.connect(p)
    c.execute('CREATE TABLE postings (country TEXT, block_key TEXT, entity_id TEXT, PRIMARY KEY (country, block_key, entity_id)) WITHOUT ROWID')
    c.execute('CREATE TABLE eligible_keys (country TEXT, block_key TEXT, PRIMARY KEY (country, block_key)) WITHOUT ROWID')
    c.execute('CREATE TABLE cache_metadata (cache_key TEXT, built_at TEXT, complete INTEGER, source_rows INTEGER)')
    c.execute("INSERT INTO cache_metadata VALUES ('key1', '', 1, 1)")
    c.execute('INSERT OR IGNORE INTO eligible_keys SELECT country, block_key FROM postings GROUP BY country, block_key HAVING COUNT(*) <= 1')
    c.commit()
    c.close()
    print("first ok")
    
    # Second pass - re-finalization / idempotence
    c2 = sqlite3.connect(p)
    c2.execute('INSERT OR IGNORE INTO eligible_keys SELECT country, block_key FROM postings GROUP BY country, block_key HAVING COUNT(*) <= 1')
    c2.execute("UPDATE cache_metadata SET built_at = '2023-01-01T00:00:00.000Z', complete = 1, source_rows = 1 WHERE cache_key = ?")
    c2.execute("UPDATE cache_metadata SET built_at = '2023-01-01T00:00:00.000Z', complete = 1, source_rows = 1 WHERE cache_key = ?", ("key1",))
    c2.commit()
    c2.close()
    print("second ok")

if __name__ == "__main__":
    test_idempotency()
