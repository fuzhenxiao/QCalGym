import fcntl
import hashlib
import json
from pathlib import Path
import sqlite3
import numpy as np

def encode(value):
    if isinstance(value, bytes):
        return {'bytes': value.hex()}
    if isinstance(value, complex):
        return {'complex': [float(value.real), float(value.imag)]}
    if isinstance(value, np.generic):
        return encode(value.item())
    if isinstance(value, (tuple, list)):
        return [encode(v) for v in value]
    return value

def get_or_compute(path, key, compute, *, per_key_lock=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps([1, encode(key)], separators=(',', ':'), allow_nan=False)
    if per_key_lock:
        directory = path.with_suffix('.locks')
        directory.mkdir(exist_ok=True)
        lock_path = directory / (hashlib.sha256(text.encode()).hexdigest() + '.lock')
        with lock_path.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            with sqlite3.connect(path, timeout=60) as db:
                db.execute('CREATE TABLE IF NOT EXISTS channels (inputs TEXT PRIMARY KEY, result TEXT NOT NULL)')
                row = db.execute('SELECT result FROM channels WHERE inputs=?', (text,)).fetchone()
            if row is not None:
                return (json.loads(row[0]), True)
            value = compute()
            with sqlite3.connect(path, timeout=60) as db:
                db.execute('INSERT OR IGNORE INTO channels VALUES (?,?)', (text, json.dumps(value, allow_nan=False)))
            return (value, False)
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with sqlite3.connect(path) as db:
            db.execute('CREATE TABLE IF NOT EXISTS channels (inputs TEXT PRIMARY KEY, result TEXT NOT NULL)')
            row = db.execute('SELECT result FROM channels WHERE inputs=?', (text,)).fetchone()
            if row is not None:
                return (json.loads(row[0]), True)
            value = compute()
            db.execute('INSERT INTO channels VALUES (?,?)', (text, json.dumps(value, allow_nan=False)))
            return (value, False)
