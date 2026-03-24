PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;
PRAGMA busy_timeout = 5000;

CREATE TABLE IF NOT EXISTS documents (
  id           TEXT PRIMARY KEY,
  project      TEXT NOT NULL,
  file_path    TEXT NOT NULL,
  title        TEXT NOT NULL DEFAULT '',
  content_hash TEXT NOT NULL,
  file_size    INTEGER NOT NULL DEFAULT 0,
  tags         TEXT NOT NULL DEFAULT '[]',
  frontmatter  TEXT NOT NULL DEFAULT '{}',
  chunk_count  INTEGER NOT NULL DEFAULT 0,
  indexed_at   TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE(project, file_path)
);

CREATE TABLE IF NOT EXISTS chunks (
  id            TEXT PRIMARY KEY,
  document_id   TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  project       TEXT NOT NULL,
  content       TEXT NOT NULL,
  content_plain TEXT NOT NULL,
  token_count   INTEGER NOT NULL DEFAULT 0,
  heading_path  TEXT NOT NULL DEFAULT '[]',
  heading_depth INTEGER NOT NULL DEFAULT 0,
  chunk_index   INTEGER NOT NULL,
  has_code_block INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_project ON chunks(project);

CREATE TABLE IF NOT EXISTS embeddings (
  chunk_id   TEXT PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
  model_name TEXT NOT NULL,
  dimensions INTEGER NOT NULL,
  vector     BLOB NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  chunk_id UNINDEXED,
  title,
  content,
  tags
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
  INSERT INTO chunks_fts(chunk_id, title, content, tags)
  SELECT NEW.id,
         COALESCE((SELECT title FROM documents WHERE id = NEW.document_id), ''),
         NEW.content_plain,
         NEW.heading_path;
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
  DELETE FROM chunks_fts WHERE chunk_id = OLD.id;
END;

CREATE TABLE IF NOT EXISTS file_hashes (
  project    TEXT NOT NULL,
  file_path  TEXT NOT NULL,
  file_hash  TEXT NOT NULL,
  file_size  INTEGER NOT NULL DEFAULT 0,
  updated_at TEXT NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (project, file_path)
);
