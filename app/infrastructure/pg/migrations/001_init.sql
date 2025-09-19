CREATE TABLE IF NOT EXISTS video_features (
  video_id     text PRIMARY KEY,
  campaign_id  text NOT NULL,
  url          text NOT NULL UNIQUE,
  phash64      bytea NOT NULL,
  seq_sig      bytea NOT NULL,
  seq_rows     int  NOT NULL,
  seq_cols     int  NOT NULL CHECK (seq_cols = 64),
  duration_s   double precision NOT NULL,
  created_at   timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS video_features_campaign_recent
  ON video_features (campaign_id, created_at DESC);
