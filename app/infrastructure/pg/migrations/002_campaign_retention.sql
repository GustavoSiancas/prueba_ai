CREATE TABLE IF NOT EXISTS campaign_retention (
  campaign_id  text PRIMARY KEY,
  end_date     date NOT NULL,
  created_at   timestamptz NOT NULL DEFAULT now(),
  updated_at   timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS campaign_retention_end_date_idx
  ON campaign_retention (end_date);
