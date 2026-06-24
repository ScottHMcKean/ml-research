-- =============================================================================
-- Gold: anomaly detection — threshold breach + rolling z-score
-- Materialized view so it recomputes over the full silver table each refresh,
-- giving correct rolling windows across all history.
-- =============================================================================
CREATE OR REFRESH MATERIALIZED VIEW live_gold_anomalies
COMMENT 'Detected anomaly readings: critical threshold breach OR |z-score| > 3'
AS
WITH scored AS (
  SELECT
    reading_ts,
    tag_id,
    asset_id,
    area,
    value,
    crit_threshold,
    AVG(value)    OVER (PARTITION BY tag_id ORDER BY reading_ts ROWS BETWEEN 12 PRECEDING AND 1 PRECEDING) AS roll_mean,
    STDDEV(value) OVER (PARTITION BY tag_id ORDER BY reading_ts ROWS BETWEEN 12 PRECEDING AND 1 PRECEDING) AS roll_std
  FROM LIVE.live_silver_readings
)
SELECT
  reading_ts,
  tag_id,
  asset_id,
  area,
  value,
  crit_threshold,
  ROUND(CASE WHEN roll_std > 0 THEN (value - roll_mean) / roll_std ELSE 0.0 END, 2) AS zscore,
  CASE
    WHEN value > crit_threshold THEN 'crit_threshold'
    ELSE 'zscore>3'
  END AS detect_reason
FROM scored
WHERE value > crit_threshold
   OR ABS(CASE WHEN roll_std > 0 THEN (value - roll_mean) / roll_std ELSE 0.0 END) > 3.0;
