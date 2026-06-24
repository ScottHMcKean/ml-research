-- =============================================================================
-- ai_query_anomaly_detection.sql
-- Score the LIVE brewery stream with the registered Isolation Forest model,
-- served at the `brewery-anomaly-iforest` Model Serving endpoint, directly from
-- SQL via ai_query() — no Python, no Spark UDF. Run in a DBSQL / SQL editor once
-- the endpoint is READY (see lab2_autoloader_model.py, Step 4, to deploy it).
--
-- Model signature: two tag-normalised features (band_pos, roll_z); the model
-- returns -1 for an anomaly and 1 for normal (Isolation Forest convention).
--
-- Note: the endpoint is scale-to-zero — the first query warms it (~30-60s), and
-- scoring the whole stream is one ai_query call per row, so a full scan takes a
-- couple of minutes. Add a WHERE (e.g. area='Glycol' or a recent reading_ts
-- window) to keep an interactive demo snappy. Validated: 137/228 GLY-01 return-temp
-- readings flagged (the live chiller-drift fault).
-- =============================================================================

WITH feat AS (
  SELECT
    reading_ts,
    tag_id,
    asset_id,
    area,
    value,
    (value - normal_low) / (normal_high - normal_low) AS band_pos,
    CASE
      WHEN stddev(value) OVER w > 0
      THEN (value - avg(value) OVER w) / (stddev(value) OVER w)
      ELSE 0.0
    END AS roll_z
  FROM ml_workshop.brewery.live_silver_readings
  WINDOW w AS (
    PARTITION BY tag_id ORDER BY reading_ts
    ROWS BETWEEN 12 PRECEDING AND 1 PRECEDING
  )
),
scored AS (
  SELECT
    reading_ts, tag_id, asset_id, area, value,
    round(band_pos, 3) AS band_pos,
    round(roll_z, 2)   AS roll_z,
    ai_query(
      'brewery-anomaly-iforest',
      named_struct('band_pos', band_pos, 'roll_z', roll_z),
      returnType => 'DOUBLE'
    ) AS anomaly_pred                       -- -1.0 = anomaly, 1.0 = normal
  FROM feat
  WHERE band_pos IS NOT NULL AND roll_z IS NOT NULL
)
SELECT *
FROM scored
WHERE anomaly_pred = -1                     -- keep only model-flagged anomalies
ORDER BY reading_ts DESC
LIMIT 100;
