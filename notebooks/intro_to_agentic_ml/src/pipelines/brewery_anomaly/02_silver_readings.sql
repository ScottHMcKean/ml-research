-- =============================================================================
-- Silver: enrich bronze with ISA-95 context + alarm thresholds
-- Reads incrementally from the bronze streaming table, joins dim_tag and
-- dim_asset to carry units, normal bands, and plant hierarchy forward.
-- =============================================================================
CREATE OR REFRESH STREAMING TABLE live_silver_readings
COMMENT 'Enriched telemetry with ISA-95 hierarchy and alarm thresholds'
AS SELECT
  CAST(b.timestamp / 1000 AS TIMESTAMP)  AS reading_ts,
  b.name                                 AS tag_id,
  b.alias                                AS asset_id,
  CAST(b.value AS DOUBLE)                AS value,
  b.quality                              AS quality_code,
  t.metric,
  t.unit,
  t.normal_low,
  t.normal_high,
  t.warn_threshold,
  t.crit_threshold,
  a.area,
  a.work_center,
  a.asset_type
FROM STREAM(LIVE.live_bronze_readings) b
LEFT JOIN ${catalog}.${brewery_schema}.dim_tag t
  ON b.name = t.tag_id
LEFT JOIN ${catalog}.${brewery_schema}.dim_asset a
  ON b.alias = a.asset_id;
