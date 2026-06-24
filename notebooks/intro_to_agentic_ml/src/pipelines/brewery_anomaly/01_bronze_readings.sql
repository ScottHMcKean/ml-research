-- =============================================================================
-- Bronze: Auto Loader streaming table from Sparkplug-B JSON micro-batches
-- =============================================================================
CREATE OR REFRESH STREAMING TABLE live_bronze_readings
COMMENT 'Raw Sparkplug-B telemetry ingested via Auto Loader from the landing volume'
AS SELECT
  *,
  current_timestamp()     AS _ingest_ts,
  _metadata.file_path     AS _source_file
FROM cloud_files(
  '/Volumes/${catalog}/${brewery_schema}/landing/incoming',
  'json',
  map('cloudFiles.inferColumnTypes', 'true',
      'rescuedDataColumn', '_rescued_data')
);
