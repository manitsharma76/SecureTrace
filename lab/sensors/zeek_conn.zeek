## Dashboard-only connection log. Per CLAUDE.md "One flow extractor": Zeek output never feeds
## the ML model, it is read by the Streamlit dashboard's Integrity panel alongside the collector's
## events. Defaults (conn.log with the standard field set) are enough for that; no custom script
## logic is added here on purpose, to keep this clearly separate from the CICFlowMeter path.
@load base/protocols/conn
