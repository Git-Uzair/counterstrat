"""Lake extraction: demo to parquet Level-0 tables."""

from counterstrat.lake.duck import connect_lake
from counterstrat.lake.extract import LakePaths, extract_lake

__all__ = ["LakePaths", "connect_lake", "extract_lake"]
