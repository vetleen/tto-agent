"""Spreadsheet (.xlsx/.xlsm) ingestion: streaming reader, header detection,
tile-mesh planning, row-batched chunking, rendering, and the per-version
manifest. See each module's docstring; the package deliberately never loads
the full openpyxl workbook DOM (memory: default mode is ~30-50x file size).
"""
