"""
Backtesting Reporting Module - Performance report generation.
"""

from .tearsheet import (
    TearsheetGenerator,
    TearsheetReport,
    generate_html_report,
    generate_json_report,
)


__all__ = [
    'TearsheetGenerator',
    'TearsheetReport',
    'generate_html_report',
    'generate_json_report',
]
