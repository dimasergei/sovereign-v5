# core/compliance/__init__.py
"""
Prop Firm Compliance Modules.

CRITICAL: These modules enforce prop firm rules to prevent account breaches.
All trading decisions MUST pass through compliance checks.
"""

from .gft_compliance import GFTComplianceChecker
from .the5ers_compliance import The5ersComplianceChecker

__all__ = ['GFTComplianceChecker', 'The5ersComplianceChecker']
