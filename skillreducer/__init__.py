"""SkillReducer — Optimize LLM agent skills for token efficiency."""

from .bundle import compare_bundles, inspect_bundle
from .pipeline import OptimizationResult, optimize_bundle

__all__ = ["OptimizationResult", "compare_bundles", "inspect_bundle", "optimize_bundle"]
__version__ = "0.2.0"
