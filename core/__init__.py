"""
AutoForge Core Module
"""

from .pipeline import Pipeline
from .pipeline_builder import build_pipeline
from .pipeline_cache import PipelineCache, CachedPipelineBuilder
from .dataset_optimizer import DatasetOptimizer
from .progress_tracker import ProgressTracker, create_progress_tracker
from .data_type_detector import DataTypeDetector, detect_data_type
from .search_space import get_search_space
from .engine_factory import EngineFactory

__all__ = [
    'Pipeline', 'build_pipeline', 'PipelineCache', 'CachedPipelineBuilder',
    'DatasetOptimizer', 'ProgressTracker', 'create_progress_tracker',
    'DataTypeDetector', 'detect_data_type', 'get_search_space',
    'EngineFactory'
]
