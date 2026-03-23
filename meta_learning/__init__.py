"""
AutoForge Meta-Learning Module
"""

from .dataset_profiler import profile_dataset
from .knowledge_base import KnowledgeBase
from .recommender import MetaRecommender
from .pattern_learner import PatternLearner

__all__ = [
    'profile_dataset', 'KnowledgeBase', 'MetaRecommender', 'PatternLearner'
]
