"""
Meta-learning functionality for AutoML
"""

import logging
from typing import Dict, Any


class MetaLearningManager:
    """
    Handles meta-learning and pattern learning from experiments
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def learn_from_experiment(self, experiment_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Learn patterns from completed experiment to improve future performance.

        Args:
            experiment_result: Results from a completed AutoML experiment

        Returns:
            Learned insights and recommendations
        """
        try:
            from meta_learning.pattern_learner import PatternLearner
            pattern_learner = PatternLearner()
            insights = pattern_learner.learn_from_experiment(experiment_result)
            self.logger.info(f"Learned patterns from experiment: {len(insights)} insights")
            return insights

        except Exception as e:
            self.logger.error(f"Failed to learn from experiment: {e}")
            return {"error": str(e)}
