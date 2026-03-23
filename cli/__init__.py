"""
AutoForge CLI Module
"""

from .main import cli
from .commands import train, predict, show_logs, validate

__all__ = ['cli', 'train', 'predict', 'show_logs', 'validate']
