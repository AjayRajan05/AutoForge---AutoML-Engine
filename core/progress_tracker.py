"""
Developer Experience Enhancement
Progress bars and live leaderboard for AutoML optimization
"""

import time
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import sys

# Try to import rich for fancy progress bars, fallback to basic
try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
    from rich.table import Table
    from rich.live import Live
    from rich.panel import Panel
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    # Fallback basic implementation
    class MockConsole:
        def print(self, *args, **kwargs): print(*args)
        def log(self, *args, **kwargs): print(*args)
    Console = MockConsole

logger = logging.getLogger(__name__)


@dataclass
class TrialResult:
    """Trial result data structure"""
    trial_number: int
    model_name: str
    score: float
    params: Dict[str, Any]
    state: str = "completed"
    duration: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class ModelPerformance:
    """Model performance tracking"""
    model_name: str
    best_score: float = float('-inf')
    avg_score: float = 0.0
    trial_count: int = 0
    scores: List[float] = field(default_factory=list)
    best_params: Dict[str, Any] = field(default_factory=dict)


class ProgressTracker:
    """
    Advanced progress tracking with live leaderboard
    """
    
    def __init__(self, 
                 show_progress: bool = True,
                 show_leaderboard: bool = True,
                 update_interval: float = 1.0,
                 console_width: int = 100):
        """
        Initialize progress tracker
        
        Args:
            show_progress: Whether to show progress bar
            show_leaderboard: Whether to show live leaderboard
            update_interval: Update interval for live displays
            console_width: Console width for formatting
        """
        self.show_progress = show_progress
        self.show_leaderboard = show_leaderboard
        self.update_interval = update_interval
        self.console_width = console_width
        
        # Trial tracking
        self.trials: List[TrialResult] = []
        self.model_performances: Dict[str, ModelPerformance] = {}
        self.current_trial = 0
        self.total_trials = 0
        self.start_time = time.time()
        
        # Rich console setup
        if RICH_AVAILABLE:
            self.console = Console(width=console_width)
            self.progress = None
            self.live_display = None
        else:
            self.console = Console()
        
        # Leaderboard data
        self.leaderboard_data = []
        self.last_update = time.time()
    
    def start_optimization(self, total_trials: int, task_type: str = "classification"):
        """
        Start optimization tracking
        
        Args:
            total_trials: Total number of trials
            task_type: Type of ML task
        """
        self.total_trials = total_trials
        self.start_time = time.time()
        self.current_trial = 0
        
        logger.info(f"Starting optimization: {total_trials} trials, task: {task_type}")
        
        if self.show_progress and RICH_AVAILABLE:
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=self.console
            )
            
            self.progress.start()
            self.task_id = self.progress.add_task(
                f"Optimizing {task_type}...", 
                total=total_trials
            )
    
    def add_trial_result(self, 
                        trial_number: int,
                        model_name: str,
                        score: float,
                        params: Dict[str, Any],
                        state: str = "completed",
                        duration: float = 0.0):
        """
        Add trial result to tracking
        
        Args:
            trial_number: Trial number
            model_name: Model name
            score: Trial score
            params: Trial parameters
            state: Trial state
            duration: Trial duration
        """
        # Create trial result
        trial = TrialResult(
            trial_number=trial_number,
            model_name=model_name,
            score=score,
            params=params.copy(),
            state=state,
            duration=duration
        )
        
        self.trials.append(trial)
        self.current_trial = trial_number
        
        # Update model performance
        self._update_model_performance(model_name, score, params)
        
        # Update displays
        self._update_displays()
        
        # Log significant events
        if state == "completed":
            if score > self._get_best_score():
                logger.info(f"🔥 NEW BEST: {model_name} = {score:.4f} (Trial {trial_number})")
            elif score > self._get_best_score() * 0.95:
                logger.info(f"Trial {trial_number} | {model_name} | {score:.4f}")
    
    def _update_model_performance(self, model_name: str, score: float, params: Dict[str, Any]):
        """Update model performance statistics"""
        if model_name not in self.model_performances:
            self.model_performances[model_name] = ModelPerformance(model_name=model_name)
        
        perf = self.model_performances[model_name]
        perf.scores.append(score)
        perf.trial_count += 1
        perf.avg_score = sum(perf.scores) / len(perf.scores)
        
        if score > perf.best_score:
            perf.best_score = score
            perf.best_params = params.copy()
    
    def _update_displays(self):
        """Update progress bar and leaderboard"""
        current_time = time.time()
        
        # Update progress bar
        if self.show_progress and self.progress:
            self.progress.update(self.task_id, completed=self.current_trial)
        
        # Update leaderboard (throttled)
        if (self.show_leaderboard and 
            current_time - self.last_update > self.update_interval):
            self._update_leaderboard()
            self.last_update = current_time
    
    def _update_leaderboard(self):
        """Update live leaderboard display"""
        if not RICH_AVAILABLE:
            self._print_basic_leaderboard()
            return
        
        # Create leaderboard table
        table = Table(title="🏆 Live Leaderboard", show_header=True, header_style="bold magenta")
        table.add_column("Rank", style="cyan", width=6)
        table.add_column("Model", style="green", width=25)
        table.add_column("Score", style="yellow", width=10)
        table.add_column("Avg", style="blue", width=8)
        table.add_column("Trials", style="white", width=7)
        table.add_column("Status", style="red", width=8)
        
        # Sort models by best score
        sorted_models = sorted(
            self.model_performances.values(),
            key=lambda x: x.best_score,
            reverse=True
        )
        
        # Add top models
        for rank, perf in enumerate(sorted_models[:10], 1):
            status = "🔥 BEST" if rank == 1 else f"Top {rank}"
            score_str = f"{perf.best_score:.4f}"
            avg_str = f"{perf.avg_score:.4f}"
            
            table.add_row(
                str(rank),
                perf.model_name,
                score_str,
                avg_str,
                str(perf.trial_count),
                status
            )
        
        # Create panel with table
        panel = Panel(
            table,
            title=f"Optimization Progress: {self.current_trial}/{self.total_trials}",
            border_style="blue"
        )
        
        # Update live display
        if self.live_display:
            self.live_display.update(panel)
        else:
            self.live_display = Live(panel, console=self.console, refresh_per_second=1)
            self.live_display.start()
    
    def _print_basic_leaderboard(self):
        """Print basic leaderboard without rich"""
        print("\n" + "="*60)
        print(f"🏆 LEADERBOARD - Trial {self.current_trial}/{self.total_trials}")
        print("="*60)
        
        sorted_models = sorted(
            self.model_performances.values(),
            key=lambda x: x.best_score,
            reverse=True
        )
        
        for rank, perf in enumerate(sorted_models[:5], 1):
            status = "🔥" if rank == 1 else f"#{rank}"
            print(f"{status} {perf.model_name:<25} {perf.best_score:.4f} ({perf.trial_count} trials)")
        
        print("="*60)
    
    def finish_optimization(self):
        """Finish optimization and display final results"""
        if self.progress:
            self.progress.stop()
        
        if self.live_display:
            self.live_display.stop()
        
        # Display final results
        self._display_final_results()
        
        logger.info(f"Optimization completed: {len(self.trials)} trials in {time.time() - self.start_time:.1f}s")
    
    def _display_final_results(self):
        """Display final optimization results"""
        if not self.trials:
            logger.info("No trials completed")
            return
        
        best_trial = max(self.trials, key=lambda t: t.score)
        best_score = best_trial.score
        best_model = best_trial.model_name
        
        print("\n" + "="*60)
        print("🎉 OPTIMIZATION COMPLETE")
        print("="*60)
        print(f"🏆 BEST MODEL: {best_model}")
        print(f"📊 BEST SCORE: {best_score:.4f}")
        print(f"🔢 TOTAL TRIALS: {len(self.trials)}")
        print(f"⏱️  TOTAL TIME: {time.time() - self.start_time:.1f}s")
        
        # Model summary
        print("\n📈 MODEL PERFORMANCE SUMMARY:")
        print("-" * 40)
        
        sorted_models = sorted(
            self.model_performances.values(),
            key=lambda x: x.best_score,
            reverse=True
        )
        
        for rank, perf in enumerate(sorted_models, 1):
            improvement = perf.best_score - perf.avg_score
            print(f"{rank:2d}. {perf.model_name:<20} {perf.best_score:.4f} (+{improvement:+.4f})")
        
        print("="*60)
    
    def _get_best_score(self) -> float:
        """Get current best score"""
        if not self.trials:
            return float('-inf')
        return max(t.score for t in self.trials if t.state == "completed")
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive optimization summary
        
        Returns:
            Optimization summary dictionary
        """
        if not self.trials:
            return {"error": "No trials completed"}
        
        completed_trials = [t for t in self.trials if t.state == "completed"]
        scores = [t.score for t in completed_trials]
        
        summary = {
            "total_trials": len(self.trials),
            "completed_trials": len(completed_trials),
            "best_score": max(scores) if scores else None,
            "best_model": self.trials[self.trials.index(max(completed_trials, key=lambda t: t.score))].model_name if completed_trials else None,
            "avg_score": sum(scores) / len(scores) if scores else None,
            "score_std": np.std(scores) if scores else None,
            "total_time": time.time() - self.start_time,
            "avg_trial_time": sum(t.duration for t in completed_trials) / len(completed_trials) if completed_trials else None,
            "models_tested": len(self.model_performances),
            "model_performance": {
                name: {
                    "best_score": perf.best_score,
                    "avg_score": perf.avg_score,
                    "trial_count": perf.trial_count
                }
                for name, perf in self.model_performances.items()
            }
        }
        
        return summary


class TrialProgress:
    """
    Individual trial progress tracking
    """
    
    def __init__(self, trial_number: int, model_name: str):
        """
        Initialize trial progress
        
        Args:
            trial_number: Trial number
            model_name: Model name
        """
        self.trial_number = trial_number
        self.model_name = model_name
        self.start_time = time.time()
        self.steps_completed = 0
        self.total_steps = 0
        self.current_score = 0.0
        self.best_trial_score = 0.0
    
    def update_step(self, step: int, score: float, total_steps: int):
        """
        Update trial progress
        
        Args:
            step: Current step
            score: Current score
            total_steps: Total steps
        """
        self.steps_completed = step
        self.total_steps = total_steps
        self.current_score = score
        
        if score > self.best_trial_score:
            self.best_trial_score = score
    
    def get_progress_info(self) -> Dict[str, Any]:
        """
        Get trial progress information
        
        Returns:
            Progress information dictionary
        """
        elapsed = time.time() - self.start_time
        progress = self.steps_completed / self.total_steps if self.total_steps > 0 else 0
        
        return {
            "trial_number": self.trial_number,
            "model_name": self.model_name,
            "progress": progress,
            "current_score": self.current_score,
            "best_score": self.best_trial_score,
            "elapsed_time": elapsed,
            "eta": elapsed / progress - elapsed if progress > 0 else None
        }


def create_progress_tracker(**kwargs) -> ProgressTracker:
    """
    Create progress tracker instance
    
    Args:
        **kwargs: Arguments for ProgressTracker
        
    Returns:
        ProgressTracker instance
    """
    return ProgressTracker(**kwargs)


# Global progress tracker instance
_global_tracker = None

def get_global_tracker() -> ProgressTracker:
    """Get global progress tracker instance"""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = ProgressTracker()
    return _global_tracker
