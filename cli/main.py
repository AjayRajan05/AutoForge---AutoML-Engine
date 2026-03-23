import click
from cli.commands import train, predict, show_logs, validate

@click.group()
@click.version_option(version="0.1.0", prog_name="AutoForge AutoML")
@click.option("--verbose", is_flag=True, help="Enable verbose logging")
def cli(verbose):
    """
    AutoForge - Research-Grade AutoML Engine
    
    Features:
    - Full Pipeline Search (Preprocessing + Model + Features)
    - Hyperparameter Optimization (Optuna powered)
    - Meta-Learning Engine (learns from past runs)
    - Multi-Model Ensembling (Stacking + Blending)
    - Experiment Tracking System
    """
    if verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)


# Add commands
cli.add_command(train)
cli.add_command(predict)
cli.add_command(show_logs)
cli.add_command(validate)


if __name__ == "__main__":
    cli()