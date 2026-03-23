from setuptools import setup, find_packages

setup(
    name="automl_engine",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "optuna",
        "xgboost",
        "click",
        "joblib"
    ],
    entry_points={
        "console_scripts": [
            "automl=automl.cli.main:cli",
        ],
    },
)