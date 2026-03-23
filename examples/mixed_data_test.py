import pandas as pd
from api.automl import AutoML

data = pd.DataFrame({
    "age": [25, 30, None, 40],
    "salary": [50000, 60000, 55000, None],
    "city": ["Mumbai", "Delhi", "Mumbai", "Chennai"],
    "target": [0, 1, 0, 1]
})

X = data.drop("target", axis=1)
y = data["target"]

automl = AutoML(n_trials=10)
automl.fit(X, y)

preds = automl.predict(X)
print(preds)