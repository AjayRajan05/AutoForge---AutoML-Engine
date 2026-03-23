from sklearn.model_selection import train_test_split


def holdout_validation(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    model.fit(X_train, y_train)
    preds = model.predict(X_val)

    return preds, y_val