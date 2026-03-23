from sklearn.feature_selection import SelectKBest, f_classif


def get_feature_selector(k=10):
    return SelectKBest(score_func=f_classif, k=k)