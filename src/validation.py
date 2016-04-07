from sklearn.cross_validation import cross_val_score


def cross_validation(classifier, feature, labels):
    """sth like crossvalidation"""
    # from sklearn import metrics
    scores = cross_val_score(classifier, feature, labels, cv=5, scoring='log_loss')
    return scores
