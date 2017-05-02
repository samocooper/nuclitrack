import numpy as np
from sklearn.ensemble import RandomForestClassifier

def classifycells(features, training_data):

    training_data = np.delete(training_data, 0, 0)
    mask = np.sum(training_data[:, 12:17], axis=0) > 0

    if sum(mask) > 1:

        clf = RandomForestClassifier(n_estimators=100)
        inds = np.where(mask)[0]
        clf = clf.fit(training_data[:, 5:10], training_data[:, 12 + inds].astype(bool))
        probs = clf.predict_proba(features[:, 5:10])

        i = 0
        for p in probs:
            if len(p[0]) == 1:
                p = np.hstack([np.asarray(p), np.zeros((len(p), 1))])
            else:
                p = np.asarray(p)

            features[:, 12 + inds[i]] = p[:, 1]
            i += 1

    if sum(mask) == 1:
        ind = np.where(mask)[0]
        features[:, 12 + ind] = 1

    return features