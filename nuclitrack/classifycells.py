import numpy as np
from sklearn.ensemble import RandomForestClassifier


def classifycells(features, training):

    training_tracking = np.delete(training['tracking'][...], 0, 0)
    training_data = np.delete(training['data'][...], 0, 0)

    mask = np.sum(training['tracking'][:, 6:12], axis=0) > 0

    if sum(mask) > 1:

        clf = RandomForestClassifier(n_estimators=100)

        inds = np.where(mask)[0]
        train = training_tracking[:, 6 + inds] == 1
        clf = clf.fit(training_data, train)
        probs = clf.predict_proba(features['data'][...])

        i = 0
        for p in probs:
            if len(p[0]) == 1:
                p = np.hstack([np.asarray(p), np.zeros((len(p), 1))])
            else:
                p = np.asarray(p)

            features['tracking'][:, 6 + inds[i]] = p[:, 1]
            i += 1

    if sum(mask) == 1:
        ind = np.where(mask)[0][0]
        features_temp = np.zeros(features['tracking'].shape)
        features_temp[:,:] = features['tracking'][:,:]
        features_temp[:, 6 + ind] = 1
        features['tracking'][:,:] = features_temp[:,:]

    return features
