import tracking_c_tools
import numpy as np

''' Create matrix tracks, Col0 = ID from feature matrix; Col1 = Score difference; Col2 = total Score;

    Col3 = mitosis; Col4 = Track_id; Col5 = frame.

    Tracking parameters are; 0) distance penalty (smaller means
    less penalty); 1) max distance searched in previous frame; 2 & 3) weighting for adding segments based
    on probability of segments, p2*(s+1) - p3*(s); 4) mitosis penalty reduce the likelihood of mitosis,
    negative values increase the likelihood; 5) weighting for gaps in tracking; 6) max gap'''

class TrackCells(object):

    def __init__(self, features, frames, track_param):

        features[:, 18] = 1
        self.features = np.vstack((np.zeros((1, features.shape[1])), features))

        self.states = np.zeros(self.features.shape[0], dtype=int)
        self.tracks = np.zeros([1, 8])

        self.d_mat = tracking_c_tools.distance_mat(self.features, frames, track_param)
        self.d_mat = self.d_mat[self.d_mat[:, 2].argsort(), :]
        self.track_param = track_param
        self.d_mat = np.vstack((self.d_mat, np.zeros((1, self.d_mat.shape[1]))))
        self.s_mat = tracking_c_tools.swaps_mat(self.d_mat, frames)
        self.cum_score = 0.

        self.count = 1
        self.optimise_count = 0

        self.min_score = 5
        self.max_score = self.min_score+1

    def addtrack(self):

        score_mat = tracking_c_tools.forward_pass(self.features, self.d_mat, self.s_mat, self.states, self.track_param)
        max_score = max(score_mat[:, 3])

        if max_score > self.min_score:

            self.cum_score += max_score
            track_temp, self.s_mat, self.states = tracking_c_tools.track_back(score_mat, self.states, self.s_mat)
            track_temp[:, 4] = self.count

            self.tracks, track_temp = tracking_c_tools.swap_test(self.tracks, track_temp, self.d_mat, self.count)
            self.tracks = np.vstack((self.tracks, track_temp))
            self.count += 1

            return True

        else:
            self.track_num = np.max(self.tracks[:, 4])
            return False

    def optimisetrack(self):

        if self.optimise_count < self.track_num:

            track_ind = self.optimise_count
            replace_mask = self.tracks[:, 4] == track_ind
            track_store = self.tracks[replace_mask, :]

            self.tracks = self.tracks[np.logical_not(replace_mask), :]

            for j in range(track_store.shape[0]):  # Remove track

                self.states[int(track_store[j, 0])] -= 1

                if j > 0:

                    ind1 = track_store[j - 1, 0]
                    ind2 = track_store[j, 0]

                    m1 = np.logical_and(self.s_mat[:, 1] == ind1, self.s_mat[:, 3] == ind2)
                    m2 = np.logical_and(self.s_mat[:, 2] == ind1, self.s_mat[:, 3] == ind2)
                    m3 = np.logical_and(self.s_mat[:, 1] == ind1, self.s_mat[:, 4] == ind2)
                    m4 = np.logical_and(self.s_mat[:, 2] == ind1, self.s_mat[:, 4] == ind2)

                    if any(m1):
                        self.s_mat[m1, 5] = 0
                        self.s_mat[m1, 7] = 0
                    if any(m2):
                        self.s_mat[m2, 6] = 0
                        self.s_mat[m2, 7] = 0
                    if any(m3):
                        self.s_mat[m3, 5] = 0
                        self.s_mat[m3, 8] = 0
                    if any(m4):
                        self.s_mat[m4, 6] = 0
                        self.s_mat[m4, 8] = 0

            score_mat = tracking_c_tools.forward_pass(self.features, self.d_mat, self.s_mat, self.states, self.track_param)
            max_score = max(score_mat[:, 3])

            if max_score > track_store[-1, 2]:

                self.cum_score = self.cum_score + max_score - track_store[-1, 2]
                track_replace, self.s_mat, self.states = tracking_c_tools.track_back(score_mat, self.states, self.s_mat)
                track_replace[:, 4] = track_ind

                self.tracks, track_replace = tracking_c_tools.swap_test(self.tracks, track_replace, self.d_mat, track_ind)
                self.tracks = np.vstack((self.tracks, track_replace))

            else:
                self.tracks = np.vstack((self.tracks, track_store))

                for j in range(track_store.shape[0]):

                    self.states[int(track_store[j, 0])] += 1

                    if j > 0:

                        ind1 = track_store[j - 1, 0]
                        ind2 = track_store[j - 1, 0]

                        m1 = np.logical_and(self.s_mat[:, 1] == ind1, self.s_mat[:, 3] == ind2)
                        m2 = np.logical_and(self.s_mat[:, 2] == ind1, self.s_mat[:, 3] == ind2)
                        m3 = np.logical_and(self.s_mat[:, 1] == ind1, self.s_mat[:, 4] == ind2)
                        m4 = np.logical_and(self.s_mat[:, 2] == ind1, self.s_mat[:, 4] == ind2)

                        if any(m1):
                            self.s_mat[m1, 5] = 1
                            self.s_mat[m1, 7] = 1
                        if any(m2):
                            self.s_mat[m2, 6] = 1
                            self.s_mat[m2, 7] = 1
                        if any(m3):
                            self.s_mat[m3, 5] = 1
                            self.s_mat[m3, 8] = 1
                        if any(m4):
                            self.s_mat[m4, 6] = 1
                            self.s_mat[m4, 8] = 1

            self.optimise_count += 1

            return True

        else:

            self.optimise_count = 0
            return False

    def get(self):

        self.tracks[:, 0] = self.tracks[:, 0] - 1
        unique, counts = np.unique(self.tracks[:, 0], return_counts=True)

        self.segment_count = len(counts)
        self.double_segment = sum(counts > 1)

        double_seg = unique[counts > 1]
        for val in double_seg:
            self.tracks = self.tracks[np.logical_not(self.tracks[:, 0] == val), :]

        self.features = self.features[1:, :]

        # Color labels

        r = np.round(252 * np.random.rand()) + 3
        ind = self.tracks[0, 4]

        for j in range(self.tracks.shape[0]):

            if self.tracks[j, 4] != ind:
                ind = self.tracks[j, 4]
                r = np.round(252 * np.random.rand()) + 3

            self.features[int(self.tracks[j, 0]), 18] = r

            if self.tracks[j, 3] > 0:
                self.features[int(self.tracks[j, 0]), 18] = 5

        self.tracks[:, 0] = self.tracks[:, 0] + 1

        return self.tracks, self.features, self.segment_count, self.double_segment
