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


''' Create matrix and write to csv for features. Features are: Track_id, Frame, X_center, Y_center, Area,
Eccentricity, Solidity, Perimeter, CH1 Mean Intensity, CH1 StdDev Intensity, CH1 Floored Mean, CH2 Mean Intensity,
CH2 StdDev Intensity, CH3 Mean Intensity, CH3 StdDev Intensity, '''


def save_csv(features, tracks, file_name):

    ''' Create matrix and write to csv for features. Features are: Track_id, Frame, X_center, Y_center, Area,
    Eccentricity, Solidity, Perimeter, CH1 Mean Intensity, CH1 StdDev Intensity, CH1 Floored Mean, CH2 Mean Intensity,
    CH2 StdDev Intensity, CH3 Mean Intensity, CH3 StdDev Intensity, '''
    if features.shape[1] == 20:
        features = np.hstack((features, np.zeros((features.shape[0], 6))))

    if features.shape[1] == 23:
        features = np.hstack((features, np.zeros((features.shape[0], 3))))

    feat_mat = np.zeros((1, 20))

    for i in range(1, int(max(tracks[:, 4])) + 1):
        if sum(tracks[:, 4] == i) > 0:

            track_temp = tracks[tracks[:, 4] == i, :]

            for j in range(track_temp.shape[0]):
                mask = features[:, 0] == track_temp[j, 0]
                fv = features[mask, :]
                feat_mat = np.vstack((feat_mat,
                                      [i, track_temp[j, 5], fv[0, 2], fv[0, 3], fv[0, 5], fv[0, 6], fv[0, 7],
                                       fv[0, 8], fv[0, 9], fv[0, 10], fv[0, 11], fv[0, 20], fv[0, 21],
                                       fv[0, 22], fv[0, 23], fv[0, 24], fv[0, 25], track_temp[j, 3],
                                       track_temp[j, 0], 0]))

    feat_mat = np.delete(feat_mat, 0, 0)

    for i in range(feat_mat.shape[0]):

        if feat_mat[i, 17] > 0:

            mask = feat_mat[:, 18] == feat_mat[i, 17]
            ind_change = feat_mat[mask, 0]

            frame_change = feat_mat[mask, 1]
            mask_change = np.logical_and(feat_mat[:, 0] == ind_change, feat_mat[:, 1] > frame_change)
            if sum(mask_change) > 0:
                # feat_mat[mask_change, 0] = max(feat_mat[:, 0]) + 1  #option to change index of parent track daughter cell

                change_list = np.where(mask_change)
                feat_mat[change_list[0][0], 19] = ind_change
                feat_mat[i, 19] = ind_change

    with open(file_name, 'wb') as f:
        f.write(b'Track ID, Frame, X center, Y center, Area, Eccentricity, Solidity, Perimeter, '
                b'CH1 Mean Intensity, CH1 StdDev Intensity, CH1 Floored Mean, CH2 Mean Intensity, '
                b'CH2 StdDev Intensity, CH3 Mean Intensity, CH3 StdDev Intensity, Parent Track ID\n')

        feat_mat2 = np.delete(feat_mat, [17, 18], 1)
        np.savetxt(f, feat_mat2, delimiter=",", fmt='%10.4f')

    '''
    ### Data formatting for iscb benchmark dataset

    for i in range(feat_mat.shape[0]):

        if feat_mat[i,15] > 0:

            mask = feat_mat[:, 16] == feat_mat[i, 15]
            ind_change = feat_mat[mask, 0]

            frame_change = feat_mat[mask, 1]
            mask_change = np.logical_and(feat_mat[:,0] == ind_change, feat_mat[:,1] > frame_change)
            if sum(mask_change)>0:
                feat_mat[mask_change, 0] = max(feat_mat[:, 0]) + 1
                change_list = np.where(mask_change)
                feat_mat[change_list[0][0], 17] = ind_change
                feat_mat[i, 17] = ind_change

    np.savetxt("Results2.csv", feat_mat, delimiter=",")

    track_text = np.zeros((int(max(feat_mat[:,0])), 4))

    for i in range(1, int(max(feat_mat[:,0])+1)):

        track_temp = feat_mat[feat_mat[:,0]==i,:]
        track_text[i-1, 0] = i
        track_text[i-1, 1] = track_temp[0,1]
        track_text[i-1, 2] = track_temp[-1,1]
        track_text[i-1, 3] = track_temp[0,17]

    np.savetxt("res_track.txt", track_text.astype(int),fmt='%i', delimiter=" ")

    for i in range(self.frames):

        im_temp = self.labels[i, :, :]
        im_tracked = np.zeros(im_temp.shape)

        l_temp = feat_mat[feat_mat[:, 1] == i,:]

        for j in range(l_temp.shape[0]):
            ind_seg = im_temp[int(l_temp[j, 3]), int(l_temp[j, 2])]
            im_tracked[im_temp == ind_seg] = l_temp[j,0]

        n = str(i)
        n = n.zfill(2)
        n = 'mask' + n + '.tif'
        im_tracked = im_tracked.astype(np.uint16)
        tifffile.imsave(n, im_tracked)
    '''


def save_sel_csv(features, tracks, tracks_stored, file_name):

    if features.shape[1] == 20:
        features = np.hstack((features, np.zeros((features.shape[0], 6))))

    if features.shape[1] == 23:
        features = np.hstack((features, np.zeros((features.shape[0], 3))))

    feat_mat = np.zeros((1, 20))

    for i in range(1, int(max(tracks[:, 4]))):
        if tracks_stored[i] == 1:
            if sum(tracks[:, 4] == i) > 0:

                track_temp = tracks[tracks[:, 4] == i, :]

                for j in range(track_temp.shape[0]):
                    mask = features[:, 0] == track_temp[j, 0]
                    fv = features[mask, :]
                    feat_mat = np.vstack((feat_mat,
                                          [i, track_temp[j, 5], fv[0, 2], fv[0, 3], fv[0, 5], fv[0, 6], fv[0, 7],
                                           fv[0, 8], fv[0, 9], fv[0, 10], fv[0, 11], fv[0, 20], fv[0, 21],
                                           fv[0, 22], fv[0, 23], fv[0, 24], fv[0, 25], track_temp[j, 3],
                                           track_temp[j, 0], 0]))

    feat_mat = np.delete(feat_mat, 0, 0)

    for i in range(feat_mat.shape[0]):

        if feat_mat[i, 17] > 0:

            mask = feat_mat[:, 18] == feat_mat[i, 17]
            ind_change = feat_mat[mask, 0]

            frame_change = feat_mat[mask, 1]
            mask_change = np.logical_and(feat_mat[:, 0] == ind_change, feat_mat[:, 1] > frame_change)
            if sum(mask_change) > 0:
                # feat_mat[mask_change, 0] = max(feat_mat[:, 0]) + 1  #option to change index of parent track daughter cell

                change_list = np.where(mask_change)
                feat_mat[change_list[0][0], 19] = ind_change
                feat_mat[i, 19] = ind_change

    with open(file_name, 'wb') as f:

        f.write(b'Track ID, Frame, X center, Y center, Area, Eccentricity, Solidity, Perimeter, '
                b'CH1 Mean Intensity, CH1 StdDev Intensity, CH1 Floored Mean, CH2 Mean Intensity, '
                b'CH2 StdDev Intensity, CH3 Mean Intensity, CH3 StdDev Intensity, Parent Track ID\n')

        feat_mat2 = np.delete(feat_mat, [17, 18], 1)
        np.savetxt(f, feat_mat2, delimiter=",", fmt='%10.4f')

