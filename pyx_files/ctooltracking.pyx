import numpy as np
cimport numpy as np
cimport cython

cdef extern from "math.h":
    float sqrt(float x)

DTYPE1 = np.float
ctypedef np.float_t DTYPE1_t

DTYPE2 = np.int
ctypedef np.int_t DTYPE2_t

@cython.boundscheck(False)
@cython.wraparound(False)


def distance_mat(np.ndarray[DTYPE1_t, ndim=2] features, int frames, np.ndarray[DTYPE1_t, ndim=1] param):

    cdef int hgt = features.shape[0]
    cdef int f, i, j, k, t

    # Determine number of edges to consider

    cdef np.ndarray[DTYPE2_t, ndim=1] frame_count = np.zeros( frames, dtype=DTYPE2)
    cdef np.ndarray[DTYPE2_t, ndim=1] frame_count_cum = np.zeros( frames+1, dtype=DTYPE2)

    for i in range(hgt):
        f = int(features[i,1])
        frame_count[f] += 1

    cdef int cons
    cdef int max_t = int(param[6])

    for t in range(1,max_t):
        for i in range(frames-t):
            cons += frame_count[i]*frame_count[i+t]


    for i in range(frames):
        frame_count_cum[i+1] = frame_count_cum[i] + frame_count[i]

    # Calculate edges and assign edge features

    cdef np.ndarray[DTYPE1_t, ndim=2] edges = np.zeros([cons,5], dtype=DTYPE1)
    cdef int count = 0
    cdef int ind2A, ind2B, ind1A, ind1B, f1, f2
    cdef float xd, x1, x2, yd, y1, y2, d

    for t in range(1,max_t):
        for i in range(t,frames):

            ind2A = frame_count_cum[i-t]
            ind2B = frame_count_cum[i-t+1]

            ind1A = frame_count_cum[i]
            ind1B = frame_count_cum[i+1]

            f1 = ind2B-ind2A
            f2 = ind1B-ind1A

            for k in range(f2):
                for j in range(f1):

                    d = sqrt((features[ind1A+k,3] - features[ind2A+j,3])**2 + (features[ind1A+k,2] - features[ind2A+j,2])**2)

                    edges[count,0] = i # Frame of cell after movement

                    edges[count,1] = features[ind2A+j,0]  # Index of the cell before movement
                    edges[count,2] = features[ind1A+k,0]  # Index of the cell after movement
                    edges[count,3] = d  # Distance moved in pixels
                    edges[count,4] = t  # Movement gap
                    count += 1

    # Determine number of edges where movements are below threshold pixels

    cdef int cons_f = 0

    for i in range(cons):
        if (edges[i,3] < param[1]) & (edges[i,1] != 0 ):
            cons_f += 1

    cdef np.ndarray[DTYPE1_t, ndim=2] edges_filtered = np.zeros([cons_f, 5], dtype=DTYPE1)

    # Filter edges where movements are below threshold pixels

    count = 0

    for i in range(cons):
        if (edges[i,3] < param[1]) & (edges[i,1] != 0 ):

            edges_filtered[count,:] = edges[i,:]
            edges_filtered[count,3] = 1 - (param[0]*edges_filtered[count,3])**2  # Distance score function

            count += 1

    edges_filtered = edges_filtered[edges_filtered[:, 2].argsort(), :]
    edges_filtered = np.vstack((edges_filtered, np.zeros((1, edges_filtered.shape[1]))))

    return edges_filtered

def cost_calc(np.ndarray[DTYPE1_t, ndim=2] frame_edges, int a, int b, int c, int d):

    cdef int hgt = frame_edges.shape[0]
    cdef int i
    cdef float cost = 0

    for i in range(hgt):
        if frame_edges[i,1] == a and frame_edges[i,2] == c:
            cost += frame_edges[i,3]
        if frame_edges[i,1] == b and frame_edges[i,2] == d:
            cost += frame_edges[i,3]
        if frame_edges[i,1] == b and frame_edges[i,2] == c:
            cost -= frame_edges[i,3]
        if frame_edges[i,1] == a and frame_edges[i,2] == d:
            cost -= frame_edges[i,3]

    return cost

def swaps_mat(np.ndarray[DTYPE1_t, ndim=2] edges, int frames):

    hgt = edges.shape[0]
    cdef np.ndarray[DTYPE1_t, ndim=2] s_mat = np.zeros([hgt*10,10], dtype=DTYPE1) # Estimate of swap No. increase if necessary
    cdef int count = 1
    cdef int i, j, k, m, n, lng
    for i in range(frames):

        frame_edges = edges[edges[:,0] == i,:]
        frame_unique = np.unique(frame_edges[:,2])

        for j in range(len(frame_unique)-1):

            l1 = set(frame_edges[frame_edges[:,2] == frame_unique[j], 1])

            for k in range(j+1,len(frame_unique)):
                l2 = set(frame_edges[frame_edges[:,2] == frame_unique[k], 1])

                edge_union = l1.intersection(l2)

                if edge_union:
                    if len(edge_union) > 1:

                        edge_union = list(edge_union)
                        lng = len(edge_union)

                        for m in range(lng-1):
                            for n in range(m+1, lng):

                                s_mat[count,0] = i
                                s_mat[count,3] = frame_unique[j]
                                s_mat[count,4] = frame_unique[k]
                                s_mat[count,1] = edge_union[m]
                                s_mat[count,2] = edge_union[n]
                                s_mat[count,9] = cost_calc(frame_edges, edge_union[m], edge_union[n], frame_unique[j], frame_unique[k])

                                count+=1

    return s_mat[:count,:]

def main_loop(np.ndarray[DTYPE1_t, ndim=2] scores, np.ndarray[DTYPE1_t, ndim=2] features, np.ndarray[DTYPE1_t, ndim=2] edges, np.ndarray[DTYPE1_t, ndim=2] swaps, np.ndarray[DTYPE2_t, ndim=1] states, np.ndarray[DTYPE1_t, ndim=1] param):

    cdef int hgt1 = features.shape[0]
    cdef int hgt2 = edges.shape[0]
    cdef int count2 = 0
    cdef int count_swap = 0
    cdef int count_frame = 0

    cdef float dist_id = edges[0,2]
    cdef float cell_id = edges[0,2]
    cdef int count1 = int(edges[0,2])

    cdef float frame, add_score, swap_score, temp_score, swap_id, max_swap, prev_id
    cdef bint test1, test2, test3, test4
    cdef int hgt_swap = swaps.shape[0]-1
    cdef float max_score = -100000
    cdef float max_id = 0

    while count2 < hgt2:

        dist_id = edges[count2,2]

        if dist_id != cell_id:

            scores[count1,1] = cell_id
            if max_score > scores[count1,3]:

                scores[count1,2] = max_id
                scores[count1,3] = max_score
                scores[count1,4] = 0
                scores[count1,5] = swap_id
                scores[count1,6] = max_swap

            count1 += int(dist_id - cell_id)
            cell_id = dist_id
            max_score = -100000

        s = states[count1]

        prev_id = edges[count2,1]

        # Track cost plus cost of adding track to the segment in question

        if s < 2:

            add_score = scores[int(prev_id),3] + param[2]*features[count1,7+s] - param[3]*features[count1,6+s]

        else:

            add_score = -10000

        # Cost associated with movement to the segment in question

        temp_score = add_score + edges[count2,3] - edges[count2,4] + 1

        if temp_score > max_score:  # Test if this score is greater than prior tested scores

            max_score = temp_score
            max_id = prev_id
            swap_id = 0

        # Identify frame starting point in swap matrix

        frame = edges[count2,0]

        while frame > swaps[count_swap,0] and count_swap < hgt_swap:
            count_swap += 1

        # Identify swaps in frame, sequential testing to minimise comparisons

        count_frame = count_swap

        while frame == swaps[count_frame,0] and count_frame < hgt_swap:

            test1 = swaps[count_frame,1] == prev_id
            test2 = swaps[count_frame,2] == prev_id

            if test1 or test2:

                test3 = swaps[count_frame,3] == dist_id
                test4 = swaps[count_frame,4] == dist_id

                if test3 or test4:

                    test1 = test1 and (swaps[count_frame,5] == 0.) and (swaps[count_frame,6] == 1.)
                    test2 = test2 and (swaps[count_frame,6] == 0.) and (swaps[count_frame,5] == 1.)

                    if test1 or test2:

                        test3 = test3 and (swaps[count_frame,7] == 0.) and (swaps[count_frame,8] == 1.)
                        test4 = test4 and (swaps[count_frame,8] == 0.) and (swaps[count_frame,7] == 1.)

                        if test3 or test4:

                            if test1 == test3:
                                swap_score = -swaps[count_frame,9]
                            else:
                                swap_score = swaps[count_frame,9]

                            temp_score = temp_score + swap_score

                            if temp_score > max_score:  # Test if this score is greater than prior tested scores

                                max_score = temp_score
                                max_id = prev_id
                                max_swap = swap_score

                                if swaps[count_frame,5] == 1:
                                    swap_id = swaps[count_frame,1]

                                if swaps[count_frame,6] == 1:
                                    swap_id = swaps[count_frame,2]

            count_frame += 1

        count2 += 1

    return scores


def forward_pass(np.ndarray[DTYPE1_t, ndim=2] features, np.ndarray[DTYPE1_t, ndim=2] edges, np.ndarray[DTYPE1_t, ndim=2] swaps, np.ndarray[DTYPE2_t, ndim=1] states, np.ndarray[DTYPE1_t, ndim=1] param):

    cdef int hgt1 = features.shape[0]
    cdef int hgt2 = edges.shape[0]

    cdef np.ndarray[DTYPE1_t, ndim=2] scores = np.zeros([hgt1,7], dtype=DTYPE1)

    cdef int count1 = 0
    cdef float temp_score, add_score, edge_score
    cdef float max_score = -100000
    cdef float max_id = 0

    ## First frame entry states ##

    while features[count1, 1] == 0:

        s = states[count1]

        if s < 2:

            scores[count1,3] = features[count1,7+s] - features[count1,6+s]
            scores[count1,1] = features[count1,0]

        else:

            scores[count1,3] = -100000

        count1 += 1  # Counts through cell features in the first frame

    ## Edge entry states ##

    cdef int count3 = count1
    cdef float temp_edge

    while count3 < hgt1:

        s = states[count3]
        scores[count3,1] = count3

        if s < 2:
            temp_edge = 1-(param[0]*features[count3,4])**2 + features[count3,7+s] - features[count3,6+s]

            if temp_edge > -3.:
                scores[count3,3] = temp_edge
            else:
                scores[count3,3] = -3.

            scores[count3,0] = features[count3,1]

        else:

            scores[count1,3] = -100000

        count3+=1 # Counts through cell features after the first frame

    ## Mitotic entry states ##

    cdef int cell_id = int(edges[0,2])
    cdef int dist_id, previous_cell_id

    cdef int count4 = int(edges[0,2])
    cdef int count5 = 0

    while count5 < hgt2:

        dist_id = int(edges[count5,2]) # ID of current segment

        if dist_id != cell_id: # Update cost matrix if next segment ID is different to current

            if max_score > scores[count4,3]: # Reassign edge if max mitosis score is greater than edge cost

                scores[count4,2] = max_id
                scores[count4,3] = max_score
                scores[count4,4] = 1

            count4 += (dist_id - cell_id)  # Counter giving position in features matrix of current ID
            cell_id = dist_id
            max_score = -100000

        s = states[count4]  # Current number of tracks assigned to segment

        previous_cell_id = int(edges[count5,1]) # ID of the cell that edge initiates from

        if states[previous_cell_id] > 0: # Test if track exists on previous frame

            # Migration Component

            if s < 2:

                add_score = param[2]*features[count4,7+s] - param[3]*features[count4,6+s]

            else:

                add_score = -100000   # need to replace with correct handling

            # Add distance subtract gap

            temp_score = add_score + edges[count5,3] - param[5]*(edges[count5,4] + 1)

             # Mitosis Component

            temp_score = temp_score + (features[previous_cell_id,9] + features[count4,10]) - param[4]

            if temp_score > max_score:

                max_score = temp_score
                max_id = edges[count5,1]

        count5 += 1

    scores = main_loop(scores, features, edges, swaps, states, param)

    return scores

def track_back(np.ndarray[DTYPE1_t, ndim=2] score_mat, np.ndarray[DTYPE2_t, ndim=1] states, np.ndarray[DTYPE1_t, ndim=2] swaps):

    cdef int hgt1 = score_mat.shape[0]
    cdef int i, max_ind
    cdef float max_score = 0

    for i in range(hgt1):
        if score_mat[i,3] > max_score:
            max_ind = i
            max_score = score_mat[i,3]

    cdef int ind = max_ind

    # Determine track size, convoluted perhaps faster method?

    cdef int track_length = 0

    while score_mat[ind,2] != 0 and score_mat[ind,4] == 0:

        ind = int(score_mat[ind,2])
        track_length += 1

    cdef np.ndarray[DTYPE1_t, ndim=2] track = np.zeros([track_length+1,8], dtype=DTYPE1)

    ind = max_ind
    cdef int count = track_length

    # maximum frame where swap may occur

    cdef swap_count = swaps.shape[0]-1

    while score_mat[ind,2] != 0 and score_mat[ind,4] == 0:

        track[count,5] = score_mat[ind,0] # Current frame of tracking
        track[count,2] = score_mat[ind,3] # Total score of track

        track[count,0] = ind # ID from score matrix

        states[ind] += 1

        ind_prev = int(score_mat[ind,2]) # 2nd column of score mat contains ind of previous segment

        count -= 1

        track[count,6] = score_mat[ind,5] # Mark any swaps that need performing
        track[count,7] = score_mat[ind,6] # Mark any swaps that need performing

        # Identify possible swaps

        while swaps[swap_count,0] > score_mat[ind,0]:
            swap_count -= 1

        # Update availability of swaps, probably better way but this is clear

        while swaps[swap_count,0] == score_mat[ind,0]:

            if ind_prev == swaps[swap_count,1] and ind == swaps[swap_count,3]:
                swaps[swap_count,5] = 1
                swaps[swap_count,7] = 1

            if ind_prev == swaps[swap_count,2] and ind == swaps[swap_count,3]:
                swaps[swap_count,6] = 1
                swaps[swap_count,7] = 1

            if ind_prev == swaps[swap_count,1] and ind == swaps[swap_count,4]:
                swaps[swap_count,5] = 1
                swaps[swap_count,8] = 1

            if ind_prev == swaps[swap_count,2] and ind == swaps[swap_count,4]:
                swaps[swap_count,6] = 1
                swaps[swap_count,8] = 1

            swap_count -=1
        ind = ind_prev

    track[count,5] = score_mat[ind,0] # Current frame of tracking
    track[count,2] = score_mat[ind,3]
    track[count,0] = ind # ID of first tracked segment
    states[ind] += 1

    if score_mat[ind,4] == 1:
           track[count,3] = score_mat[ind,2] # Flag mitotic event

    # Calculate gradients

    track[0,1] = track[0,2]

    for i in range(1,track_length+1):

        track[i,1] = track[i,2]-track[i-1,2]  # Score difference

    return track, swaps, states

def swap_test(np.ndarray[DTYPE1_t, ndim=2] tracks, np.ndarray[DTYPE1_t, ndim=2] track_temp, np.ndarray[DTYPE1_t, ndim=2] edges, int count):

    j = 0
    flag = False
    while j < track_temp.shape[0]:

        if track_temp[j, 6] != 0:

            # Handle Swaps

            # Identify current frame

            c_frame = track_temp[j, 5]

            # Identify and extract track to swap with

            if sum(tracks[:, 0] == track_temp[j, 6]) > 0: #FIX

                swap_id = tracks[tracks[:, 0] == track_temp[j, 6], 4]

                track_temp[j, 6] = 0

                for i in range(len(swap_id)): #FIX?

                    mask = tracks[:, 4] == swap_id[i]

                    swap_track = tracks[mask, :]

                    if sum(swap_track[:, 5] <= c_frame)>0 and sum(swap_track[:, 5] > c_frame)>0:
                        tracks = tracks[np.logical_not(mask), :]

                        # Identify swap position

                        t1 = track_temp[:j+1, :]
                        t2 = track_temp[j+1:, :]

                        st1 = swap_track[swap_track[:, 5] <= c_frame, :]
                        st2 = swap_track[swap_track[:, 5] > c_frame, :]

                        score_diff1 = st2[0, 2] - t1[-1, 2]
                        score_diff2 = t2[0, 2] - st1[-1, 2]

                        # Determine old and new distance cost

                        d_mask1 = np.logical_and(edges[:, 1] == t1[-1, 0], edges[:, 2] == st2[0, 0])
                        d_mask2 = np.logical_and(edges[:, 1] == st1[-1, 0], edges[:, 2] == t2[0, 0])

                        d_mask1a = np.logical_and(edges[:, 1] == t1[-1, 0], edges[:, 2] == t2[0, 0])
                        d_mask2a = np.logical_and(edges[:, 1] == st1[-1, 0], edges[:, 2] == st2[0, 0])

                        if sum(d_mask1)>0 and sum(d_mask2)>0 and sum(d_mask1a)>0 and sum(d_mask2a)>0:

                            d_score1 = edges[d_mask1, 3]
                            d_score2 = edges[d_mask2, 3]

                            d_score1a = edges[d_mask1a, 3]
                            d_score2a = edges[d_mask2a, 3]

                            score_swap1 = st2[0, 1] - d_score2a + d_score2
                            score_swap2 = t2[0, 1] - d_score1a + d_score1 - track_temp[j, 7]

                            # Update scores

                            st2[0, 1] = score_swap1
                            t2[0, 1] = score_swap2

                            st2[:, 2] = st2[:, 2] - score_diff1 + score_swap1
                            t2[:, 2] = t2[:, 2] - score_diff2 + score_swap2

                            # Perform swap

                            swap_track_new = np.vstack((t1, st2))
                            track_temp = np.vstack((st1, t2))

                            swap_track_new[:, 4] = swap_id[i]
                            track_temp[:, 4] = count

                            tracks = np.vstack((tracks, swap_track_new))
                            break
                        else:
                            flag = True
                            tracks = np.vstack((tracks, swap_track))
                    else:
                        flag = True
            else:
                flag = True
        j += 1

    return tracks, track_temp