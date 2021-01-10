#!/usr/bin/env python3
# Developed by Xieyuanli Chen and Thomas Läbe
# This file is covered by the LICENSE file in the root of this project.

import numpy as np


def overlap_orientation_npz_file2string_string_nparray(npzfilenames, shuffle=True):
    """ Load overlap files as npz array and convert it to a tuple.
        Args:
          npzfilenames: list of npz filenames
          shuffle: Boolean, define whether to shuffle.
        Returns:
          A tupple (imgf1, imgf2, dir1, dir2, overlap) with
            imgf1: first partner of a pair, list of  n strings, formatted as the filenames (%06d)
            imgf1: second partner of a pair, list of n strings, formatted as the filenames (%06d)
            dir1 : n top directories of the dataset for imgf1 as list. This is the directory name of the sequence,
                   e.g. dir1[0]/depth_map/imgf1[0].png
            dir2 : n top directories of the dataset for imgf2 as list. This is the directory name of the sequence,
                   e.g. dir2[0]/depth_map/imgf2[0].png
            overlap: numpy array of size (n,)

       If no directory information is available in the npz (old format, only one
       array in the npz file), then dir1 and dir2 are all empty strings.
       
       The overlap and image file number are assumed to be in the second nx3 float array of
       the npz, whereas the directory names are in the first array nx2 (string) of the
       npz file.
       
       The data will be shuffled after loading if shuffle=True, thus the sequence of pairs will be 
       changed randomly.
    """
    imgf1_all = []
    imgf2_all = []
    dir1_all = []
    dir2_all = []
    overlap_all = []

    orientation_yaw_all = []
    orientation_pitch_all = []
    orientation_roll_all = []

    translation_x_all = []
    translation_y_all = []
    translation_z_all = []


    for npzfilename in npzfilenames:
        h = np.load(npzfilename, allow_pickle=True)

        if len(h.files) == 1:
            # old format
            imgf1 = np.char.mod('%06d', h[h.files[0]][:, 0]).tolist()
            imgf2 = np.char.mod('%06d', h[h.files[0]][:, 1]).tolist()
            overlap = h[h.files[0]][:, 2]
            orientation_yaw = h[h.files[0]][:, 3]
            orientation_pitch = h[h.files[0]][:, 4]
            orientation_roll = h[h.files[0]][:, 5]

            translation_x = h[h.files[0]][:, 6]
            translation_y = h[h.files[0]][:, 7]
            translation_z = h[h.files[0]][:, 8]

            n = len(imgf1)
            dir1 = np.array(['' for _ in range(n)]).tolist()
            dir2 = np.array(['' for _ in range(n)]).tolist()
        else:
            imgf1 = np.char.mod('%06d', h['overlaps'][:, 0]).tolist()
            imgf2 = np.char.mod('%06d', h['overlaps'][:, 1]).tolist()

            overlap = h['overlaps'][:, 2]

            orientation_yaw = h['overlaps'][:, 3]
            orientation_pitch = h['overlaps'][:, 4]
            orientation_roll = h['overlaps'][:, 5]

            translation_x = h['overlaps'][:, 6]
            translation_y = h['overlaps'][:, 7]
            translation_z = h['overlaps'][:, 8]

            dir1 = (h['seq'][:, 0]).tolist()
            dir2 = (h['seq'][:, 1]).tolist()

        if shuffle:
            shuffled_idx = np.random.permutation(overlap.shape[0])
            imgf1 = (np.array(imgf1)[shuffled_idx]).tolist()
            imgf2 = (np.array(imgf2)[shuffled_idx]).tolist()
            dir1 = (np.array(dir1)[shuffled_idx]).tolist()
            dir2 = (np.array(dir2)[shuffled_idx]).tolist()

            overlap = overlap[shuffled_idx]

            orientation_yaw = orientation_yaw[shuffled_idx]
            orientation_pitch = orientation_pitch[shuffled_idx]
            orientation_roll = orientation_roll[shuffled_idx]

            translation_x = translation_x[shuffled_idx]
            translation_y = translation_y[shuffled_idx]
            translation_z = translation_z[shuffled_idx]

        imgf1_all.extend(imgf1)
        imgf2_all.extend(imgf2)
        dir1_all.extend(dir1)
        dir2_all.extend(dir2)
        overlap_all.extend(overlap)

        orientation_yaw_all.extend(orientation_yaw)
        orientation_pitch_all.extend(orientation_pitch)
        orientation_roll_all.extend(orientation_roll)

        translation_x_all.extend(translation_x)
        translation_y_all.extend(translation_y)
        translation_z_all.extend(translation_z)

    return (imgf1_all,
            imgf2_all,
            dir1_all,
            dir2_all,
            np.asarray(overlap_all),
            np.asarray(orientation_yaw_all),
            np.asarray(orientation_pitch_all),
            np.asarray(orientation_roll_all),
            np.asarray(translation_x_all),
            np.asarray(translation_y_all),
            np.asarray(translation_z_all))
