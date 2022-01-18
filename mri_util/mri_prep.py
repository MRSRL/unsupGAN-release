"""Data preparation for training."""
import os
import os.path
import random
import shutil
import subprocess
import sys
import zipfile
from os import path

import numpy as np
import scipy.misc
import tensorflow as tf
import wget

from mri_util import cfl, fftc, recon, tf_util

BIN_BART = "bart"

def setup_data_original(
    dir_in_root,
    dir_out,
    data_divide=(0.75, 0.05, 0.2),
    min_shape=[80, 180],
    num_maps=1,
    crop_maps=False,
    verbose=False,
):
    """Setups training data as tfrecords.
    prep_data.setup_data('/mnt/raid3/data/Studies_DCE/recon-ccomp6/',
        '/mnt/raid3/jycheng/Project/deepspirit/data/train/', verbose=True)
    """
    if verbose:
        print("Directory names:")
        print("  Input root:  %s" % dir_in_root)
        print("  Output root: %s" % dir_out)

    file_kspace = "kspace"
    file_sensemap = "sensemap"

    case_list = os.listdir(dir_in_root)
    random.shuffle(case_list)
    num_cases = len(case_list)

    i_train_1 = np.round(data_divide[0] * num_cases).astype(int)
    i_validate_0 = i_train_1 + 1
    i_validate_1 = np.round(
        data_divide[1] * num_cases).astype(int) + i_validate_0

    if not os.path.exists(dir_out):
        os.mkdir(dir_out)
    if not os.path.exists(os.path.join(dir_out, "train")):
        os.mkdir(os.path.join(dir_out, "train"))
    if not os.path.exists(os.path.join(dir_out, "validate")):
        os.mkdir(os.path.join(dir_out, "validate"))
    if not os.path.exists(os.path.join(dir_out, "test")):
        os.mkdir(os.path.join(dir_out, "test"))

    i_case = 0
    for case_name in case_list:
        file_kspace_i = os.path.join(dir_in_root, case_name, file_kspace)
        file_sensemap_i = os.path.join(dir_in_root, case_name, file_sensemap)

        if i_case < i_train_1:
            dir_out_i = os.path.join(dir_out, "train")
        elif i_case < i_validate_1:
            dir_out_i = os.path.join(dir_out, "validate")
        else:
            dir_out_i = os.path.join(dir_out, "test")

        if verbose:
            print("Processing [%d] %s..." % (i_case, case_name))
        i_case = i_case + 1

        if not os.path.exists(file_kspace_i + ".hdr"):
            print("skipping due to kspace header not existing in this folder")
            continue

        kspace = np.squeeze(cfl.read(file_kspace_i))

        num_coils = kspace.shape[0]
        if num_coils is not 32:
            print("skipping due to incorrect number of coils")
            continue

        if (min_shape is None) or (
            min_shape[0] <= kspace.shape[1] and min_shape[1] <= kspace.shape[2]
        ):

            if verbose:
                print("  Slice shape: (%d, %d)" %
                      (kspace.shape[1], kspace.shape[2]))
                print("  Num channels: %d" % kspace.shape[0])
            shape_x = kspace.shape[-1]
            kspace = fftc.ifftc(kspace, axis=-1)
            kspace = kspace.astype(np.complex64)

            # print(kspace.shape)
            # resize kspace by cropping
            # kspace_resized = recon.crop(
            #     kspace,
            #     out_shape=[num_coils, min_shape[0], min_shape[1], shape_x],
            #     verbose=True,
            # )

            # file_kspace_i = file_kspace_i + "_resized"
            # cfl.write(file_kspace_i, kspace_resized)

            cmd_flags = ""
            if crop_maps:
                cmd_flags = cmd_flags + " -c 1e-9"
            cmd_flags = cmd_flags + (" -m %d" % num_maps)
            cmd = "%s ecalib %s %s %s" % (
                BIN_BART,
                cmd_flags,
                file_kspace_i,
                file_sensemap_i,
            )
            if verbose:
                print("  Estimating sensitivity maps (bart espirit)...")
                print("    %s" % cmd)
            subprocess.check_call(["bash", "-c", cmd])
            sensemap = np.squeeze(cfl.read(file_sensemap_i))
            sensemap = np.expand_dims(sensemap, axis=0)
            sensemap = sensemap.astype(np.complex64)

            if verbose:
                print("  Creating tfrecords (%d)..." % shape_x)
            for i_x in range(shape_x):
                file_out = os.path.join(
                    dir_out_i, "%s_x%03d.tfrecords" % (case_name, i_x)
                )
                kspace_x = kspace[:, :, :, i_x]
                sensemap_x = sensemap[:, :, :, :, i_x]

                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            "name": _bytes_feature(str.encode(case_name)),
                            "xslice": _int64_feature(i_x),
                            "ks_shape_x": _int64_feature(kspace.shape[3]),
                            "ks_shape_y": _int64_feature(kspace.shape[2]),
                            "ks_shape_z": _int64_feature(kspace.shape[1]),
                            "ks_shape_c": _int64_feature(kspace.shape[0]),
                            "map_shape_x": _int64_feature(sensemap.shape[4]),
                            "map_shape_y": _int64_feature(sensemap.shape[3]),
                            "map_shape_z": _int64_feature(sensemap.shape[2]),
                            "map_shape_c": _int64_feature(sensemap.shape[1]),
                            "map_shape_m": _int64_feature(sensemap.shape[0]),
                            "ks": _bytes_feature(kspace_x.tostring()),
                            "map": _bytes_feature(sensemap_x.tostring()),
                        }
                    )
                )

                tf_writer = tf.python_io.TFRecordWriter(file_out)
                tf_writer.write(example.SerializeToString())
                tf_writer.close()


def setup_data_tfrecords_DCE(
    dir_in_root,
    dir_out,
    data_divide=(0.75, 0.05, 0.2),
    min_shape=[80, 180],
    num_maps=1,
    crop_maps=False,
    verbose=False,
    shuffle=True,
):
    """Setups training data as tfrecords.

    prep_data.setup_data('/mnt/raid3/data/Studies_DCE/recon-ccomp6/',
        '/mnt/raid3/jycheng/Project/deepspirit/data/train/', verbose=True)
    """
    if verbose:
        print("Directory names:")
        print("  Input root:  %s" % dir_in_root)
        print("  Output root: %s" % dir_out)

    # edits for /mnt/dense/data/MFAST_DCE and /home_local/ekcole/MFAST_DCE
    dir_kspace = "sort-ccomp6"
    dir_map = "recon-ccomp6"
    #     dir_kspace = "dce-ccomp6"

    dir_cases = os.path.join(dir_in_root, dir_kspace)

    file_kspace = "ks_sorted"
    file_sensemap = "map"

    case_list = os.listdir(dir_cases)
    # shuffle cases (patients)
    if shuffle is True:
        random.shuffle(case_list)
    else:
        print("don't shuffle dataset")
    num_cases = len(case_list)

    i_train_1 = np.round(data_divide[0] * num_cases).astype(int)
    i_validate_0 = i_train_1 + 1
    i_validate_1 = np.round(
        data_divide[1] * num_cases).astype(int) + i_validate_0

    if not os.path.exists(dir_out):
        os.mkdir(dir_out)
    if not os.path.exists(os.path.join(dir_out, "train")):
        os.mkdir(os.path.join(dir_out, "train"))
    if not os.path.exists(os.path.join(dir_out, "validate")):
        os.mkdir(os.path.join(dir_out, "validate"))
    if not os.path.exists(os.path.join(dir_out, "test")):
        os.mkdir(os.path.join(dir_out, "test"))

    i_case = 0
    for case_name in case_list:
        file_kspace_i = os.path.join(
            dir_in_root, dir_kspace, case_name, file_kspace)
        file_sensemap_i = os.path.join(
            dir_in_root, dir_map, case_name, file_sensemap)
        file_sensemap_i_check = os.path.join(
            dir_in_root, dir_map, case_name, file_sensemap + ".cfl"
        )

        # if no map, skip this case and do nothing
        if not path.exists(file_sensemap_i_check):
            print("Does not exist")
            continue

        # get dims from .hdr
        h = open(file_kspace_i + ".hdr", "r")
        h.readline()  # skip
        l = h.readline()
        h.close()
        dims = [int(i) for i in l.split()]
        print(dims)
        ky = dims[1]
        kz = dims[2]
        if ky != 180 or kz != 80:
            print("wrong dimensions")
            continue

        if i_case < i_train_1:
            dir_out_i = os.path.join(dir_out, "train")
        elif i_case < i_validate_1:
            dir_out_i = os.path.join(dir_out, "validate")
        else:
            dir_out_i = os.path.join(dir_out, "test")

        if verbose:
            print("Processing [%d] %s..." % (i_case, case_name))
        i_case = i_case + 1

        #         if(i_case >= 50):
        #             break

        kspace = np.squeeze(cfl.read(file_kspace_i))
        # it wants coils x y z frames
        kspace = np.transpose(kspace, [1, -1, -2, 2, 0])
        if verbose:
            print("  Slice shape: (%d, %d)" %
                  (kspace.shape[2], kspace.shape[3]))
            print("  Num channels: %d" % kspace.shape[0])
            print("  Num frames: %d" % kspace.shape[-1])
        # number of frames
        shape_f = kspace.shape[-1]
        # number of slices in x direction
        shape_x = kspace.shape[1]

        kspace = fftc.ifftc(kspace, axis=1)
        kspace = kspace.astype(np.complex64)

        sensemap = np.squeeze(cfl.read(file_sensemap_i))
        sensemap = sensemap[0, :, :, :, :]
        # we want coils x y z
        sensemap = np.transpose(sensemap, [0, -1, 2, 1])
        sensemap = np.expand_dims(sensemap, axis=0)
        sensemap = sensemap.astype(np.complex64)

        if verbose:
            print("  Creating tfrecords (%d)..." % shape_x)
        # Need to iterate over both z and frames

        for i_x in range(shape_x):
            # normalization across time frames
            kspace_x = kspace[:, i_x, :, :, :]
            max_frames = np.max(np.abs(kspace_x))
            #             print(max_frames)
            for i_f in range(shape_f):
                file_out = os.path.join(
                    dir_out_i, "%s_x%03d_f%03d.tfrecords" % (
                        case_name, i_x, i_f)
                )
                kspace_x = kspace[:, i_x, :, :, i_f] / max_frames
                sensemap_x = sensemap[:, :, i_x, :, :]

                #                 #save images as pngs to check if time frames shuffling is done here
                # #                 ks = np.squeeze(kspace_x)
                #                 ks = kspace_x
                #                 print(ks.shape)
                #                 ks = np.transpose(ks, [1,2,0])
                # #                 ks = np.expand_dims(ks, 0)
                #                 ks = tf.convert_to_tensor(ks)
                #                 print("ks")
                #                 print(ks)

                #                 sense = np.squeeze(sensemap_x)
                #                 print(sense.shape)
                #                 sense = np.transpose(sense, [1,2,0])
                #                 sense = np.expand_dims(sense, -2)
                #                 sense = tf.convert_to_tensor(sense)
                #                 print("sensemap")
                #                 print(sense)

                #                 image_x = tf_util.model_transpose(ks, sense)

                #                 sess = tf.Session()

                #                 # Evaluate the tensor `c`.
                #                 image_x = sess.run(image_x)

                #                 filename = dir_out_i + '/images/case' + str(i_x) + '_f' + str(i_f) + '.png'
                #                 print(filename)
                #                 scipy.misc.imsave(filename, np.squeeze(np.abs(image_x)))

                # at this stage, the images were not shuffled

                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            "name": _bytes_feature(str.encode(case_name)),
                            "xslice": _int64_feature(i_x),
                            "ks_shape_x": _int64_feature(kspace.shape[1]),
                            "ks_shape_y": _int64_feature(kspace.shape[2]),
                            "ks_shape_z": _int64_feature(kspace.shape[3]),
                            "ks_shape_c": _int64_feature(kspace.shape[0]),
                            "map_shape_x": _int64_feature(sensemap.shape[2]),
                            "map_shape_y": _int64_feature(sensemap.shape[3]),
                            "map_shape_z": _int64_feature(sensemap.shape[4]),
                            "map_shape_c": _int64_feature(sensemap.shape[1]),
                            "map_shape_m": _int64_feature(sensemap.shape[0]),
                            "ks": _bytes_feature(kspace_x.tostring()),
                            "map": _bytes_feature(sensemap_x.tostring()),
                        }
                    )
                )

                tf_writer = tf.python_io.TFRecordWriter(file_out)
                tf_writer.write(example.SerializeToString())
                tf_writer.close()


def setup_data_tfrecords_3d(
    dir_in_root,
    dir_out,
    data_divide=(0.8, 0.1, 0.2),
    min_shape=[80, 180],
    num_maps=1,
    crop_maps=False,
    verbose=False,
    shuffle=True,
):
    """Setups training data as tfrecords.

    prep_data.setup_data('/mnt/raid3/data/Studies_DCE/recon-ccomp6/',
        '/mnt/raid3/jycheng/Project/deepspirit/data/train/', verbose=True)
    """

    # Check for two echos in here
    # Use glob to find if have echo01

    if verbose:
        print("Directory names:")
        print("  Input root:  %s" % dir_in_root)
        print("  Output root: %s" % dir_out)

    # edits for /mnt/dense/data/MFAST_DCE and /home_local/ekcole/MFAST_DCE
    dir_kspace = "sort-ccomp6"
    dir_map = "recon-ccomp6"

    dir_cases = os.path.join(dir_in_root, dir_kspace)

    file_kspace = "ks_sorted"
    file_sensemap = "map"

    case_list = os.listdir(dir_cases)
    if shuffle is True:
        random.shuffle(case_list)
    else:
        print("don't shuffle dataset")
    num_cases = len(case_list)

    i_train_1 = np.round(data_divide[0] * num_cases).astype(int)
    i_validate_0 = i_train_1 + 1
    i_validate_1 = np.round(
        data_divide[1] * num_cases).astype(int) + i_validate_0

    if not os.path.exists(dir_out):
        os.mkdir(dir_out)
    if not os.path.exists(os.path.join(dir_out, "train")):
        os.mkdir(os.path.join(dir_out, "train"))
    if not os.path.exists(os.path.join(dir_out, "validate")):
        os.mkdir(os.path.join(dir_out, "validate"))
    if not os.path.exists(os.path.join(dir_out, "test")):
        os.mkdir(os.path.join(dir_out, "test"))

    i_case = 0
    for case_name in case_list:
        file_kspace_i = os.path.join(
            dir_in_root, dir_kspace, case_name, file_kspace)
        file_sensemap_i = os.path.join(
            dir_in_root, dir_map, case_name, file_sensemap)
        file_sensemap_i_check = os.path.join(
            dir_in_root, dir_map, case_name, file_sensemap + ".cfl"
        )
        if i_case < i_train_1:
            dir_out_i = os.path.join(dir_out, "train")
        elif i_case < i_validate_1:
            dir_out_i = os.path.join(dir_out, "validate")
        else:
            dir_out_i = os.path.join(dir_out, "test")
        print("dir out")
        print(dir_out_i)

        if verbose:
            print("Processing [%d] %s..." % (i_case, case_name))
        i_case = i_case + 1

        # if no map, skip this case
        # do nothing
        if not path.exists(file_sensemap_i_check):
            print("Sensitivity map does not exist")
            continue

        # get dims from .hdr
        h = open(file_kspace_i + ".hdr", "r")
        h.readline()  # skip
        l = h.readline()
        h.close()
        dims = [int(i) for i in l.split()]
        print(dims)
        ky = dims[1]
        kz = dims[2]
        if ky != 180 or kz != 80:
            print("wrong dimensions")
            continue

        kspace = np.squeeze(cfl.read(file_kspace_i))

        # it wants coils x y z frames
        kspace = np.transpose(kspace, [1, -1, -2, 2, 0])

        if verbose:
            print("  Slice shape: (%d, %d)" %
                  (kspace.shape[2], kspace.shape[3]))
            print("  Num channels: %d" % kspace.shape[0])
            print("  Num frames: %d" % kspace.shape[-1])
        # number of frames
        shape_f = kspace.shape[-1]
        # number of slices in x direction
        num_slices = kspace.shape[1]

        kspace = fftc.ifftc(kspace, axis=1)
        kspace = kspace.astype(np.complex64)

        print("Exists")
        sensemap = np.squeeze(cfl.read(file_sensemap_i))
        sensemap = sensemap[0, :, :, :, :]
        # it has coils z y x
        # 6, 80, 156, 192
        #         print(sensemap.shape)
        # we want coils x y z
        sensemap = np.transpose(sensemap, [0, -1, 2, 1])
        sensemap = np.expand_dims(sensemap, axis=0)
        sensemap = sensemap.astype(np.complex64)

        if verbose:
            print("  Creating tfrecords (%d)..." % num_slices)
        # for 2D plus time, only iterate over slices, not time frames

        for i_slice in range(num_slices):
            # normalization across time frames
            kspace_x = kspace[:, i_slice, :, :, :]
            file_out = os.path.join(
                dir_out_i, "%s_x%03d.tfrecords" % (case_name, i_slice)
            )
            #             kspace_x = kspace[:, i_x, :, :, i_f]/max_frames
            sensemap_x = sensemap[:, :, i_slice, :, :]
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "name": _bytes_feature(str.encode(case_name)),
                        "slice": _int64_feature(i_slice),
                        "ks_shape_x": _int64_feature(kspace.shape[1]),
                        "ks_shape_y": _int64_feature(kspace.shape[2]),
                        "ks_shape_z": _int64_feature(kspace.shape[3]),
                        "ks_shape_t": _int64_feature(kspace.shape[4]),
                        "ks_shape_c": _int64_feature(kspace.shape[0]),
                        "map_shape_x": _int64_feature(sensemap.shape[2]),
                        "map_shape_y": _int64_feature(sensemap.shape[3]),
                        "map_shape_z": _int64_feature(sensemap.shape[4]),
                        "map_shape_c": _int64_feature(sensemap.shape[1]),
                        "map_shape_m": _int64_feature(sensemap.shape[0]),
                        "ks": _bytes_feature(kspace_x.tostring()),
                        "map": _bytes_feature(sensemap_x.tostring()),
                    }
                )
            )

            tf_writer = tf.python_io.TFRecordWriter(file_out)
            tf_writer.write(example.SerializeToString())
            tf_writer.close()


def setup_data_tfrecords_MFAST(
    dir_in_root,
    dir_out,
    data_divide=(0.75, 0.05, 0.2),
    min_shape=[80, 180],
    num_maps=1,
    crop_maps=False,
    verbose=False,
):
    """Setups training data as tfrecords.
    prep_data.setup_data('/mnt/raid3/data/Studies_DCE/recon-ccomp6/',
        '/mnt/raid3/jycheng/Project/deepspirit/data/train/', verbose=True)
    """
    if verbose:
        print("Directory names:")
        print("  Input root:  %s" % dir_in_root)
        print("  Output root: %s" % dir_out)

    file_kspace = "kspace"
    file_sensemap = "sensemap"

    case_list = os.listdir(dir_in_root)
    random.shuffle(case_list)
    num_cases = len(case_list)

    i_train_1 = np.round(data_divide[0] * num_cases).astype(int)
    i_validate_0 = i_train_1 + 1
    i_validate_1 = np.round(
        data_divide[1] * num_cases).astype(int) + i_validate_0

    if not os.path.exists(dir_out):
        os.mkdir(dir_out)
    if not os.path.exists(os.path.join(dir_out, "train")):
        os.mkdir(os.path.join(dir_out, "train"))
    if not os.path.exists(os.path.join(dir_out, "validate")):
        os.mkdir(os.path.join(dir_out, "validate"))
    if not os.path.exists(os.path.join(dir_out, "test")):
        os.mkdir(os.path.join(dir_out, "test"))

    i_case = 0
    for case_name in case_list:
        file_kspace_i = os.path.join(dir_in_root, case_name, file_kspace)
        file_sensemap_i = os.path.join(dir_in_root, case_name, file_sensemap)

        if not os.path.exists(file_kspace_i + ".hdr"):
            print("skipping due to kspace not existing in this folder")
            continue

        if i_case < i_train_1:
            dir_out_i = os.path.join(dir_out, "train")
        elif i_case < i_validate_1:
            dir_out_i = os.path.join(dir_out, "validate")
        else:
            dir_out_i = os.path.join(dir_out, "test")

        if verbose:
            print("Processing [%d] %s..." % (i_case, case_name))
        i_case = i_case + 1

        kspace = np.squeeze(cfl.read(file_kspace_i))

        num_coils = kspace.shape[0]
        if num_coils is not 32:
            print("skipping due to incorrect number of coils")
            continue

        if verbose:
            print("  Slice shape: (%d, %d)" %
                  (kspace.shape[1], kspace.shape[2]))
            print("  Num channels: %d" % kspace.shape[0])
        shape_x = kspace.shape[-1]

        # for n in range(shape_x):
        #     modulation = (-1) ** n
        #     kspace[:, :, :, n] = kspace[:, :, :, n] * [
        #         1,
        #         1,
        #         1,
        #         np.exp(-1j * modulation),
        #     ]
        # print("kspace shape after modulation")

        # should this be here?
        kspace = fftc.ifftc(kspace, axis=-1)
        print("original kspace shape")
        print(kspace.shape)

        # x,y,z,coils
        # crop or zero pad to the correct size
        #         if (kspace.shape[0] is not min_shape[0]) or (kspace_shape[1] is not min_shape[1]):
        #             print("resizing")
        #             image = fftc.ifft2c(kspace, do_orthonorm=True, order='C')
        #             new_shape = (kspace.shape[0], min_shape[0], min_shape[1], kspace.shape[-1])
        #             resized_im = image.copy()
        #             resized_im.resize(new_shape, refcheck=False)
        #             kspace = fftc.fft2c(resized_im, do_orthonorm=True, order='C')

        kspace = kspace.astype(np.complex64)

        print("new kspace shape")
        print(kspace.shape)
        file_kspace_i = file_kspace_i + "_resized"
        cfl.write(file_kspace_i, kspace)

        cmd_flags = ""
        if crop_maps:
            cmd_flags = cmd_flags + " -c 1e-9"
        cmd_flags = cmd_flags + (" -m %d" % num_maps)
        cmd = "%s ecalib %s %s %s" % (
            BIN_BART,
            cmd_flags,
            file_kspace_i,
            file_sensemap_i,
        )
        if verbose:
            print("  Estimating sensitivity maps (bart espirit)...")
            print("    %s" % cmd)
        subprocess.check_call(["bash", "-c", cmd])
        sensemap = np.squeeze(cfl.read(file_sensemap_i))
        sensemap = np.expand_dims(sensemap, axis=0)
        sensemap = sensemap.astype(np.complex64)

        if verbose:
            print("  Creating tfrecords (%d)..." % shape_x)
        for i_x in range(shape_x):
            file_out = os.path.join(
                dir_out_i, "%s_x%03d.tfrecords" % (case_name, i_x))
            kspace_x = kspace[:, :, :, i_x]
            sensemap_x = sensemap[:, :, :, :, i_x]

            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "name": _bytes_feature(str.encode(case_name)),
                        "xslice": _int64_feature(i_x),
                        "ks_shape_x": _int64_feature(kspace.shape[3]),
                        "ks_shape_y": _int64_feature(kspace.shape[2]),
                        "ks_shape_z": _int64_feature(kspace.shape[1]),
                        "ks_shape_c": _int64_feature(kspace.shape[0]),
                        "map_shape_x": _int64_feature(sensemap.shape[4]),
                        "map_shape_y": _int64_feature(sensemap.shape[3]),
                        "map_shape_z": _int64_feature(sensemap.shape[2]),
                        "map_shape_c": _int64_feature(sensemap.shape[1]),
                        "map_shape_m": _int64_feature(sensemap.shape[0]),
                        "ks": _bytes_feature(kspace_x.tostring()),
                        "map": _bytes_feature(sensemap_x.tostring()),
                    }
                )
            )

            tf_writer = tf.python_io.TFRecordWriter(file_out)
            tf_writer.write(example.SerializeToString())
            tf_writer.close()


def setup_knee_data_tfrecords(dir_in_root, dir_out,
                              data_divide=(.75, .05, .2),
                              min_shape=[80, 180],
                              num_maps=1, crop_maps=False,
                              verbose=False):
    """Setups training data as tfrecords.
    prep_data.setup_data('/mnt/raid3/data/Studies_DCE/recon-ccomp6/',
        '/mnt/raid3/jycheng/Project/deepspirit/data/train/', verbose=True)
    """
    if verbose:
        print("Directory names:")
        print("  Input root:  %s" % dir_in_root)
        print("  Output root: %s" % dir_out)

    file_kspace = "kspace"
    file_sensemap = "sensemap"

    case_list = os.listdir(dir_in_root)
    random.shuffle(case_list)
    num_cases = len(case_list)

    i_train_1 = np.round(data_divide[0]*num_cases).astype(int)
    i_validate_0 = i_train_1 + 1
    i_validate_1 = np.round(data_divide[1]*num_cases).astype(int) \
        + i_validate_0

    if not os.path.exists(dir_out):
        os.mkdir(dir_out)
    if not os.path.exists(os.path.join(dir_out, "train")):
        os.mkdir(os.path.join(dir_out, "train"))
    if not os.path.exists(os.path.join(dir_out, "validate")):
        os.mkdir(os.path.join(dir_out, "validate"))
    if not os.path.exists(os.path.join(dir_out, "test")):
        os.mkdir(os.path.join(dir_out, "test"))

    i_case = 0
    for case_name in case_list:
        file_kspace_i = os.path.join(dir_in_root, case_name, file_kspace)
        file_sensemap_i = os.path.join(dir_in_root, case_name, file_sensemap)

        if i_case < i_train_1:
            dir_out_i = os.path.join(dir_out, "train")
        elif i_case < i_validate_1:
            dir_out_i = os.path.join(dir_out, "validate")
        else:
            dir_out_i = os.path.join(dir_out, "test")

        if verbose:
            print("Processing [%d] %s..." % (i_case, case_name))
        i_case = i_case + 1

        kspace = np.squeeze(cfl.read(file_kspace_i))
        if (min_shape is None) or (min_shape[0] <= kspace.shape[1] and
                                   min_shape[1] <= kspace.shape[2]):
            if verbose:
                print("  Slice shape: (%d, %d)" %
                      (kspace.shape[1], kspace.shape[2]))
                print("  Num channels: %d" % kspace.shape[0])
            shape_x = kspace.shape[-1]
            kspace = fftc.ifftc(kspace, axis=-1)
            kspace = kspace.astype(np.complex64)

            # if shape_c_out < shape_c:
            #     if verbose:
            #         print("  applying coil compression (%d -> %d)..." %
            #               (shape_c, shape_c_out))
            #     shape_cal = 24
            #     ks_cal = recon.crop(ks, [-1, shape_cal, shape_cal, -1])
            #     ks_cal = np.reshape(ks_cal, [shape_c,
            #                                  shape_cal*shape_cal,
            #                                  shape_x])
            #     cc_mat = coilcomp.calc_gcc_weights_c(ks_cal, shape_c_out)
            #     ks_cc = np.reshape(ks, [shape_c, -1, shape_x])
            #     ks_cc = coilcomp.apply_gcc_weights_c(ks_cc, cc_mat)
            #     ks = np.reshape(ks_cc, [shape_c_out, shape_z, shape_y, shape_x])

            cmd_flags = ""
            if crop_maps:
                cmd_flags = cmd_flags + " -c 1e-9"
            cmd_flags = cmd_flags + (" -m %d" % num_maps)
            cmd = "%s ecalib %s %s %s" % (
                BIN_BART, cmd_flags, file_kspace_i, file_sensemap_i)
            if verbose:
                print("  Estimating sensitivity maps (bart espirit)...")
                print("    %s" % cmd)
            subprocess.check_call(["bash", "-c", cmd])
            sensemap = np.squeeze(cfl.read(file_sensemap_i))
            sensemap = np.expand_dims(sensemap, axis=0)
            sensemap = sensemap.astype(np.complex64)

            if verbose:
                print("  Creating tfrecords (%d)..." % shape_x)
            for i_x in range(shape_x):
                file_out = os.path.join(
                    dir_out_i, "%s_x%03d.tfrecords" % (case_name, i_x))
                kspace_x = kspace[:, :, :, i_x]
                sensemap_x = sensemap[:, :, :, :, i_x]

                example = tf.train.Example(features=tf.train.Features(feature={
                    'name': _bytes_feature(str.encode(case_name)),
                    'xslice': _int64_feature(i_x),
                    'ks_shape_x': _int64_feature(kspace.shape[3]),
                    'ks_shape_y': _int64_feature(kspace.shape[2]),
                    'ks_shape_z': _int64_feature(kspace.shape[1]),
                    'ks_shape_c': _int64_feature(kspace.shape[0]),
                    'map_shape_x': _int64_feature(sensemap.shape[4]),
                    'map_shape_y': _int64_feature(sensemap.shape[3]),
                    'map_shape_z': _int64_feature(sensemap.shape[2]),
                    'map_shape_c': _int64_feature(sensemap.shape[1]),
                    'map_shape_m': _int64_feature(sensemap.shape[0]),
                    'ks': _bytes_feature(kspace_x.tostring()),
                    'map': _bytes_feature(sensemap_x.tostring())
                }))

                tf_writer = tf.python_io.TFRecordWriter(file_out)
                tf_writer.write(example.SerializeToString())
                tf_writer.close()


def setup_data_tfrecords(
    dir_in_root,
    dir_out,
    data_divide=(0.75, 0.05, 0.2),
    min_shape=[80, 180],
    num_maps=1,
    crop_maps=False,
    verbose=False,
):
    """Setups training data as tfrecords.
    prep_data.setup_data('/mnt/raid3/data/Studies_DCE/recon-ccomp6/',
        '/mnt/raid3/jycheng/Project/deepspirit/data/train/', verbose=True)
    """
    if verbose:
        print("Directory names:")
        print("  Input root:  %s" % dir_in_root)
        print("  Output root: %s" % dir_out)

    file_kspace = "kspace"
    file_sensemap = "sensemap"

    case_list = os.listdir(dir_in_root)
    random.shuffle(case_list)
    num_cases = len(case_list)

    i_train_1 = np.round(data_divide[0] * num_cases).astype(int)
    i_validate_0 = i_train_1 + 1
    i_validate_1 = np.round(
        data_divide[1] * num_cases).astype(int) + i_validate_0

    if not os.path.exists(dir_out):
        os.mkdir(dir_out)
    if not os.path.exists(os.path.join(dir_out, "train")):
        os.mkdir(os.path.join(dir_out, "train"))
    if not os.path.exists(os.path.join(dir_out, "validate")):
        os.mkdir(os.path.join(dir_out, "validate"))
    if not os.path.exists(os.path.join(dir_out, "test")):
        os.mkdir(os.path.join(dir_out, "test"))

    i_case = 0
    for case_name in case_list:
        file_kspace_i = os.path.join(dir_in_root, case_name, file_kspace)
        file_sensemap_i = os.path.join(dir_in_root, case_name, file_sensemap)

        if verbose:
            print("Processing [%d] %s..." % (i_case, case_name))

        if i_case < i_train_1:
            dir_out_i = os.path.join(dir_out, "train")
        elif i_case < i_validate_1:
            dir_out_i = os.path.join(dir_out, "validate")
        else:
            dir_out_i = os.path.join(dir_out, "test")

        i_case = i_case + 1

        if not os.path.exists(file_kspace_i + ".hdr"):
            print("skipping due to kspace not existing in this folder")
            continue

        kspace = np.squeeze(cfl.read(file_kspace_i))
        print("original kspace shape")
        print(kspace.shape)

        shape_x = kspace.shape[3]
        shape_y = kspace.shape[2]
        shape_z = kspace.shape[1]
        num_coils = kspace.shape[0]

        if num_coils is not 32:
            print("skipping due to incorrect number of coils")
            continue

        if min_shape[0] == kspace.shape[1] and min_shape[1] == kspace.shape[2]:
            if verbose:
                print("  Slice shape: (%d, %d)" %
                      (kspace.shape[1], kspace.shape[2]))
                print("  Num channels: %d" % kspace.shape[0])

            #  shape_x = kspace.shape[-1]
            # fix +1, -1 modulation along readout direction
            # for n in range(shape_x):
            #     modulation = (-1)**n
            #     kspace[:,:,:,n] = kspace[:,:,:,n]*np.exp(-1j*modulation)
            # print("kspace shape after modulation")
            # print(kspace.shape)
            # readout in kx
            kspace = fftc.ifftc(kspace, axis=-1)

            cmd_flags = ""
            if crop_maps:
                cmd_flags = cmd_flags + " -c 1e-9"
            # smoothing flag
            cmd_flags = cmd_flags + (" -S")

            cmd_flags = cmd_flags + (" -m %d" % num_maps)
            cmd = "%s ecalib %s %s %s" % (
                BIN_BART,
                cmd_flags,
                file_kspace_i,
                file_sensemap_i,
            )
            if verbose:
                print("  Estimating sensitivity maps (bart espirit)...")
                print("    %s" % cmd)
            subprocess.check_call(["bash", "-c", cmd])
            sensemap = np.squeeze(cfl.read(file_sensemap_i))
            sensemap = np.expand_dims(sensemap, axis=0)
            sensemap = sensemap.astype(np.complex64)

            if verbose:
                print("  Creating tfrecords (%d)..." % shape_x)
            for i_x in range(shape_x):
                file_out = os.path.join(
                    dir_out_i, "%s_x%03d.tfrecords" % (case_name, i_x)
                )
                kspace_x = kspace[:, :, :, i_x]
                sensemap_x = sensemap[:, :, :, :, i_x]

                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            "name": _bytes_feature(str.encode(case_name)),
                            "xslice": _int64_feature(i_x),
                            "ks_shape_x": _int64_feature(kspace.shape[3]),
                            "ks_shape_y": _int64_feature(kspace.shape[2]),
                            "ks_shape_z": _int64_feature(kspace.shape[1]),
                            "ks_shape_c": _int64_feature(kspace.shape[0]),
                            "map_shape_x": _int64_feature(sensemap.shape[4]),
                            "map_shape_y": _int64_feature(sensemap.shape[3]),
                            "map_shape_z": _int64_feature(sensemap.shape[2]),
                            "map_shape_c": _int64_feature(sensemap.shape[1]),
                            "map_shape_m": _int64_feature(sensemap.shape[0]),
                            "ks": _bytes_feature(kspace_x.tostring()),
                            "map": _bytes_feature(sensemap_x.tostring()),
                        }
                    )
                )

                tf_writer = tf.python_io.TFRecordWriter(file_out)
                tf_writer.write(example.SerializeToString())
                tf_writer.close()
        else:
            print("skipping due to wrong slice dimensions")


def process_DCE(example, num_channels=None, num_emaps=None):
    """Process TFRecord to actual tensors."""
    features = tf.parse_single_example(
        example,
        features={
            "name": tf.FixedLenFeature([], tf.string),
            "slice": tf.FixedLenFeature([], tf.int64),
            "ks_shape_x": tf.FixedLenFeature([], tf.int64),
            "ks_shape_y": tf.FixedLenFeature([], tf.int64),
            "ks_shape_z": tf.FixedLenFeature([], tf.int64),
            "ks_shape_t": tf.FixedLenFeature([], tf.int64),
            "ks_shape_c": tf.FixedLenFeature([], tf.int64),
            "map_shape_x": tf.FixedLenFeature([], tf.int64),
            "map_shape_y": tf.FixedLenFeature([], tf.int64),
            "map_shape_z": tf.FixedLenFeature([], tf.int64),
            "map_shape_c": tf.FixedLenFeature([], tf.int64),
            "map_shape_m": tf.FixedLenFeature([], tf.int64),
            "ks": tf.FixedLenFeature([], tf.string),
            "map": tf.FixedLenFeature([], tf.string),
        },
    )

    name = features["name"]

    xslice = tf.cast(features["slice"], dtype=tf.int32)

    #     ks_shape_y = tf.cast(features['ks_shape_y'], dtype=tf.int32)
    #     ks_shape_z = tf.cast(features['ks_shape_z'], dtype=tf.int32)
    #     ks_shape_c = tf.cast(features['ks_shape_c'], dtype=tf.int32)
    #     ks_shape_t = tf.cast(features['ks_shape_t'], dtype=tf.int32)
    ks_shape_y = 180
    ks_shape_z = 80
    ks_shape_c = 6
    ks_shape_t = 18

    map_shape_y = 180
    map_shape_z = 80
    map_shape_c = 6

    if num_emaps is None:
        map_shape_m = tf.cast(features["map_shape_m"], dtype=tf.int32)
    else:
        map_shape_m = num_emaps

    with tf.name_scope("kspace"):
        ks_record_bytes = tf.decode_raw(features["ks"], tf.float32)
        # treat time frames as separate slices
        #         image_shape = [ks_shape_c, ks_shape_y, ks_shape_z]
        image_shape = [ks_shape_c, ks_shape_y, ks_shape_z, ks_shape_t]
        ks_x = tf.reshape(ks_record_bytes, image_shape + [2])
        ks_x = tf_util.channels_to_complex(ks_x)
        ks_x = tf.reshape(ks_x, image_shape)

    with tf.name_scope("sensemap"):
        map_record_bytes = tf.decode_raw(features["map"], tf.float32)
        map_shape = [map_shape_m * map_shape_c, map_shape_y, map_shape_z]
        map_x = tf.reshape(map_record_bytes, map_shape + [2])
        map_x = tf_util.channels_to_complex(map_x)
        map_x = tf.reshape(map_x, map_shape)

    return name, xslice, ks_x, map_x


def process_DCE_2D(example, num_channels=None, num_emaps=None):
    """Process TFRecord to actual tensors."""
    features = tf.parse_single_example(
        example,
        features={
            "name": tf.FixedLenFeature([], tf.string),
            "xslice": tf.FixedLenFeature([], tf.int64),
            "ks_shape_x": tf.FixedLenFeature([], tf.int64),
            "ks_shape_y": tf.FixedLenFeature([], tf.int64),
            "ks_shape_z": tf.FixedLenFeature([], tf.int64),
            "ks_shape_c": tf.FixedLenFeature([], tf.int64),
            "map_shape_x": tf.FixedLenFeature([], tf.int64),
            "map_shape_y": tf.FixedLenFeature([], tf.int64),
            "map_shape_z": tf.FixedLenFeature([], tf.int64),
            "map_shape_c": tf.FixedLenFeature([], tf.int64),
            "map_shape_m": tf.FixedLenFeature([], tf.int64),
            "ks": tf.FixedLenFeature([], tf.string),
            "map": tf.FixedLenFeature([], tf.string),
        },
    )
    name = features["name"]
    xslice = tf.cast(features["xslice"], dtype=tf.int32)
    ks_shape_y = 180
    ks_shape_z = 80
    ks_shape_c = 6

    map_shape_y = 180
    map_shape_z = 80
    map_shape_c = 6

    if num_emaps is None:
        map_shape_m = tf.cast(features["map_shape_m"], dtype=tf.int32)
    else:
        map_shape_m = num_emaps

    with tf.name_scope("kspace"):
        ks_record_bytes = tf.decode_raw(features["ks"], tf.float32)
        # treat time frames as separate slices
        #         image_shape = [ks_shape_c, ks_shape_y, ks_shape_z]
        image_shape = [ks_shape_c, ks_shape_y, ks_shape_z]
        ks_x = tf.reshape(ks_record_bytes, image_shape + [2])
        ks_x = tf_util.channels_to_complex(ks_x)
        ks_x = tf.reshape(ks_x, image_shape)

    with tf.name_scope("sensemap"):
        map_record_bytes = tf.decode_raw(features["map"], tf.float32)
        map_shape = [map_shape_m * map_shape_c, map_shape_y, map_shape_z]
        map_x = tf.reshape(map_record_bytes, map_shape + [2])
        map_x = tf_util.channels_to_complex(map_x)
        map_x = tf.reshape(map_x, map_shape)

    return name, xslice, ks_x, map_x


def process_tfrecord(example, num_channels=None, num_emaps=None):
    """Process TFRecord to actual tensors."""
    features = tf.parse_single_example(
        example,
        features={
            "name": tf.FixedLenFeature([], tf.string),
            "xslice": tf.FixedLenFeature([], tf.int64),
            "ks_shape_x": tf.FixedLenFeature([], tf.int64),
            "ks_shape_y": tf.FixedLenFeature([], tf.int64),
            "ks_shape_z": tf.FixedLenFeature([], tf.int64),
            "ks_shape_c": tf.FixedLenFeature([], tf.int64),
            "map_shape_x": tf.FixedLenFeature([], tf.int64),
            "map_shape_y": tf.FixedLenFeature([], tf.int64),
            "map_shape_z": tf.FixedLenFeature([], tf.int64),
            "map_shape_c": tf.FixedLenFeature([], tf.int64),
            "map_shape_m": tf.FixedLenFeature([], tf.int64),
            "ks": tf.FixedLenFeature([], tf.string),
            "map": tf.FixedLenFeature([], tf.string),
        },
    )

    name = features["name"]

    xslice = tf.cast(features["xslice"], dtype=tf.int32)
    # shape_x = tf.cast(features['shape_x'], dtype=tf.int32)
    ks_shape_y = tf.cast(features["ks_shape_y"], dtype=tf.int32)
    ks_shape_z = tf.cast(features["ks_shape_z"], dtype=tf.int32)
    if num_channels is None:
        ks_shape_c = tf.cast(features["ks_shape_c"], dtype=tf.int32)
    else:
        ks_shape_c = num_channels
    map_shape_y = tf.cast(features["map_shape_y"], dtype=tf.int32)
    map_shape_z = tf.cast(features["map_shape_z"], dtype=tf.int32)
    if num_channels is None:
        map_shape_c = tf.cast(features["map_shape_c"], dtype=tf.int32)
    else:
        map_shape_c = num_channels
    if num_emaps is None:
        map_shape_m = tf.cast(features["map_shape_m"], dtype=tf.int32)
    else:
        map_shape_m = num_emaps

    with tf.name_scope("kspace"):
        ks_record_bytes = tf.decode_raw(features["ks"], tf.float32)
        image_shape = [ks_shape_c, ks_shape_z, ks_shape_y]
        ks_x = tf.reshape(ks_record_bytes, image_shape + [2])
        ks_x = tf_util.channels_to_complex(ks_x)
        ks_x = tf.reshape(ks_x, image_shape)

    with tf.name_scope("sensemap"):
        map_record_bytes = tf.decode_raw(features["map"], tf.float32)
        map_shape = [map_shape_m * map_shape_c, map_shape_z, map_shape_y]
        map_x = tf.reshape(map_record_bytes, map_shape + [2])
        map_x = tf_util.channels_to_complex(map_x)
        map_x = tf.reshape(map_x, map_shape)

    return name, xslice, ks_x, map_x


def read_tfrecord_with_sess(tf_sess, filename_tfrecord):
    """Read TFRecord for debugging."""
    tf_reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer([filename_tfrecord])
    _, serialized_example = tf_reader.read(filename_queue)
    name, xslice, ks_x, map_x = process_tfrecord(serialized_example)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=tf_sess, coord=coord)
    name, xslice, ks_x, map_x = tf_sess.run([name, xslice, ks_x, map_x])
    coord.request_stop()
    coord.join(threads)

    return {"name": name, "xslice": xslice, "ks": ks_x, "sensemap": map_x}


def read_tfrecord(filename_tfrecord):
    """Read TFRecord for debugging."""
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    tf_sess = tf.Session(config=session_config)
    data = read_tfrecord_with_sess(tf_sess, filename_tfrecord)
    tf_sess.close()
    return data


def download_dataset_knee(dir_out, dir_tmp="tmp", verbose=False, do_cleanup=True):
    """Download and unzip knee dataset from mridata.org."""
    if not os.path.isdir(dir_out):
        os.makedirs(dir_out)
    if os.path.isdir(dir_tmp):
        print("WARNING! Temporary folder exists (%s)" % dir_tmp)
    else:
        os.makedirs(dir_tmp)

    num_data = 1
    for i in range(num_data):
        if verbose:
            print("Processing data (%d)..." % i)

        url = "http://old.mridata.org/knees/fully_sampled/p%d/e1/s1/P%d.zip" % (
            i + 1,
            i + 1,
        )
        dir_name_i = os.path.join(dir_out, "data%02d" % i)

        if verbose:
            print("  dowloading from %s..." % url)
        if not os.path.isdir(dir_name_i):
            os.makedirs(dir_name_i)
        print(url)
        exit()
        file_download = wget.download(url, out=dir_tmp)

        if verbose:
            print("  unzipping contents to %s..." % dir_name_i)
        with zipfile.ZipFile(file_download, "r") as zip_ref:
            for member in zip_ref.namelist():
                filename = os.path.basename(member)
                if not filename:
                    continue
                file_src = zip_ref.open(member)
                file_dest = open(os.path.join(dir_name_i, filename), "wb")
                with file_src, file_dest:
                    shutil.copyfileobj(file_src, file_dest)

    if do_cleanup:
        if verbose:
            print("Cleanup...")
        shutil.rmtree(dir_tmp)

    if verbose:
        print("Done")


def create_masks(
        dir_out,
        shape_y=320,
        shape_z=256,
        verbose=False,
        acc_y=(1, 2, 3),
        acc_z=(1, 2, 3),
        shape_calib=1,
        variable_density=True,
        num_repeat=4):
    """Create sampling masks using BART."""
    flags = ""
    file_fmt = "mask_%0.1fx%0.1f_c%d_%02d"
    if variable_density:
        flags = flags + " -v "
        file_fmt = file_fmt + "_vd"

    if not os.path.exists(dir_out):
        os.mkdir(dir_out)

    for a_y in acc_y:
        for a_z in acc_z:
            if a_y * a_z != 1:
                num_repeat_i = num_repeat
                if (a_y == acc_y[-1]) and (a_z == acc_z[-1]):
                    num_repeat_i = num_repeat_i * 2
                for i in range(num_repeat_i):
                    random_seed = 1e6 * random.random()
                    file_name = file_fmt % (a_y, a_z, shape_calib, i)
                    if verbose:
                        print("creating mask (%s)..." % file_name)
                    file_name = os.path.join(dir_out, file_name)
                    cmd = "%s poisson -C %d -Y %d -Z %d -y %d -z %d -s %d %s %s" % (
                        BIN_BART,
                        shape_calib,
                        shape_y,
                        shape_z,
                        a_y,
                        a_z,
                        random_seed,
                        flags,
                        file_name,
                    )
                    subprocess.check_output(["bash", "-c", cmd])


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
