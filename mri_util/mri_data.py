"""Util for data management."""
import glob
import os
import random

import numpy as np
import tensorflow as tf

from mri_util import cfl, mri_prep, recon, tf_util


def prepare_filenames(dir_name, search_str="/*.tfrecords", shuffle=True):
    print("dirname", dir_name)
    """Find and return filenames."""
    if not tf.gfile.Exists(dir_name) or not tf.gfile.IsDirectory(dir_name):
        print("data files not found")
        exit

    full_path = os.path.join(dir_name)
    case_list = sorted(glob.glob(full_path + search_str))

    if shuffle is True:
        random.shuffle(case_list)

    return case_list


def load_masks_cfl(filenames, image_shape=None):
    """Read masks from files."""
    if image_shape is None:
        # First find masks shape...
        image_shape = [0, 0]
        for f in filenames:
            f_cfl = os.path.splitext(f)[0]
            mask = np.squeeze(cfl.read(f_cfl))
            shape_z = mask.shape[-2]
            shape_y = mask.shape[-1]
            if image_shape[-2] < shape_z:
                image_shape[-2] = shape_z
            if image_shape[-1] < shape_y:
                image_shape[-1] = shape_y

    masks = np.zeros([len(filenames)] + image_shape, dtype=np.complex64)

    i_file = 0
    for f in filenames:
        f_cfl = os.path.splitext(f)[0]
        tmp = np.squeeze(cfl.read(f_cfl))
        tmp = recon.zeropad(tmp, image_shape)
        masks[i_file, :, :] = tmp
        i_file = i_file + 1

    return masks


def prep_tfrecord(
    example,
    masks,
    out_shape=[80, 180],
    num_channels=6,
    num_emaps=2,
    random_seed=0,
    verbose=False,
    data_type="DCE",
):
    """Prepare tfrecord for training"""
    name = "prep_tfrecord"
    features = {}

    if data_type is "DCE_2D":
        casename, _, ks_x, map_x = mri_prep.process_DCE_2D(
            example, num_channels=num_channels, num_emaps=num_emaps
        )
        # kz, ky, channels
        ks_x = tf.transpose(ks_x, [1, 2, 0])
        map_x = tf.transpose(map_x, [1, 2, 0])

        ks_x = tf.image.flip_left_right(ks_x)
        map_x = tf.image.flip_left_right(map_x)

        # Initially set image size to be all the same
        # ks_x = tf.image.resize_image_with_crop_or_pad(
        #     ks_x, out_shape[0], out_shape[1])

        mask_recon = tf.abs(ks_x) / tf.reduce_max(tf.abs(ks_x))
        mask_recon = tf.cast(mask_recon > 1e-7, dtype=tf.complex64)

        ks_x = ks_x * mask_recon

        # placeholder for truth for DCE since we don't have one
        features["ks_truth"] = ks_x

        # map_x = tf.image.resize_image_with_crop_or_pad(map_x,
        #                                                out_shape[0],
        #                                                out_shape[1])
        map_x = tf.reshape(
            map_x, [out_shape[0], out_shape[1], num_emaps, num_channels])

        features["ks_input"] = ks_x
        features["sensemap"] = map_x
        features["mask_recon"] = mask_recon

    if data_type is "DCE":
        casename, _, ks_x, map_x = mri_prep.process_DCE(
            example, num_channels=num_channels, num_emaps=num_emaps
        )
        # kz, ky, time, channels
        # for 2D plus time
        map_x = tf.expand_dims(map_x, -1)
        ks_x = tf.transpose(ks_x, [1, 2, 3, 0])
        map_x = tf.transpose(map_x, [1, 2, 3, 0])

        ks_x = tf.image.flip_left_right(ks_x)
        map_x = tf.image.flip_left_right(map_x)

        mask_recon = tf.abs(ks_x) / tf.reduce_max(tf.abs(ks_x))
        mask_recon = tf.cast(mask_recon > 1e-7, dtype=tf.complex64)

        ks_x = ks_x * mask_recon

        # placeholder for truth for DCE since we don't have one
        features["ks_truth"] = ks_x

        map_x = tf.reshape(
            map_x, [out_shape[0], out_shape[1], num_emaps, 1, num_channels]
        )

        features["ks_input"] = ks_x
        features["sensemap"] = map_x
        features["mask_recon"] = mask_recon

    if data_type is "knee":
        casename, _, ks_x, map_x = mri_prep.process_tfrecord(
            example, num_channels=num_channels, num_emaps=num_emaps
        )

        # Randomly select mask
        mask_x = tf.constant(masks, dtype=tf.complex64)
        mask_x = tf.random_shuffle(mask_x)
        mask_x = tf.slice(mask_x, [0, 0, 0], [1, -1, -1])
        # Augment sampling masks

        # print(mask_x)
        # exit()
        mask_x = tf.image.random_flip_up_down(mask_x, seed=random_seed)
        mask_x = tf.image.random_flip_left_right(mask_x, seed=random_seed)

        # Tranpose to store data as (kz, ky, channels)
        mask_x = tf.transpose(mask_x, [1, 2, 0])
        ks_x = tf.transpose(ks_x, [1, 2, 0])
        map_x = tf.transpose(map_x, [1, 2, 0])

        ks_x = tf.image.flip_up_down(ks_x)
        map_x = tf.image.flip_up_down(map_x)

        # Initially set image size to be all the same
        ks_x = tf.image.resize_image_with_crop_or_pad(
            ks_x, out_shape[0], out_shape[1])
        mask_x = tf.image.resize_image_with_crop_or_pad(
            mask_x, out_shape[0], out_shape[1]
        )

        shape_cal = 5
        if shape_cal > 0:
            with tf.name_scope("CalibRegion"):
                if verbose:
                    print(
                        "%s>  Including calib region (%d, %d)..."
                        % (name, shape_cal, shape_cal)
                    )
                mask_calib = tf.ones(
                    [shape_cal, shape_cal, 1], dtype=tf.complex64)
                mask_calib = tf.image.resize_image_with_crop_or_pad(
                    mask_calib, out_shape[0], out_shape[1]
                )
                mask_x = mask_x * (1 - mask_calib) + mask_calib

            mask_recon = tf.abs(ks_x) / tf.reduce_max(tf.abs(ks_x))
            mask_recon = tf.cast(mask_recon > 1e-7, dtype=tf.complex64)
            mask_x = mask_x * mask_recon

            # Assuming calibration region is fully sampled
            shape_sc = 5
            scale = tf.image.resize_image_with_crop_or_pad(
                ks_x, shape_sc, shape_sc)
            scale = tf.reduce_mean(tf.square(tf.abs(scale))) * (
                shape_sc * shape_sc / 1e5
            )
            scale = tf.cast(1.0 / tf.sqrt(scale), dtype=tf.complex64)
            ks_x = ks_x * scale

            # Make sure size is correct
            map_shape = tf.shape(map_x)
            map_shape_z = tf.slice(map_shape, [0], [1])
            map_shape_y = tf.slice(map_shape, [1], [1])
            assert_z = tf.assert_equal(out_shape[0], map_shape_z)
            assert_y = tf.assert_equal(out_shape[1], map_shape_y)
            with tf.control_dependencies([assert_z, assert_y]):
                map_x = tf.identity(map_x, name="sensemap_size_check")
            map_x = tf.image.resize_image_with_crop_or_pad(
                map_x, out_shape[0], out_shape[1]
            )
            map_x = tf.reshape(
                map_x, [out_shape[0], out_shape[1], num_emaps, num_channels]
            )

            # Ground truth
            ks_truth = ks_x

            # Masked input
            ks_x = tf.multiply(ks_x, mask_x)

        features["name"] = casename
        features["ks_input"] = ks_x
        features["sensemap"] = map_x
        features["mask_recon"] = mask_recon
        features["scale"] = scale
        features["ks_truth"] = ks_truth

    if data_type is "new_mfast":
        print("new_mfast")
        casename, _, ks_x, map_x = mri_prep.process_tfrecord(
            example, num_channels=num_channels, num_emaps=num_emaps
        )
        # kz, ky, channels
        ks_x = tf.transpose(ks_x, [1, 2, 0])
        map_x = tf.transpose(map_x, [1, 2, 0])

        ks_x = tf.image.flip_up_down(ks_x)
        map_x = tf.image.flip_up_down(map_x)

        # Initially set image size to be all the same
        ks_x = tf.image.resize_image_with_crop_or_pad(
            ks_x, out_shape[0], out_shape[1])

        mask_recon = tf.abs(ks_x) / tf.reduce_max(tf.abs(ks_x))
        mask_recon = tf.cast(mask_recon > 1e-7, dtype=tf.complex64)

        ks_x = ks_x * mask_recon

        # Make sure size is correct
        # map_shape = tf.shape(map_x)
        # map_shape_z = tf.slice(map_shape, [0], [1])
        # map_shape_y = tf.slice(map_shape, [1], [1])
        # assert_z = tf.assert_equal(out_shape[0], map_shape_z)
        # assert_y = tf.assert_equal(out_shape[1], map_shape_y)
        # with tf.control_dependencies([assert_z, assert_y]):
        #     map_x = tf.identity(map_x, name="sensemap_size_check")
        map_x = tf.image.resize_image_with_crop_or_pad(
            map_x, out_shape[0], out_shape[1]
        )
        map_x = tf.reshape(
            map_x, [out_shape[0], out_shape[1], num_emaps, num_channels])

        # placeholder for truth for MFAST since we don't have one
        features["ks_truth"] = ks_x
        features["ks_input"] = ks_x
        features["sensemap"] = map_x
        features["mask_recon"] = mask_recon

    return features


def create_dataset(
    train_data_dir,
    mask_data_dir,
    batch_size=16,
    buffer_size=10,
    out_shape=[80, 180],
    num_channels=6,
    num_maps=2,
    verbose=True,
    random_seed=0,
    name="create_dataset",
    data_type="DCE",
    shuffle=True,
    repeat=-1,
    search_str=None,
):
    """Setups input tensors."""
    train_filenames_tfrecord = prepare_filenames(
        train_data_dir, search_str=search_str, shuffle=shuffle
    )

    q = []
    for file in train_filenames_tfrecord:
        q.append(file)
        q.append(file)
    train_filenames_tfrecord = q

    mask_filenames_cfl = prepare_filenames(mask_data_dir, search_str="/*.cfl")
    if verbose:
        print(
            "%s> Number of training files (%s): %d"
            % (name, train_data_dir, len(train_filenames_tfrecord))
        )
        print(
            "%s> Number of mask files (%s): %d"
            % (name, mask_data_dir, len(mask_filenames_cfl))
        )

    masks = load_masks_cfl(mask_filenames_cfl)

    with tf.variable_scope(name):
        dataset = tf.data.TFRecordDataset(train_filenames_tfrecord)

        def _prep_tfrecord_with_param(example):
            return prep_tfrecord(
                example,
                masks,
                out_shape=out_shape,
                num_channels=num_channels,
                num_emaps=num_maps,
                random_seed=random_seed,
                verbose=verbose,
                data_type=data_type,
            )

        dataset = dataset.map(_prep_tfrecord_with_param)
        dataset = dataset.prefetch(batch_size * buffer_size)
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat(-1)

    return dataset, len(train_filenames_tfrecord)
