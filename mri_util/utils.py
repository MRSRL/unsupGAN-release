
import os
import sys

import tensorflow as tf

from mri_util import mri_data, tf_util


class Iterator(object):
    def __init__(
        self,
        batch_size,
        mask_path,
        data_type,
        dirname,
        out_shape,
        data_dir=None,
        shuffle=True,
        verbose=False,
        train_acc=None,
        search_str="/*.tfrecords",
    ):
        if dirname == "test":
            shuffle = False
        else:
            shuffle = True

        # Mask directory
        if train_acc is None:
            self.mask_path = data_dir + "/masks"
        else:
            self.mask_path = "/home_local/ekcole/knee_masks_%d" % train_acc
        if dirname is "test":
            self.mask_path = mask_path

        print("Masks directory:", self.mask_path)

        # Datasets directory (knee scans)
        if data_dir is None:
            dataset_dir = "/home_local/ekcole/knee_data"
        else:
            dataset_dir = data_dir + "/data"

        print("Datasets directory:", dataset_dir)

        num_coils = 8  # coils
        num_emaps = 1  # sensitivity maps
        self.output_height = 256  # height of images
        self.output_width = 320  # width of images
        out_shape = [self.output_height, self.output_width]

        if not tf.gfile.Exists(mask_path) or not tf.gfile.IsDirectory(mask_path):
            print("mask path not found")
            exit

        mask_filenames_cfl = mri_data.prepare_filenames(
            self.mask_path, search_str="/*.cfl"
        )
        loaded_masks = mri_data.load_masks_cfl(mask_filenames_cfl)
        self.masks = tf.constant(loaded_masks, dtype=tf.complex64)

        self.dataset, self.num_files = mri_data.create_dataset(
            os.path.join(dataset_dir, dirname),
            self.mask_path,
            num_channels=num_coils,
            num_maps=num_emaps,
            batch_size=batch_size,
            buffer_size=2,
            out_shape=out_shape,
            data_type=data_type,
            shuffle=shuffle,
            verbose=False,
            search_str=search_str
        )

        self.iterator = self.dataset.make_one_shot_iterator()

    def get_truth_image(self, features):
        ks_truth = features["ks_truth"]
        sensemap = features["sensemap"]
        mask_recon = features["mask_recon"]

        image_truth = tf_util.model_transpose(ks_truth * mask_recon, sensemap)
        image_truth = tf.identity(image_truth, name="truth_image")

        # complex to channels
        image_truth = tf_util.complex_to_channels(image_truth)

        return image_truth

    def get_lowres_mask(self, hparams):
        total_masks = None
        for i in range(hparams.batch_size):
            # Randomly select mask
            mask_x = tf.random_shuffle(self.masks)
            mask_x = tf.slice(mask_x, [0, 0, 0], [1, -1, -1])
            if total_masks is not None:
                total_masks = tf.concat([total_masks, mask_x], 0)
            else:
                total_masks = mask_x
        total_masks = tf.complex(tf.real(total_masks), tf.real(total_masks))
        total_masks = tf.expand_dims(total_masks, -1)
        total_masks = tf.tile(total_masks, [1, 1, 1, hparams.num_coils])
        return total_masks

    def get_undersampled(self, features):
        ks_input = features["ks_input"]
        sensemap = features["sensemap"]
        im_lowres = tf_util.model_transpose(ks_input, sensemap)
        im_lowres = tf.identity(im_lowres, name="low_res_image")
        im_lowres = tf_util.complex_to_channels(im_lowres)
        return im_lowres

    def get_images(self, features):
        ks_input = features["ks_input"]
        sensemap = features["sensemap"]
        mask = tf_util.kspace_mask(ks_input, dtype=tf.complex64)

        if mask is None:
            mask = tf_util.kspace_mask(ks_input, dtype=tf.complex64)

        ks_input = mask * ks_input

        im_lowres = tf_util.model_transpose(ks_input, sensemap)
        im_lowres = tf.identity(im_lowres, name="low_res_image")

        # complex to channels
        im_lowres = tf_util.complex_to_channels(im_lowres)

        ks_truth = features["ks_truth"]
        mask_recon = features["mask_recon"]

        im_truth = tf_util.model_transpose(ks_truth * mask_recon, sensemap)
        im_truth = tf.identity(im_truth, name="truth_image")
        im_truth = tf_util.complex_to_channels(im_truth)

        return im_lowres, im_truth

    def get_everything(self, sess, features):
        ks_input = features["ks_input"]
        sensemap = features["sensemap"]
        mask = tf_util.kspace_mask(ks_input, dtype=tf.complex64)

        if mask is None:
            mask = tf_util.kspace_mask(ks_input, dtype=tf.complex64)

        ks_input = mask * ks_input

        window = 1
        im_lowres = tf_util.model_transpose(ks_input * window, sensemap)
        # im_lowres = tf_util.ifft2c(ks_input)
        im_lowres = tf.identity(im_lowres, name="low_res_image")

        # complex to channels
        im_lowres = tf_util.complex_to_channels(im_lowres)

        ks_truth = features["ks_truth"]
        mask_recon = features["mask_recon"]

        im_truth = tf_util.model_transpose(ks_truth * mask_recon, sensemap)

        im_truth = tf.identity(im_truth, name="truth_image")
        im_truth = tf_util.complex_to_channels(im_truth)

        im_lowres, im_truth, sensemap, mask = sess.run(
            [im_lowres, im_truth, sensemap, mask]
        )

        return im_lowres, im_truth, sensemap, mask

    def prepare_filenames(dir_name, search_str="/*.tfrecords"):
        """Find and return filenames."""
        if not tf.gfile.Exists(dir_name) or not tf.gfile.IsDirectory(dir_name):
            raise FileNotFoundError("Could not find folder `%s'" % (dir_name))

        full_path = os.path.join(dir_name)
        case_list = glob.glob(full_path + search_str)
        random.shuffle(case_list)

        return case_list
