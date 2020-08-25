import glob
import os
import subprocess

import numpy as np
import scipy.misc
import tensorflow as tf

import imageio
import unrolled
import unrolled_3d
from mri_util import cfl, mask, metrics, mri_data, tf_util
from mri_util import utils as mri_utils

BIN_BART = "bart"


class WGAN(object):
    def __init__(
        self,
        sess,
        do_separable=False,
        batch_size=64,
        g_dim=None,
        d_dim=None,
        res_blocks=4,
        iterations=5,
        c_dim=2,     #number of channels
        log_dir=None,
        max_epoch=50,
        d_steps=1,
        g_steps=1,
        lr=1e-4,
        beta1=0.0,
        beta2=0.9,
        mask_path=None,
        arch="unrolled",
        data_type="knee",
        verbose=True,
        time=False,
        train_acc=None,
        data_dir=None
    ):
        self.data_dir = data_dir
        self.shuffle = False
        self.sess = sess
        self.batch_size = batch_size
        self.c_dim = c_dim
        self.g_dim = g_dim
        self.d_dim = d_dim
        self.res_blocks = res_blocks
        self.iterations = iterations
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.image_dir = os.path.join(self.log_dir, "images")
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)
        gif_dir = os.path.join(self.log_dir, "gifs")
        if not os.path.exists(gif_dir):
            os.makedirs(gif_dir)
        dicom_dir = os.path.join(self.log_dir, "dicoms")
        if not os.path.exists(dicom_dir):
            os.makedirs(dicom_dir)
        self.bart_dir = os.path.join(self.log_dir, "bart_recon")
        if not os.path.exists(self.bart_dir):
            os.makedirs(self.bart_dir)
        self.max_epoch = max_epoch
        self.d_steps = d_steps
        self.g_steps = g_steps
        self.lr = lr  # learning rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.mask_path = mask_path
        self.arch = arch
        self.data_type = data_type
        self.verbose = verbose
        self.do_separable = do_separable
        self.train_acc = train_acc
        if data_type is "knee":
            self.num_coils = 8
            self.height = 256
            self.width = 320
            self.dims = 4
            self.out_shape = [self.height, self.width]
            self.ks_shape = [None, self.height, self.width, self.num_coils]
            self.sense_shape = [None, self.height,
                                self.width, 1, self.num_coils]
            self.image_shape = [self.batch_size, self.height, self.width, 1]
            self.truth_shape = [self.batch_size, self.height, self.width, 2]
            self.max_frames = 0
        if data_type is "DCE" or data_type is "DCE_2D":
            self.num_coils = 6
            self.height = 180
            self.width = 80
            self.max_frames = 0
        if data_type is "new_mfast":
            self.num_coils = 32
            # self.height = 80
            # self.width = 320
            self.height = 70
            self.width = 256
            self.max_frames = 0
        if data_type is "DCE_2D":
            self.max_frames = 18
            print("Separate time frames")
            self.dims = 4
            self.out_shape = [self.height, self.width]
            self.ks_shape = [None, self.height, self.width, self.num_coils]
            self.sense_shape = [None, self.height,
                                self.width, 1, self.num_coils]
            self.image_shape = [self.batch_size, self.height, self.width, 1]
            self.truth_shape = [self.batch_size, self.height, self.width, 2]
        if data_type is "DCE":
            self.max_frames = 18
            self.dims = 5
            self.out_shape = [self.height, self.width, self.max_frames]
            self.ks_shape = [
                None,
                self.height,
                self.width,
                self.max_frames,
                self.num_coils,
            ]
            self.sense_shape = [None, self.height,
                                self.width, 1, 1, self.num_coils]
            self.image_shape = [
                self.batch_size,
                self.height,
                self.width,
                self.max_frames,
                1,
            ]
            self.truth_shape = [
                self.batch_size,
                self.height,
                self.width,
                self.max_frames,
                2,
            ]
        # self.build_model()

    def build_model(self, mode):
        self.Y_real, self.data_num, real_mask = self.read_real()

        # Read in train or test input images to be reconstructed by generator
        train_iterator = mri_utils.Iterator(
            self.batch_size,
            self.mask_path,
            self.data_type,
            mode,  # either train or test
            self.out_shape,
            verbose=self.verbose,
            train_acc=self.train_acc,
            data_dir=self.data_dir
        )
        self.input_files = train_iterator.num_files
        train_dataset = train_iterator.iterator.get_next()
        ks_truth = train_dataset["ks_truth"]
        ks_input = train_dataset["ks_input"]
        sensemap = train_dataset["sensemap"]
        image_truth = tf_util.model_transpose(ks_truth, sensemap)
        self.complex_truth = image_truth

        image_truth = tf_util.complex_to_channels(image_truth)
        self.z_truth = image_truth
        self.ks = ks_input
        self.sensemap = sensemap

        # generate image X_gen
        self.X_gen = self.generator(self.ks, self.sensemap)
        # no measurement for supervised
        self.Y_fake = self.X_gen

        # output of discriminator for fake and real images
        self.d_logits_fake = self.discriminator(self.Y_fake, reuse=False)
        self.d_logits_real = self.discriminator(self.Y_real, reuse=True)

        # discriminator loss
        self.d_loss = tf.reduce_mean(self.d_logits_fake) - tf.reduce_mean(
            self.d_logits_real
        )
        # add total variation loss
        # tv_loss = -tf.reduce_sum(tf.image.total_variation(self.Y_fake))
        # generator loss
        self.g_loss = -tf.reduce_mean(self.d_logits_fake)  # + tv_loss

        # Gradient Penalty
        self.epsilon = tf.random_uniform(
            shape=[self.batch_size, 1, 1, 1], minval=0.0, maxval=1.0
        )
        Y_hat = self.Y_real + self.epsilon * (self.Y_fake - self.Y_real)
        D_Y_hat = self.discriminator(Y_hat, reuse=True)
        grad_D_Y_hat = tf.gradients(D_Y_hat, [Y_hat])[0]
        red_idx = range(1, Y_hat.shape.ndims)
        slopes = tf.sqrt(
            tf.reduce_sum(tf.square(grad_D_Y_hat),
                          reduction_indices=list(red_idx))
        )
        self.gradient_penalty = tf.reduce_mean((slopes - 1.0) ** 2)
        # updated discriminator loss
        self.d_loss = self.d_loss + 10.0 * self.gradient_penalty

        train_vars = tf.trainable_variables()
        for v in train_vars:
            # v = tf.cast(v, tf.float32)
            tf.add_to_collection("reg_loss", tf.nn.l2_loss(v))
        self.generator_vars = [v for v in train_vars if "generator" in v.name]
        self.discriminator_vars = [
            v for v in train_vars if "discriminator" in v.name]

        self.g_optimizer = tf.train.AdamOptimizer(
            learning_rate=self.lr, name="g_opt", beta1=self.beta1, beta2=self.beta2
        ).minimize(self.g_loss, var_list=self.generator_vars)
        self.d_optimizer = tf.train.AdamOptimizer(
            learning_rate=self.lr, name="d_opt", beta1=self.beta1, beta2=self.beta2
        ).minimize(self.d_loss, var_list=self.discriminator_vars)

        self.output_image = tf_util.channels_to_complex(self.X_gen)
        self.im_out = self.output_image
        self.mag_output = tf.abs(self.output_image)
        self.create_summary()

        with tf.variable_scope("counter"):
            self.counter = tf.get_variable(
                "counter",
                shape=[1],
                initializer=tf.constant_initializer([1]),
                dtype=tf.int32,
            )
            self.update_counter = tf.assign(
                self.counter, tf.add(self.counter, 1))
        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter(
            self.log_dir, self.sess.graph)
        self.initialize_model()

    def train(self):
        batch_epoch = self.data_num // (self.batch_size *
                                        self.d_steps * self.g_steps)
        max_iterations = int(self.max_epoch * batch_epoch)

        # calculate number of parameters in model
        total_parameters = 0
        for variable in tf.trainable_variables():
            variable_parameters = 1
            for dim in variable.get_shape():
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print("Total number of trainable parameters: %d" % total_parameters)
        tf.summary.scalar("parameters/parameters", total_parameters)

        print("[*] Start from step %d." % (self.sess.run(self.counter)))
        print("max iterations", max_iterations)

        begin = int(self.sess.run(self.counter))
        for step in range(begin, max_iterations):
            print("step ", step)

            # Discriminator
            for critic_iter in range(self.d_steps):
                _, summary_str = self.sess.run(
                    [self.d_optimizer, self.train_sum])
            # Generator
            for critic_iter in range(self.g_steps):
                _, summary_str = self.sess.run(
                    [self.g_optimizer, self.train_sum])
            if step % 20 == 0:
                #Adds tensorboard summary every 20 steps
                self.summary_writer.add_summary(summary_str, step)
            if step % 100 == 0:
                print("saving a checkpoint")
                self.saver.save(self.sess, self.log_dir + "/model.ckpt")
            self.sess.run(self.update_counter)

        self.saver.save(
            self.sess, self.log_dir + "/model.ckpt", global_step=max_iterations
        )

    def test(self):
        print("testing")
        # read in a correct test dicom file to change it later
        # dicom_filename = get_testdata_files("MR_small.dcm")[0]
        # self.ds = pydicom.dcmread(dicom_filename)

        # 18 time frames in each DICOM
        max_frame = self.max_frames
        frame = 1
        case = 1
        gif = []
        print("number of test cases", self.input_files)

        total_acc = []
        mask_input = tf_util.kspace_mask(self.ks, dtype=tf.complex64)
        numel = tf.cast(tf.size(mask_input), tf.float32)
        acc = numel / tf.reduce_sum(tf.abs(mask_input))

        output_psnr = []
        output_nrmse = []
        output_ssim = []
        cs_psnr = []
        cs_nrmse = []
        cs_ssim = []

        for step in range(self.input_files):
            # whenever you do a lot of TF operations or sess.run(-c) in a loop, cpu memory builds up

            print("test file #", step)

            acc_run = self.sess.run(acc)
            # acc = np.round(acc, decimals=2)
            # print("acceleration", acc_run)
            total_acc.append(acc_run)
            print(
                "total test acc:",
                np.round(np.mean(total_acc), decimals=2),
                np.round(np.std(total_acc), decimals=2),
            )

            # if (step < select*max_frame) or (step > (select+1)*max_frame):
            #     continue
            if self.data_type is "knee":
                # l1 = 0.015
                l1 = 0.02
            if self.data_type is "DCE":
                l1 = 0.05
            if self.data_type is "DCE_2D":
                l1 = 0.07

            # bart_test = self.bart_cs(sample_ks, sample_sensemap, l1=l1)
            # im_in = tf_util.model_transpose(self.ks, self.sensemap)
            # output_image, input_image, complex_truth = self.sess.run([self.im_out, im_in, self.complex_truth])
            output_image, complex_truth = self.sess.run(
                [self.im_out, self.complex_truth]
            )
            if self.data_type is "knee":
                # input_image = np.squeeze(input_image)
                output_image = np.squeeze(output_image)
                truth_image = np.squeeze(complex_truth)

                psnr, nrmse, ssim = metrics.compute_all(
                    truth_image, output_image, sos_axis=-1
                )
                # psnr, nrmse, ssim = metrics.compute_all(
                #     truth_image, input_image, sos_axis=-1
                # )

                # psnr = np.round(psnr, decimals=2)
                # nrmse = np.round(nrmse, decimals=2)
                # ssim = np.round(ssim, decimals=2)

                output_psnr.append(psnr)
                output_nrmse.append(nrmse)
                output_ssim.append(ssim)

                print("output psnr, nrmse, ssim")
                print(
                    np.round(np.mean(output_psnr), decimals=2),
                    np.round(np.mean(output_nrmse), decimals=2),
                    np.round(np.mean(output_ssim), decimals=2),
                )

                # psnr, nrmse, ssim = metrics.compute_all(
                #     complex_truth, bart_test, sos_axis=-1
                # )

                # psnr = np.round(psnr, decimals=2)
                # nrmse = np.round(nrmse, decimals=2)
                # ssim = np.round(ssim, decimals=2)

                # cs_psnr.append(psnr)
                # cs_nrmse.append(nrmse)
                # cs_ssim.append(ssim)
                # print("cs psnr, nrmse, ssim")
                # print(
                #     np.mean(psnr), np.mean(nrmse), np.mean(ssim),
                # )

            # mag_input = np.squeeze(np.absolute(input_image))
            # mag_cs = np.squeeze(np.absolute(bart_test))
            # mag_output = np.squeeze(np.absolute(output_image))

            # if self.data_type is "knee":
            #     # save as png
            #     mag_input = np.rot90(mag_input, k=3)
            #     # norm_max = np.amax(mag_input)
            #     mag_cs = np.rot90(mag_cs, k=3)
            #     mag_output = np.rot90(mag_output, k=3)

            #     mag_images = np.concatenate((mag_input, mag_cs, mag_output), axis=1)
            #     filename = self.image_dir + "/mag_" + str(step) + ".png"
            #     scipy.misc.imsave(filename, mag_images)
            if self.data_type is "DCE":
                gif = []
                for f in range(max_frame):
                    frame_input = mag_input[:, :, f]
                    frame_output = mag_output[:, :, f]
                    frame_cs = mag_cs[:, :, f]
                    rotated_input = np.rot90(frame_input, 1)
                    rotated_output = np.rot90(frame_output, 1)
                    rotated_cs = np.rot90(frame_cs, 1)

                    # normalize to help the CS brightness
                    newMax = np.max(rotated_input)
                    newMin = np.min(rotated_input)
                    oldMax = np.max(rotated_cs)
                    oldMin = np.min(rotated_cs)
                    rotated_cs = (rotated_cs - oldMin) * (newMax - newMin) / (
                        oldMax - oldMin
                    ) + newMin
                    full_images = np.concatenate(
                        (rotated_input, rotated_output, rotated_cs), axis=1
                    )
                    #                 filename = self.log_dir + '/images/case' + str(step) + '_f' + str(f) + '.png'
                    #                 scipy.misc.imsave(filename, full_images)
                    new_filename = (
                        self.log_dir + "/dicoms/" +
                        str(step) + "_f" + str(f) + ".dcm"
                    )
                    self.write_dicom(full_images, new_filename, step, f)

                    # add each frame to a gif
                    gif.append(full_images)

                print("Saving gif")
                gif_path = self.log_dir + "/gifs/slice_" + str(step) + ".gif"
                imageio.mimsave(gif_path, gif, duration=0.2)

            # Save as PNG
            # filename = self.log_dir + '/images/case' + str(case) + '_f' + str(frame) + '.png'
            # scipy.misc.imsave(filename, saved)
            # Save as DICOM
            if self.data_type is "DCE_2D":
                rotated_input = np.rot90(mag_input, 1)
                rotated_output = np.rot90(mag_output, 1)
                rotated_cs = np.rot90(mag_cs, 1)

                full_images = np.concatenate(
                    (rotated_input, rotated_output, rotated_cs), axis=1
                )
                full_images = full_images * 10.0
                new_filename = (
                    self.log_dir + "/dicoms/" +
                    str(case) + "_f" + str(frame) + ".dcm"
                )
                dicom_output = np.squeeze(np.abs(full_images))
                self.write_dicom(dicom_output, new_filename, case, frame)

                # Save as PNG
                filename = (
                    self.log_dir
                    + "/images/case"
                    + str(case)
                    + "_f"
                    + str(frame)
                    + ".png"
                )
                scipy.misc.imsave(filename, full_images)

                if frame <= max_frame:
                    gif.append(full_images)
                    frame = frame + 1
                if frame > max_frame:
                    print("Max frame")
                    gif.append(full_images)
                    # gif = gif+100
                    # gif = gif.astype('uint8')
                    # timeMax = np.max(gif, axis=-1)
                    # gif = gif/np.max(gif)
                    # if self.shuffle == "False":
                    print("Saving gif")
                    gif_path = self.log_dir + \
                        "/gifs/new_gif" + str(case) + ".gif"
                    imageio.mimsave(gif_path, gif, "GIF", duration=0.2)

                    # return back to next case
                    frame = 1
                    case = case + 1
                    gif = []

                    # Create a gif
                    if frame <= max_frame:
                        mypath = self.log_dir + "/images/case" + str(case)
                        search_str = "*.png"
                        filenames = sorted(glob.glob(mypath + search_str))
                        # make and save gif
                        gif_path = self.log_dir + \
                            "/gifs/case_" + str(case) + ".gif"

                        images = []
                        for f in filenames:
                            image = scipy.misc.imread(f)
                            images.append(image)
                        filename_gif = mypath + "/case.gif"
                        imageio.mimsave(gif_path, images, duration=0.3)

        txt_path = os.path.join(self.log_dir, "output_metrics.txt")
        f = open(txt_path, "w")
        f.write(
            "output psnr = "
            + str(np.mean(output_psnr))
            + " +\- "
            + str(np.std(output_psnr))
            + "\n"
            + "output nrmse = "
            + str(np.mean(output_nrmse))
            + " +\- "
            + str(np.std(output_nrmse))
            + "\n"
            + "output ssim = "
            + str(np.mean(output_ssim))
            + " +\- "
            + str(np.std(output_ssim))
            + "\n"
            + "test acc = "
            + str(np.mean(total_acc))
            + " +\-"
            + str(np.std(total_acc))
        )
        f.close()
        # txt_path = os.path.join(self.log_dir, "cs_metrics.txt")
        # f = open(txt_path, "w")
        # f.write(
        #     "cs psnr = "
        #     + str(np.mean(cs_psnr))
        #     + " +\- "
        #     + str(np.std(cs_psnr))
        #     + "\n"
        #     + "output nrmse = "
        #     + str(np.mean(cs_nrmse))
        #     + " +\- "
        #     + str(np.std(cs_nrmse))
        #     + "\n"
        #     + "output ssim = "
        #     + str(np.mean(cs_ssim))
        #     + " +\- "
        #     + str(np.std(cs_ssim))
        # )
        # f.close()

    def generator(self, ks_input, sensemap, reuse=False):
        mask_example = tf_util.kspace_mask(ks_input, dtype=tf.complex64)
        with tf.variable_scope("generator") as scope:
            if reuse:
                scope.reuse_variables()
            # 2D data
            # batch, height, width, channels
            if self.dims == 4:
                if self.arch == "unrolled":
                    c_out = unrolled.unroll_fista(
                        ks_input,
                        sensemap,
                        num_grad_steps=self.iterations,
                        resblock_num_features=self.g_dim,
                        resblock_num_blocks=self.res_blocks,
                        is_training=True,
                        scope="MRI",
                        mask_output=1,
                        window=None,
                        do_hardproj=True,
                        num_summary_image=6,
                        mask=mask_example,
                        verbose=False,
                    )
                    c_out = tf_util.complex_to_channels(c_out)
                else:
                    z = tf_util.model_transpose(ks_input, sensemap)
                    z = tf_util.complex_to_channels(z)
                    res_size = self.g_dim
                    kernel_size = 3
                    num_channels = 2
                    # could try tf.nn.tanh instead
                    # act = tf.nn.sigmoid
                    num_blocks = 5
                    c = tf.layers.conv2d(
                        z,
                        res_size,
                        kernel_size,
                        padding="same",
                        activation=tf.nn.relu,
                        use_bias=True,
                    )
                    for i in range(num_blocks):
                        c = tf.layers.conv2d(
                            c,
                            res_size,
                            kernel_size,
                            padding="same",
                            activation=tf.nn.relu,
                            use_bias=True,
                        )
                        c1 = tf.layers.conv2d(
                            c,
                            res_size,
                            kernel_size,
                            padding="same",
                            activation=tf.nn.relu,
                            use_bias=True,
                        )
                        c = tf.add(c, c1)
                    c8 = tf.layers.conv2d(
                        c,
                        num_channels,
                        kernel_size,
                        padding="same",
                        activation=None,
                        use_bias=True,
                    )
                    c_out = tf.add(c8, z)
                    c_out = tf.nn.tanh(c_out)

            # 3D data
            # batch, height, width, channels, time frames
            else:
                if self.arch == "unrolled":
                    c_out = unrolled_3d.unroll_fista(
                        ks_input,
                        sensemap,
                        num_grad_steps=self.iterations,
                        num_features=self.g_dim,
                        num_resblocks=self.res_blocks,
                        is_training=True,
                        scope="MRI",
                        mask_output=1,
                        window=None,
                        do_hardproj=True,
                        mask=mask_example,
                        verbose=False,
                        data_format="channels_last",
                        do_separable=self.do_separable,
                    )
                    c_out = tf_util.complex_to_channels(c_out)
            return c_out

    def discriminator(self, input_image, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            res_size = self.d_dim
            kernel_size = 3
            num_channels = 2
            act = tf.nn.leaky_relu
            # act=tf.nn.relu
            num_blocks = 8

            if self.dims == 4:
                c = tf.layers.conv2d(
                    input_image,
                    res_size,
                    kernel_size,
                    padding="same",
                    activation=act,
                    use_bias=True,
                )
                for i in range(num_blocks):
                    c = tf.layers.conv2d(
                        c,
                        res_size,
                        kernel_size,
                        padding="same",
                        activation=act,
                        use_bias=True,
                    )
                    c1 = tf.layers.conv2d(
                        c,
                        res_size,
                        kernel_size,
                        padding="same",
                        activation=act,
                        use_bias=True,
                    )

                    # c = tf.add(c, c1)
                    c = c1
            else:
                c = tf.layers.conv3d(
                    input_image,
                    res_size,
                    kernel_size,
                    padding="same",
                    activation=act,
                    use_bias=True,
                )
                for i in range(num_blocks):
                    c = tf.layers.conv3d(
                        c,
                        res_size,
                        kernel_size,
                        padding="same",
                        activation=act,
                        use_bias=True,
                    )
                    c1 = tf.layers.conv3d(
                        c,
                        res_size,
                        kernel_size,
                        padding="same",
                        activation=act,
                        use_bias=True,
                    )

                    # c = tf.add(c, c1)
                    c = c1

            c = tf.add(c, c1)
            c8 = tf.layers.dense(c, 1, use_bias=True)
            return c8

    def create_summary(self):
        # note that ks is based on the input data not on the rotated data
        output_ks = tf_util.model_forward(self.output_image, self.sensemap)

        # Input image to generator
        self.input_image = tf_util.model_transpose(self.ks, self.sensemap)

        if self.data_type is "knee":
            truth_image = tf_util.channels_to_complex(self.z_truth)

            sum_input = tf.image.flip_up_down(
                tf.image.rot90(tf.abs(self.input_image)))
            sum_output = tf.image.flip_up_down(
                tf.image.rot90(tf.abs(self.output_image))
            )
            sum_truth = tf.image.flip_up_down(
                tf.image.rot90(tf.abs(truth_image)))

            train_out = tf.concat((sum_input, sum_output, sum_truth), axis=2)
            tf.summary.image("input-output-truth", train_out)

            mask_input = tf_util.kspace_mask(self.ks, dtype=tf.complex64)

            loss_l1 = tf.reduce_mean(tf.abs(self.X_gen - self.z_truth))
            loss_l2 = tf.reduce_mean(
                tf.square(tf.abs(self.X_gen - self.z_truth)))
            tf.summary.scalar("l1", loss_l1)
            tf.summary.scalar("l2", loss_l2)

            # to check supervised/unsupervised
            y_real = tf_util.channels_to_complex(self.Y_real)
            y_real = tf.image.flip_up_down(
                tf.image.rot90(tf.abs(y_real)))
            tf.summary.image("y_real/mag", y_real)

        # Plot losses
        self.d_loss_sum = tf.summary.scalar("Discriminator_loss", self.d_loss)
        self.g_loss_sum = tf.summary.scalar("Generator_loss", self.g_loss)
        self.gp_sum = tf.summary.scalar(
            "Gradient_penalty", self.gradient_penalty)
        self.d_fake = tf.summary.scalar(
            "subloss/D_fake", tf.reduce_mean(self.d_logits_fake)
        )
        self.d_real = tf.summary.scalar(
            "subloss/D_real", tf.reduce_mean(self.d_logits_real)
        )
        self.z_sum = tf.summary.histogram(
            "z", tf_util.complex_to_channels(self.input_image)
        )
        self.d_sum = tf.summary.merge(
            [self.z_sum, self.d_loss_sum, self.d_fake, self.d_real]
        )
        self.g_sum = tf.summary.merge([self.z_sum, self.g_loss_sum])
        self.train_sum = tf.summary.merge_all()

    def calculate_metrics(self, output_image, bart_test, sample_truth):
        cs_psnr = []
        cs_nrmse = []
        cs_ssim = []
        output_psnr = []
        output_nrmse = []
        output_ssim = []

        complex_truth = tf_util.channels_to_complex(sample_truth)
        complex_truth = self.sess.run(complex_truth)

        psnr, nrmse, ssim = metrics.compute_all(
            complex_truth, bart_test, sos_axis=-1)
        cs_psnr.append(psnr)
        cs_nrmse.append(nrmse)
        cs_ssim.append(ssim)

        psnr, nrmse, ssim = metrics.compute_all(
            complex_truth, output_image, sos_axis=-1
        )
        output_psnr.append(psnr)
        output_nrmse.append(nrmse)
        output_ssim.append(ssim)
        return output_psnr, output_nrmse, output_ssim

    def read_real(self):
        # Read in "real" undersampled images
        real_iterator = mri_utils.Iterator(
            self.batch_size,
            self.mask_path,
            self.data_type,
            "validate",
            self.out_shape,
            verbose=self.verbose,
            data_dir=self.data_dir
        )
        data_num = real_iterator.num_files
        real_dataset = real_iterator.iterator.get_next()

        img_real = real_iterator.get_truth_image(real_dataset)
        self.masks = real_iterator.masks

        ks = real_dataset["ks_input"]
        real_mask = tf_util.kspace_mask(ks, dtype=tf.complex64)

        return img_real, data_num, real_mask

    def initialize_model(self):
        print("[*] initializing network...")
        if not self.load(self.log_dir):
            self.sess.run(tf.global_variables_initializer())
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(self.sess, self.coord)

    def write_dicom(self, dicom_output, filename, case, index):
        # normalize to uint16
        # horos is 16 bit
        # dicom images must be stored as int
        # normalize between 100 and 1000
        newMax = 1000
        newMin = 100
        oldMax = np.max(dicom_output)
        oldMin = np.min(dicom_output)
        dicom_output = (dicom_output - oldMin) * (newMax - newMin) / (
            oldMax - oldMin
        ) + newMin
        dicom_output = dicom_output.astype(np.uint16)

        self.ds.BitsAllocated = 16
        self.ds.BitsStored = 16
        self.ds.HighBit = self.ds.BitsStored - 1
        self.ds.SamplesPerPixel = 1
        self.ds.SmallestImagePixelValue = dicom_output.min()
        self.ds.LargestImagePixelValue = dicom_output.max()
        self.ds.PixelRepresentation = 0
        self.ds.PatientName = "Case" + str(case) + "Frame" + str(index)

        self.ds.PixelData = dicom_output.tobytes()
        self.ds.Rows, self.ds.Columns = dicom_output.shape

        self.ds.save_as(filename)
        return

    def bart_cs(self, ks, sensemap, l1=0.01):
        cfl_ks = np.squeeze(ks)
        cfl_ks = np.expand_dims(cfl_ks, -2)
        cfl_sensemap = np.squeeze(sensemap)
        cfl_sensemap = np.expand_dims(cfl_sensemap, -2)

        ks_dir = os.path.join(self.bart_dir, "file_ks")
        sense_dir = os.path.join(self.bart_dir, "file_sensemap")
        img_dir = os.path.join(self.bart_dir, "file_img")

        cfl.write(ks_dir, cfl_ks, "R")
        cfl.write(sense_dir, cfl_sensemap, "R")

        # L1-wavelet regularized
        # cmd_flags = "-S -e -R W:3:0:%f -i 100" % l1
        cmd_flags = "-S -e -R W:0:0:%f -i 100" % l1
        # Low-rank
        # might be 3:3
        # cmd_flags = "-S -e -R L:7:7:%f -i 100" % l1
        cmd = "%s pics %s %s %s %s" % (
            BIN_BART, cmd_flags, ks_dir, sense_dir, img_dir,)
        subprocess.check_call(["bash", "-c", cmd])
        bart_recon = self.load_recon(img_dir, sense_dir)
        return bart_recon

    def load_recon(self, file, file_sensemap):
        bart_recon = np.squeeze(cfl.read(file))
        # 18, 80, 180
        if bart_recon.ndim == 2:
            bart_recon = np.transpose(bart_recon, [1, 0])
            bart_recon = np.expand_dims(bart_recon, axis=0)
            bart_recon = np.expand_dims(bart_recon, axis=-1)
        if bart_recon.ndim == 3:
            bart_recon = np.transpose(bart_recon, [2, 1, 0])
            bart_recon = np.expand_dims(bart_recon, axis=-1)
        return bart_recon

    def measure(self, X_gen, sensemap, real_mask):
        name = "measure"
        random_seed = 0
        verbose = True
        image = tf_util.channels_to_complex(X_gen)
        kspace = tf_util.model_forward(image, sensemap)
        # input_shape = tf.shape(kspace)

        total_kspace = None
        if (
            self.data_type is "DCE"
            or self.data_type is "DCE_2D"
            or self.data_type is "mfast"
        ):
            print("DCE measure")
            for i in range(self.batch_size):
                if self.dims == 4:
                    ks_x = kspace[i, :, :]
                else:
                    # 2D plus time
                    ks_x = kspace[i, :, :, :]
                # lazy: use original applied mask and just apply it again
                # won't work because it isn't doing anything unless it gets flipped
                # mask_x = tf_util.kspace_mask(ks_x, dtype=tf.complex64)
                # # Augment sampling masks

                # # New mask - taken from image B
                # # mask = real_mask
                # mask_x = tf.image.flip_up_down(mask_x)
                # mask_x = tf.image.flip_left_right(mask_x)

                # if self.dims != 4:
                #     mask_x = mask_x[:,:,:,0,:]

                # data dimensions
                shape_y = self.width
                shape_t = self.max_frames
                sim_partial_ky = 0.0

                accs = [1, 6]
                rand_accel = (accs[1] - accs[0]) * \
                    tf.random_uniform([]) + accs[0]
                fn_inputs = [
                    shape_y,
                    shape_t,
                    rand_accel,
                    10,
                    2.0,
                    sim_partial_ky,
                ]  # ny, nt, accel, ncal, vd_degree
                mask_x = tf.py_func(
                    mask.generate_perturbed2dvdkt, fn_inputs, tf.complex64
                )
                ks_x = ks_x * tf.reshape(mask_x, [1, shape_y, shape_t, 1])

                if total_kspace is not None:
                    total_kspace = tf.concat([total_kspace, ks_x], 0)
                else:
                    total_kspace = ks_x
        if self.data_type is "knee":
            print("knee")
            for i in range(self.batch_size):
                ks_x = kspace[i, :, :]
                print("ks_x", ks_x)
                # Randomly select mask
                mask_x = tf.random_shuffle(self.masks)
                print("self masks", mask_x)
                mask_x = tf.slice(mask_x, [0, 0, 0], [1, -1, -1])
                print("sliced masks", mask_x)
                # Augment sampling masks
                mask_x = tf.image.random_flip_up_down(mask_x, seed=random_seed)
                mask_x = tf.image.random_flip_left_right(
                    mask_x, seed=random_seed)
                # Tranpose to store data as (kz, ky, channels)
                mask_x = tf.transpose(mask_x, [1, 2, 0])
                print("transposed mask", mask_x)
                ks_x = tf.image.flip_up_down(ks_x)
                # Initially set image size to be all the same
                ks_x = tf.image.resize_image_with_crop_or_pad(
                    ks_x, self.height, self.width
                )
                mask_x = tf.image.resize_image_with_crop_or_pad(
                    mask_x, self.height, self.width
                )
                print("resized mask", mask_x)
                shape_cal = 20
                if shape_cal > 0:
                    with tf.name_scope("CalibRegion"):
                        if self.verbose:
                            print(
                                "%s>  Including calib region (%d, %d)..."
                                % (name, shape_cal, shape_cal)
                            )
                        mask_calib = tf.ones(
                            [shape_cal, shape_cal, 1], dtype=tf.complex64
                        )
                        mask_calib = tf.image.resize_image_with_crop_or_pad(
                            mask_calib, self.height, self.width
                        )
                        mask_x = mask_x * (1 - mask_calib) + mask_calib

                    mask_recon = tf.abs(ks_x) / tf.reduce_max(tf.abs(ks_x))
                    mask_recon = tf.cast(mask_recon > 1e-7, dtype=tf.complex64)
                    mask_x = mask_x * mask_recon
                    print("mask x", mask_x)
                    # Assuming calibration region is fully sampled
                    shape_sc = 5
                    scale = tf.image.resize_image_with_crop_or_pad(
                        ks_x, shape_sc, shape_sc
                    )
                    scale = tf.reduce_mean(tf.square(tf.abs(scale))) * (
                        shape_sc * shape_sc / 1e5
                    )
                    scale = tf.cast(1.0 / tf.sqrt(scale), dtype=tf.complex64)
                    ks_x = ks_x * scale
                    # Masked input
                    ks_x = tf.multiply(ks_x, mask_x)
                if total_kspace is not None:
                    total_kspace = tf.concat([total_kspace, ks_x], 0)
                else:
                    total_kspace = ks_x

        x_measured = tf_util.model_transpose(
            total_kspace, sensemap, name="x_measured")
        x_measured = tf_util.complex_to_channels(x_measured)
        return x_measured

    def load(self, log_dir):
        ckpt = tf.train.get_checkpoint_state(log_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("[*] Model restored.")
            return True
        else:
            print("[*] Failed to find a checkpoint")
            return False
