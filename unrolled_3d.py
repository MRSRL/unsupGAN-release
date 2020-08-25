"""MRI model."""
from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.python.client import device_lib

# import complex_utils
from mri_util import tf_util


def _batch_norm(tf_input, data_format="channels_last", training=False):
    tf_output = tf.layers.batch_normalization(
        tf_input,
        axis=(1 if data_format == "channels_first" else -1),
        training=training,
        renorm=True,
        fused=True,
    )
    return tf_output


def _batch_norm_relu(
    tf_input, data_format="channels_last", batchnorm=True, leaky=False, training=False
):
    if batchnorm:
        tf_output = _batch_norm(tf_input, data_format=data_format, training=training)
    else:
        tf_output = tf_input
    if leaky:
        tf_output = tf.nn.leaky_relu(tf_output)
    else:
        tf_output = tf.nn.relu(tf_output)
    return tf_output


def _bias_add(tf_input, num_features, data_format="channels_last", name="bias_add"):
    """Bias add for 3D conv functionality."""
    with tf.name_scope(name):
        b_shape = [1, int(num_features), 1, 1, 1]
        bias = tf.Variable(tf.zeros(b_shape), name="bias")
        if data_format == "channels_last":
            bias = tf.transpose(bias, [0, 2, 3, 4, 1])
        tf_output = tf_input + bias

    return tf_output


def _conv3d(
    tf_input,
    num_features=128,
    kernel_size=[3, 3, 5],
    data_format="channels_last",
    circular=True,
    separable=False,
    use_bias=False,
):
    """Conv3d with option for circular convolution and separable convolution."""
    if data_format == "channels_last":
        # (batch, z, y, t, channels)
        axis_x = 1
        axis_y = 2
        axis_t = 3
        axis_c = 4
    else:
        # (batch, channels, z, y, t)
        axis_c = 1
        axis_x = 2
        axis_y = 3
        axis_t = 4

    pad_x = int((kernel_size[0] - 0.5) / 2)
    pad_y = int((kernel_size[1] - 0.5) / 2)
    pad_t = int((kernel_size[2] - 0.5) / 2)

    tf_output = tf_input
    shape_c = tf.shape(tf_input)[axis_c]
    shape_x = tf.shape(tf_input)[axis_x] + 2 * pad_x
    shape_y = tf.shape(tf_input)[axis_y] + 2 * pad_y
    shape_t = tf.shape(tf_input)[axis_t] + 2 * pad_t

    if circular and (pad_t > 0) and (pad_y > 0):
        with tf.name_scope("circular_pad"):
            # tf_output = tf_util.circular_pad(tf_output, pad_x, axis_x)
            tf_output = tf_util.circular_pad(tf_output, pad_y, axis_y)
            tf_output = tf_util.circular_pad(tf_output, pad_t, axis_t)

    if separable and (kernel_size != [1, 1, 1]):
        # number of latent features is chosen to make DW conv3d have the same
        # number of parameters as a regular conv3d
        d2 = kernel_size[0] * kernel_size[1]
        t = kernel_size[2]
        N1 = int(tf_input.get_shape()[axis_c])  # input features
        N2 = int(num_features)  # output features
        num_latent_features = int(t * d2 * N1 * N2 // (d2 * N1 + t * N2))
        # if num_latent_features%2 != 0:
        #     num_latent_features -= 1
        with tf.name_scope("spconv"):
            sp_kernel = [kernel_size[0], kernel_size[1], 1]
            tf_output = tf.layers.conv3d(tf_output, num_latent_features, sp_kernel,
                                         padding='same', use_bias=False,
                                         data_format=data_format)

            if use_bias:
                tf_output = _bias_add(
                    tf_output, num_latent_features, data_format=data_format
                )

        tf_output = tf.nn.relu(tf_output)

        with tf.name_scope("tconv"):
            t_kernel = [1, 1, kernel_size[2]]
            tf_output = tf.layers.conv3d(
                tf_output,
                num_features,
                t_kernel,
                padding="same",
                use_bias=False,
                data_format=data_format,
            )

    else:
        tf_output = tf.layers.conv3d(
            tf_output,
            num_features,
            kernel_size,
            padding="same",
            use_bias=False,
            data_format=data_format,
        )

    if use_bias:
        tf_output = _bias_add(tf_output, num_features, data_format=data_format)

    if circular and (pad_t > 0) and (pad_y > 0):
        with tf.name_scope("circular_crop"):
            if data_format == "channels_last":
                tf_output = tf_output[
                    :, :, pad_y : (shape_y + pad_y), pad_t : (shape_t + pad_t), :
                ]
            else:
                tf_output = tf_output[
                    :, :, :, pad_y : (shape_y + pad_y), pad_t : (shape_t + pad_t)
                ]

    return tf_output


def _res_block(
    net_input,
    num_features=32,
    kernel_size=[3, 3, 5],
    data_format="channels_last",
    circular=True,
    separable=False,
    leaky=False,
    batchnorm=False,
    training=True,
    name="res_block",
):
    """Create ResNet block.
    [1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
        Identity Mappings in Deep Residual Networks. arXiv: 1603.05027
    """
    if data_format == "channels_last":
        # (batch, x, y, t, channels)
        axis_x = 1
        axis_y = 2
        axis_t = 3
        axis_c = 4
    else:
        # (batch, channels, x, y, t)
        axis_c = 1
        axis_x = 2
        axis_y = 3
        axis_t = 4

    pad_x = int((kernel_size[0] - 0.5) / 2)
    pad_y = int((kernel_size[1] - 0.5) / 2)
    pad_t = int((kernel_size[2] - 0.5) / 2)

    shape_c = tf.shape(net_input)[axis_c]
    shape_x = tf.shape(net_input)[axis_x]
    shape_y = tf.shape(net_input)[axis_y]
    shape_t = tf.shape(net_input)[axis_t]

    with tf.name_scope(name):

        shortcut = net_input
        if num_features != shape_c:
            shortcut = _conv3d(
                shortcut,
                num_features=num_features,
                kernel_size=[1, 1, 1],
                data_format=data_format,
                circular=False,
                separable=separable,
                use_bias=(not batchnorm),
            )

        net_cur = net_input

        if circular:
            with tf.name_scope("circular_pad"):
                # net_cur = tf_util.circular_pad(net_cur, pad_x, axis_x)
                net_cur = tf_util.circular_pad(net_cur, pad_y, axis_y)
                net_cur = tf_util.circular_pad(net_cur, pad, axis_t)

        net_cur = _batch_norm_relu(
            net_cur,
            data_format=data_format,
            batchnorm=batchnorm,
            leaky=leaky,
            training=training,
        )
        net_cur = _conv3d(
            net_cur,
            num_features=num_features,
            kernel_size=kernel_size,
            data_format=data_format,
            circular=False,
            separable=separable,
            use_bias=(not batchnorm),
        )
        net_cur = _batch_norm_relu(
            net_cur,
            data_format=data_format,
            batchnorm=batchnorm,
            leaky=leaky,
            training=training,
        )
        net_cur = _conv3d(
            net_cur,
            num_features=num_features,
            kernel_size=kernel_size,
            data_format=data_format,
            circular=False,
            separable=separable,
            use_bias=(not batchnorm),
        )

        if circular:
            with tf.name_scope("circular_crop"):
                if data_format == "channels_last":
                    net_cur = net_cur[
                        :, :, pad_y : (pad_y + shape_y), pad_t : (pad_t + shape_t), :
                    ]
                else:
                    net_cur = net_cur[
                        :, :, :, pad_y : (pad_y + shape_y), pad_t : (pad_t + shape_t)
                    ]

        net_cur = net_cur + shortcut

    return net_cur


def prior_grad_res_net(
    curr_x,
    num_features=32,
    num_blocks=2,
    kernel_size=[3, 3, 5],
    circular=True,
    separable=False,
    data_format="channels_last",
    do_residual=True,
    batchnorm=True,
    leaky=False,
    training=True,
    num_features_out=None,
    name="prior_grad_resnet",
):
    """Create prior gradient."""
    if data_format == "channels_last":
        # (batch, x, y, t, channels)
        axis_x = 1
        axis_y = 2
        axis_t = 3
        axis_c = 4
    else:
        # (batch, channels, x, y, t)
        axis_c = 1
        axis_x = 2
        axis_y = 3
        axis_t = 4

    num_features_in = curr_x.shape[axis_c]
    if num_features_out is None:
        num_features_out = num_features_in

    num_conv3d = num_blocks * 2 + 1
    pad_x = int((num_conv3d * (kernel_size[0] - 1) + 0.5) / 2)
    pad_y = int((num_conv3d * (kernel_size[1] - 1) + 0.5) / 2)
    pad_t = int((num_conv3d * (kernel_size[2] - 1) + 0.5) / 2)

    shape_x = tf.shape(curr_x)[axis_x]
    shape_y = tf.shape(curr_x)[axis_y]
    shape_t = tf.shape(curr_x)[axis_t]

    with tf.name_scope(name):
        net = curr_x
        shortcut = net

        if do_residual and (num_features_in != num_features_out):
            shortcut = _conv3d(
                shortcut,
                num_features=num_features_out,
                kernel_size=[1, 1, 1],
                data_format=data_format,
                circular=False,
                separable=separable,
                use_bias=(not batchnorm),
            )

        if circular:
            with tf.name_scope("circular_pad"):
                # net = tf_util.circular_pad(net, pad_x, axis_x)
                net = tf_util.circular_pad(net, pad_y, axis_y)
                net = tf_util.circular_pad(net, pad_t, axis_t)

        for _ in range(num_blocks):
            net = _res_block(
                net,
                training=training,
                num_features=num_features,
                kernel_size=kernel_size,
                data_format=data_format,
                batchnorm=batchnorm,
                circular=False,
                separable=separable,
                leaky=leaky,
            )

        # Save network before last conv for densely connected network
        net_dense = net
        net = _batch_norm_relu(
            net,
            data_format=data_format,
            batchnorm=batchnorm,
            leaky=leaky,
            training=training,
        )
        net = _conv3d(
            net,
            num_features=num_features_out,
            kernel_size=kernel_size,
            data_format=data_format,
            circular=False,
            separable=separable,
            use_bias=(not batchnorm),
        )

        if circular:
            with tf.name_scope("circular_crop"):
                if data_format == "channels_last":
                    net = net[
                        :, :, pad_y : (pad_y + shape_y), pad_t : (pad_t + shape_t), :
                    ]
                    net_dense = net_dense[
                        :, :, pad_y : (pad_y + shape_y), pad_t : (pad_t + shape_t), :
                    ]
                else:
                    net = net[
                        :, :, :, pad_y : (pad_y + shape_y), pad_t : (pad_t + shape_t)
                    ]
                    net_dense = net_dense[
                        :, :, :, pad_y : (pad_y + shape_y), pad_t : (pad_t + shape_t)
                    ]

        if do_residual:
            net = net + shortcut

    return net, net_dense


def unroll_fista(
    ks_input,
    sensemap,
    scope="MRI",
    num_grad_steps=5,
    num_resblocks=4,
    num_features=64,
    kernel_size=[3, 3, 5],
    is_training=True,
    mask_output=1,
    mask=None,
    window=None,
    do_hardproj=False,
    do_dense=False,
    do_separable=False,
    do_rnn=False,
    do_circular=True,
    batchnorm=False,
    leaky=False,
    fix_update=False,
    data_format="channels_first",
    verbose=False,
):
    """Create general unrolled network for MRI.
    x_{k+1} = S( x_k - 2 * t * A^T W (A x- b) )
            = S( x_k - 2 * t * (A^T W A x - A^T W b))
    """
    # get list of GPU devices
    local_device_protos = device_lib.list_local_devices()
    device_list = [x.name for x in local_device_protos if x.device_type == "GPU"]

    if window is None:
        window = 1
    summary_iter = {}

    if verbose:
        print("%s> Building FISTA unrolled network...." % scope)
        print("%s>   Num of gradient steps: %d" % (scope, num_grad_steps))
        print(
            "%s>   Prior: %d ResBlocks, %d features"
            % (scope, num_resblocks, num_features)
        )
        print("%s>   Kernel size: [%d x %d x %d]" % ((scope,) + tuple(kernel_size)))
        if do_rnn:
            print("%s>   Sharing weights across iterations..." % scope)
        if sensemap is not None:
            print("%s>   Using sensitivity maps..." % scope)
        if do_dense:
            print("%s>   Inserting dense connections..." % scope)
        if do_circular:
            print("%s>   Using circular padding..." % scope)
        if do_separable:
            print("%s>   Using depth-wise separable convolutions..." % scope)
        if not batchnorm:
            print("%s>   Turning off batch normalization..." % scope)

    with tf.variable_scope(scope):
        if mask is None:
            mask = tf_util.kspace_mask(ks_input, dtype=tf.complex64)
        ks_input = mask * ks_input
        ks_0 = ks_input
        # x0 = A^T W b
        im_0 = tf_util.model_transpose(ks_0 * window, sensemap)
        im_0 = tf.identity(im_0, name="input_image")
        # To be updated
        ks_k = ks_0
        im_k = im_0
        im_dense = None

        for i_step in range(num_grad_steps):
            iter_name = "iter_%02d" % i_step
            if do_rnn:
                scope_name = "iter"
            else:
                scope_name = iter_name

            # figure out which GPU to use for this step
            # i_device = int(len(device_list) * i_step / num_grad_steps)
            # cur_device = device_list[i_device]

            # with tf.device(cur_device):
            with tf.variable_scope(
                scope_name, reuse=(tf.AUTO_REUSE if do_rnn else False)
            ):
                with tf.variable_scope("update"):
                    # = S( x_k - 2 * t * (A^T W A x_k - A^T W b))
                    # = S( x_k - 2 * t * (A^T W A x_k - x0))
                    im_k_orig = im_k
                    # xk = A^T A x_k
                    ks_k = tf_util.model_forward(im_k, sensemap)
                    ks_k = mask * ks_k
                    im_k = tf_util.model_transpose(ks_k * window, sensemap)
                    # xk = A^T A x_k - A^T b
                    im_k = tf_util.complex_to_channels(im_k - im_0)
                    im_k_orig = tf_util.complex_to_channels(im_k_orig)
                    # Update step
                    if fix_update:
                        t_update = -2.0
                    else:
                        t_update = tf.get_variable(
                            "t", dtype=tf.float32, initializer=tf.constant([-2.0])
                        )
                    im_k = im_k_orig + t_update * im_k

                with tf.variable_scope("prox"):
                    # default is channels_last
                    num_channels_out = im_k.shape[-1]
                    if data_format == "channels_first":
                        im_k = tf.transpose(im_k, [0, 4, 1, 2, 3])

                    if im_dense is not None:
                        im_k = tf.concat([im_k, im_dense], axis=1)

                    im_k, im_dense_k = prior_grad_res_net(
                        im_k,
                        training=is_training,
                        num_features=num_features,
                        num_blocks=num_resblocks,
                        num_features_out=num_channels_out,
                        kernel_size=kernel_size,
                        data_format=data_format,
                        circular=do_circular,
                        separable=do_separable,
                        batchnorm=batchnorm,
                        leaky=leaky,
                    )

                    if do_dense:
                        if im_dense is not None:
                            im_dense = tf.concat([im_dense, im_dense_k], axis=1)
                        else:
                            im_dense = im_dense_k

                    if data_format == "channels_first":
                        im_k = tf.transpose(im_k, [0, 2, 3, 4, 1])

                    im_k = tf_util.channels_to_complex(im_k)

                im_k = tf.identity(im_k, name="image")

                with tf.name_scope("summary"):
                    # tmp = tf_util.sumofsq(im_k, keep_dims=True)
                    summary_iter[iter_name] = im_k

        ks_k = tf_util.model_forward(im_k, sensemap)
        if do_hardproj:
            if verbose:
                print("%s>   Final hard data projection..." % scope)
            ks_k = mask * ks_0 + (1 - mask) * ks_k
            if mask_output is not None:
                ks_k = ks_k * mask_output
            im_k = tf_util.model_transpose(ks_k * window, sensemap)

        ks_k = tf.identity(ks_k, name="output_kspace")
        im_k = tf.identity(im_k, name="output_image")

    #     return im_k, ks_k, summary_iter
    return im_k
