"""MRI model."""
from __future__ import absolute_import, division, print_function

import tensorflow as tf

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


def _batch_norm_relu(tf_input, data_format="channels_last", training=False):
    #     tf_output = _batch_norm(
    #         tf_input, data_format=data_format, training=training)
    tf_output = tf.nn.relu(tf_input)
    return tf_output


def _circular_pad(tf_input, pad, axis):
    """Perform circular padding."""
    shape_input = tf.shape(tf_input)
    shape_0 = tf.cast(tf.reduce_prod(shape_input[:axis]), dtype=tf.int32)
    shape_axis = shape_input[axis]
    tf_output = tf.reshape(tf_input, tf.stack((shape_0, shape_axis, -1)))

    tf_pre = tf_output[:, shape_axis - pad :, :]
    tf_post = tf_output[:, :pad, :]
    tf_output = tf.concat((tf_pre, tf_output, tf_post), axis=1)

    shape_out = tf.concat(
        (shape_input[:axis], [shape_axis + 2 * pad], shape_input[axis + 1 :]), axis=0
    )
    tf_output = tf.reshape(tf_output, shape_out)

    return tf_output


def _conv2d(
    tf_input,
    num_features=128,
    kernel_size=3,
    data_format="channels_last",
    circular=True,
    conjugate=False,
):
    """Conv2d with option for circular convolution."""
    if data_format == "channels_last":
        # (batch, z, y, channels)
        axis_z = 1
        axis_y = 2
        axis_c = 3
    else:
        # (batch, channels, z, y)
        axis_c = 1
        axis_z = 2
        axis_y = 3

    pad = int((kernel_size - 0.5) / 2)
    tf_output = tf_input

    if circular:
        with tf.name_scope("circular_pad"):
            tf_output = _circular_pad(tf_output, pad, axis_z)
            tf_output = _circular_pad(tf_output, pad, axis_y)

    tf_output = tf.layers.conv2d(
        tf_output,
        num_features,
        kernel_size,
        padding="same",
        use_bias=False,
        data_format=data_format,
    )

    # print("complex convolution")
    # channels to complex
    # tf_output = tf_util.channels_to_complex(tf_output)
    # if num_features != 2:
    #     num_features = num_features // 2
    # tf_output = complex_utils.complex_conv2d(
    #     tf_output,
    #     num_features=num_features,
    #     kernel_size=kernel_size,
    #     padding="same",
    #     use_bias=False,
    #     data_format=data_format,
    # )
    # # complex to channels
    # tf_output = tf_util.complex_to_channels(tf_output)

    if circular:
        shape_input = tf.shape(tf_input)
        shape_z = shape_input[axis_z]
        shape_y = shape_input[axis_y]
        with tf.name_scope("circular_crop"):
            if data_format == "channels_last":
                tf_output = tf_output[
                    :, pad : (shape_z + pad), pad : (shape_y + pad), :
                ]
            else:
                tf_output = tf_output[
                    :, :, pad : (shape_z + pad), pad : (shape_y + pad)
                ]
    # add all needed attributes to tensor
    else:
        with tf.name_scope("non_circular"):
            tf_output = tf_output[:, :, :, :]

    return tf_output


def _res_block(
    net_input,
    num_features=32,
    kernel_size=3,
    data_format="channels_last",
    circular=True,
    training=True,
    name="res_block",
):
    """Create ResNet block.
    [1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
        Identity Mappings in Deep Residual Networks. arXiv: 1603.05027
    """
    if data_format == "channels_last":
        axis_z = 1
        axis_y = 2
        axis_c = 3
    else:
        axis_c = 1
        axis_z = 2
        axis_y = 3
    shape_c = net_input.shape[axis_c]
    pad = int((2 * (kernel_size - 1) + 0.5) / 2)

    with tf.name_scope(name):
        shortcut = net_input
        if num_features != shape_c:
            shortcut = _conv2d(
                shortcut,
                num_features=num_features,
                kernel_size=1,
                data_format=data_format,
                circular=circular,
            )

        net_cur = net_input

        if circular:
            with tf.name_scope("circular_pad"):
                net_cur = _circular_pad(net_cur, pad, axis_z)
                net_cur = _circular_pad(net_cur, pad, axis_y)

        net_cur = _batch_norm_relu(net_cur, data_format=data_format, training=training)
        net_cur = _conv2d(
            net_cur,
            num_features=num_features,
            kernel_size=kernel_size,
            data_format=data_format,
            circular=False,
        )
        net_cur = _batch_norm_relu(net_cur, data_format=data_format, training=training)
        net_cur = _conv2d(
            net_cur,
            num_features=num_features,
            kernel_size=kernel_size,
            data_format=data_format,
            circular=False,
        )

        if circular:
            shape_input = tf.shape(net_input)
            shape_z = shape_input[axis_z]
            shape_y = shape_input[axis_y]
            with tf.name_scope("circular_crop"):
                if data_format == "channels_last":
                    net_cur = net_cur[
                        :, pad : (pad + shape_z), pad : (pad + shape_y), :
                    ]
                else:
                    net_cur = net_cur[
                        :, :, pad : (pad + shape_z), pad : (pad + shape_y)
                    ]

        net_cur = net_cur + shortcut

    return net_cur


def prior_grad_res_net(
    curr_x,
    num_features=32,
    kernel_size=3,
    num_blocks=2,
    data_format="channels_last",
    do_residual=True,
    training=True,
    num_features_out=None,
    circular=True,
    name="prior_grad_resnet",
):
    """Create prior gradient."""
    if data_format == "channels_last":
        num_features_in = curr_x.shape[-1]
    else:
        num_features_in = curr_x.shape[1]
    if num_features_out is None:
        num_features_out = num_features_in

    with tf.name_scope(name):
        net = curr_x

        if do_residual:
            if num_features_in != num_features_out:
                shortcut = _conv2d(
                    net,
                    num_features=num_features_out,
                    kernel_size=1,
                    data_format=data_format,
                    circular=circular,
                )
            else:
                shortcut = net

        for _ in range(num_blocks):
            net = _res_block(
                net,
                training=training,
                num_features=num_features,
                kernel_size=kernel_size,
                data_format=data_format,
                circular=circular,
            )

        net = _batch_norm_relu(net, data_format=data_format, training=training)
        net = _conv2d(
            net,
            num_features=num_features_out,
            kernel_size=kernel_size,
            data_format=data_format,
            circular=circular,
        )

        if do_residual:
            net = net + shortcut

    return net


def prior_grad_simple(
    net_input,
    num_features=128,
    num_features_out=None,
    kernel_size=3,
    num_blocks=5,
    data_format="channels_last",
    do_residual=True,
    circular=True,
    training=True,
    name="prior_grad_simple",
):
    """Create prior gradient.
    This is based on the original work proposed by Diamond et al.
    """
    if data_format == "channels_last":
        axis_z = 1
        axis_y = 2
        axis_c = 3
    else:
        axis_c = 1
        axis_z = 2
        axis_y = 3
    num_features_in = net_input.shape[axis_c]
    if num_features_out is None:
        num_features_out = num_features_in
    # Number of total conv2d: 2 + num_blocks
    pad = int(((2 + num_blocks) * (kernel_size - 1) + 0.5) / 2)

    with tf.name_scope(name):
        net_cur = net_input

        if circular:
            with tf.name_scope("circular_pad"):
                net_cur = _circular_pad(net_cur, pad, axis_z)
                net_cur = _circular_pad(net_cur, pad, axis_y)

        # Expand to specified number of features
        net_cur = _conv2d(
            net_cur,
            num_features=num_features,
            kernel_size=kernel_size,
            data_format=data_format,
            circular=False,
        )
        net_cur = _batch_norm_relu(net_cur, data_format=data_format, training=training)

        # Repeat conv2d, bn, relu
        for _ in range(num_blocks):
            net_cur = _conv2d(
                net_cur,
                num_features=num_features,
                kernel_size=kernel_size,
                data_format=data_format,
                circular=False,
            )
            net_cur = _batch_norm_relu(
                net_cur, data_format=data_format, training=training
            )

        net_cur = _conv2d(
            net_cur,
            num_features=num_features_out,
            kernel_size=kernel_size,
            data_format=data_format,
            circular=False,
        )

        if circular:
            shape_input = tf.shape(net_input)
            shape_z = shape_input[axis_z]
            shape_y = shape_input[axis_y]

            with tf.name_scope("circular_crop"):
                if data_format == "channels_last":
                    net_cur = net_cur[
                        :, pad : (pad + shape_z), pad : (pad + shape_y), :
                    ]
                else:
                    net_cur = net_cur[
                        :, :, pad : (pad + shape_z), pad : (pad + shape_y)
                    ]

        if do_residual:
            net_cur = net_cur + net_input

    return net_cur


def unroll_fista(
    ks_input,
    sensemap,
    num_grad_steps=5,
    resblock_num_features=128,
    resblock_num_blocks=2,
    is_training=True,
    scope="MRI",
    mask_output=1,
    window=None,
    do_hardproj=True,
    num_summary_image=0,
    mask=None,
    verbose=False,
):
    """Create general unrolled network for MRI.
    x_{k+1} = S( x_k - 2 * t * A^T W (A x- b) )
            = S( x_k - 2 * t * (A^T W A x - A^T W b))
    """
    if window is None:
        window = 1
    summary_iter = None

    if verbose:
        print(
            "%s> Building FISTA unrolled network (%d steps)...."
            % (scope, num_grad_steps)
        )
        if sensemap is not None:
            print("%s>   Using sensitivity maps..." % scope)

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

        for i_step in range(num_grad_steps):
            # if i_step < 0:
            #     device_num = 0
            # else:
            #     device_num = 1
            # device_name = "/device:GPU:%d" % device_num
            # print(device_name)

            iter_name = "iter_%02d" % i_step
            # with tf.device(device_name), tf.variable_scope(iter_name):
            with tf.variable_scope(iter_name):
                # = S( x_k - 2 * t * (A^T W A x_k - A^T W b))
                # = S( x_k - 2 * t * (A^T W A x_k - x0))
                with tf.variable_scope("update"):
                    im_k_orig = im_k

                    # xk = A^T A x_k
                    ks_k = tf_util.model_forward(im_k, sensemap)
                    ks_k = mask * ks_k
                    im_k = tf_util.model_transpose(ks_k * window, sensemap)
                    # xk = A^T A x_k - A^T b
                    im_k = tf_util.complex_to_channels(im_k - im_0)
                    im_k_orig = tf_util.complex_to_channels(im_k_orig)
                    # Update step
                    t_update = tf.get_variable(
                        "t", dtype=tf.float32, initializer=tf.constant([-2.0])
                    )
                    im_k = im_k_orig + t_update * im_k

                with tf.variable_scope("prox"):
                    num_channels_out = im_k.shape[-1]
                    # Default is channels last
                    # im_k = prior_grad_res_net(im_k, training=is_training,
                    #                           cout=num_channels_out)

                    # Transpose channels_last to channels_first
                    im_k = tf.transpose(im_k, [0, 3, 1, 2])
                    im_k = prior_grad_res_net(
                        im_k,
                        training=is_training,
                        num_features=resblock_num_features,
                        num_blocks=resblock_num_blocks,
                        num_features_out=num_channels_out,
                        data_format="channels_first",
                    )
                    im_k = tf.transpose(im_k, [0, 2, 3, 1])

                    im_k = tf_util.channels_to_complex(im_k)

                im_k = tf.identity(im_k, name="image")
                if num_summary_image > 0:
                    with tf.name_scope("summary"):
                        tmp = tf_util.sumofsq(im_k, keep_dims=True)
                        if summary_iter is None:
                            summary_iter = tmp
                        else:
                            summary_iter = tf.concat((summary_iter, tmp), axis=2)
                        tf.summary.scalar("max/" + iter_name, tf.reduce_max(tmp))

        ks_k = tf_util.model_forward(im_k, sensemap)
        if do_hardproj:
            if verbose:
                print("%s>   Final hard data projection..." % scope)
            # Final data projection
            ks_k = mask * ks_0 + (1 - mask) * ks_k
            if mask_output is not None:
                ks_k = ks_k * mask_output
            im_k = tf_util.model_transpose(ks_k * window, sensemap)

        ks_k = tf.identity(ks_k, name="output_kspace")
        im_k = tf.identity(im_k, name="output_image")

    if summary_iter is not None:
        tf.summary.image("iter/image", summary_iter, max_outputs=num_summary_image)

    return im_k