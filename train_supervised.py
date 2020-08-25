import os
from sys import argv

import numpy as np
import tensorflow as tf
from tensorflow.python.util import deprecation

from supervised_model import WGAN

deprecation._PRINT_DEPRECATION_WARNINGS = False
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

if len(argv) > 1:
    data_dir = argv[1]  # Data directory (datasets and masks)
    model_dir = argv[2]  # Directory where model  will be saved

flags = tf.app.flags
flags.DEFINE_float("max_epoch", 1, "Maximum epoch")
flags.DEFINE_integer("batch_size", 1, "Batch size")
flags.DEFINE_boolean("time", False, "False if 2D; True if 2D plus time")
flags.DEFINE_integer("d_steps", 1, "Number of critic iteration")
flags.DEFINE_integer("g_steps", 1, "Number of generator iteration")
flags.DEFINE_float("beta1", 0.9, "beta1 for Adam Optimizer")
flags.DEFINE_float("beta2", 0.999, "beta2 for Adam Optimizer")
flags.DEFINE_float("learning_rate", 1e-8, "Learning rate for Adam Optimizer")
flags.DEFINE_integer("g_dim", 128, "Number of feature maps in generator")
flags.DEFINE_integer("d_dim", 128, "Number of feature maps in discriminator")
flags.DEFINE_integer(
    "res_blocks", 4, "Number of resblocks in unrolled generator")
flags.DEFINE_boolean("train", True, "train")
flags.DEFINE_string(
    "mask_path", "/home_local/ekcole/vd_masks_uniform", "mask path")
flags.DEFINE_string("arch", "unrolled", "architecture of generator")
flags.DEFINE_string("data_type", "knee", "knee or DCE_2D")

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "log_root",
    model_dir + FLAGS.data_type + "/supervised"
    + "/g_steps"
    + str(FLAGS.g_steps)
    + "_d_steps"
    + str(FLAGS.d_steps)
    + "_g_dim"
    + str(FLAGS.g_dim)
    + "_d_dim"
    + str(FLAGS.d_dim)
    + "_res_blocks"
    + str(FLAGS.res_blocks)
    + "_lr"
    + str(FLAGS.learning_rate),
    "Log directory path",
)
FLAGS = flags.FLAGS
print("Model directory", FLAGS.log_root)


def main(_):
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    with tf.Session(config=run_config) as sess:
        wgan = WGAN(
            sess,
            batch_size=FLAGS.batch_size,
            g_dim=FLAGS.g_dim,
            d_dim=FLAGS.d_dim,
            res_blocks=FLAGS.res_blocks,
            log_dir=FLAGS.log_root,
            max_epoch=FLAGS.max_epoch,
            d_steps=FLAGS.d_steps,
            g_steps=FLAGS.g_steps,
            lr=FLAGS.learning_rate,
            beta1=FLAGS.beta1,
            beta2=FLAGS.beta2,
            mask_path=FLAGS.mask_path,
            data_type=FLAGS.data_type,
            time=FLAGS.time,
            data_dir=data_dir
        )
        if FLAGS.train:
            wgan.build_model(mode="train")
            wgan.train()
        else:
            wgan.build_model(mode="test")
            wgan.test()


if __name__ == "__main__":
    tf.app.run()
