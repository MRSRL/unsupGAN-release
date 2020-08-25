import os

import numpy as np
import scipy.misc
import tensorflow as tf
from tensorflow.python.util import deprecation

from supervised_model import WGAN

deprecation._PRINT_DEPRECATION_WARNINGS = False
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
flags = tf.app.flags
flags.DEFINE_float("max_epoch", 2, "Maximum epoch")
flags.DEFINE_integer("batch_size", 1, "Batch size")
flags.DEFINE_boolean("time", False, "2D versus 2D plus time")
flags.DEFINE_integer("d_steps", 1, "Number of critic iteration")
flags.DEFINE_integer("g_steps", 1, "Number of generator iteration")
flags.DEFINE_float("beta1", 0.9, "beta1 for Adam Optimizer")
flags.DEFINE_float("beta2", 0.999, "beta2 for Adam Optimizer")
flags.DEFINE_float("learning_rate", 1e-12, "learning rate")
flags.DEFINE_integer("g_dim", 128, "Dimension of generator")
flags.DEFINE_integer("d_dim", 128, "Dimension of discriminator")
flags.DEFINE_integer(
    "res_blocks", 4, "Number of resblocks in unrolled generator")
flags.DEFINE_integer(
    "iterations", 5, "Number of iterations in unrolled generator")
flags.DEFINE_boolean("train", False, "train")
flags.DEFINE_string("mask_path", "/home_local/ekcole/knee_masks", "mask path")
flags.DEFINE_string("arch", "unrolled", "architecture of generator")
flags.DEFINE_string("data_type", "knee", "knee, DCE, or mfast")
flags.DEFINE_integer("train_acc", None, "R of training dataset")

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "log_root",
    "/home_local/ekcole/"
    + FLAGS.data_type
    + "/supervised_"
    + "5_cal"
    + str(FLAGS.train_acc),
    "Log directory path",
)
FLAGS = flags.FLAGS
print("Model directory", FLAGS.log_root)


def main(_):
    run_config = tf.ConfigProto()
    with tf.Session(config=run_config) as sess:
        tf.logging.set_verbosity(tf.logging.INFO)
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
            train_acc=FLAGS.train_acc,
            iterations=FLAGS.iterations,
        )
        if FLAGS.train:
            wgan.build_model(mode="train")
            wgan.train()
        else:
            wgan.build_model(mode="test")
            wgan.test()


if __name__ == "__main__":
    tf.app.run()
