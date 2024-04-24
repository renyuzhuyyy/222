import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

import losses
from utils import load_img_from_mat
from NNBmode import NNBmode
import scipy.io as sio


def predict(input_mat, model_h5, output_mat):
    # Network specifications
    depth = 8  # 8 layers of conv2d, relu, batchnorm
    nfilters = 32  # 32 filters per layer
    ksizes = 3  # Kernel size of 3x3 in all layers

    # Load data
    x = load_img_from_mat(input_mat)

    # Create model and load trained weights
    inputs = tf.keras.layers.Input(shape=x[0].shape, dtype=tf.float32, name="input")
    yhat = NNBmode(inputs, depth, nfilters, ksizes)
    model = tf.keras.Model(inputs=inputs, outputs=yhat)
    model.compile(
        tf.keras.optimizers.Adam(1e-3),
        loss=losses.get_l1loss_wopt_log,
        metrics=[
            losses.get_l1loss_wopt_log,
            losses.get_l2loss_wopt_log,
            losses.get_msssim_wopt_log,
        ],
    )

    # Load model weights
    model.load_weights(model_h5)

    # Run predictions
    p = model.predict(x=x, batch_size=1)

    # Transpose data to be read by MATLAB
    p = np.transpose(p, [3, 2, 1, 0])

    # Save output
    sio.savemat(output_mat, {"p": p})

    # Clear graph
    tf.keras.backend.clear_session()


if __name__ == "__main__":
    predict(
        "E:/Colab/data/test_PICMUS.mat",
        "E:/Colab/runs/pretrained/model.h5",
        "E:/Colab/runs/pretrained/results_PICMUS.mat",
    )
