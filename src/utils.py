import numpy as np
import h5py
import tensorflow as tf
import math
import losses


def load_from_mat(dataset_path, nimgs=0):

    print("Loading data from %s..." % dataset_path)

    if dataset_path != "":
        if nimgs == 0:
            x = np.array(h5py.File(dataset_path)["img"])
            y = np.array(h5py.File(dataset_path)["ref"])
        else:
            x = np.array(h5py.File(dataset_path)["img"][:nimgs])
            y = np.array(h5py.File(dataset_path)["ref"][:nimgs])

        # Transpose to put channels in dim 1, R/I in dim 2
        x = np.transpose(x, [0, 3, 4, 1, 2])
        y = np.transpose(y, [0, 3, 4, 1, 2])
        szi = x.shape
        szr = y.shape

        # For img, ref, combine the channels and R/I in dim 1
        x = x.reshape([szi[0], szi[1] * szi[2], szi[3], szi[4]])
        y = y.reshape([szr[0], szr[1] * szr[2], szr[3], szr[4]])
        # Get rid of any accidental non-positive values in ground truth
        # e.g., due to spline interpolation of ImageNet images
        y[y <= 0] = 0
        y += 1e-32  # Add eps to avoid NaN (will be added to network too)

        return x, y

    else:
        return None


def load_img_from_mat(dataset_path, nimgs=0):

    print("Loading data from %s..." % dataset_path)

    if dataset_path != "":
        if nimgs == 0:
            x = np.array(h5py.File(dataset_path)["img"])
        else:
            x = np.array(h5py.File(dataset_path)["img"][:nimgs])

        # Transpose to put channels in dim 1, R/I in dim 2
        x = np.transpose(x, [0, 3, 4, 1, 2])
        sz = x.shape

        # For img, ref, combine the channels and R/I in dim 1
        x = x.reshape([sz[0], sz[1] * sz[2], sz[3], sz[4]])

        return x

    else:
        return None


def make_bmode_tf(x):
    sz = x.shape.as_list()
    N = sz[1] // 2
    z = tf.complex(x[:, ::2], x[:, 1::2])
    z = tf.reduce_sum(x, axis=1, keepdims=True)
    z = tf.abs(z)
    z = z * z
    return z


def make_tensorboard_images(dynamic_range, p, b, y):
    # p = p / tf.reduce_max(p, [1, 2, 3], keepdims=True)
    # b = b / tf.reduce_max(b, [1, 2, 3], keepdims=True)
    # y = y / tf.reduce_max(y, [1, 2, 3], keepdims=True)

    # Normalize the B-mode image by its maximum value
    bnorm = b / tf.reduce_max(b, [1, 2, 3], keepdims=True)

    def compShift(img, dr, y):
        # Apply L2 optimal weight to img (w.r.t. y)
        wopt_dB = losses.compute_l2_wopt_dB(y, img)
        img_dB = losses._dB(img) + wopt_dB
        # Clip img_dB by dynamic range
        img_dB -= dr[0]
        img_dB /= dr[1] - dr[0]
        img_dB = tf.clip_by_value(img_dB, 0, 1)
        img_dB = tf.transpose(img_dB, [0, 3, 2, 1])
        return img_dB

    p = compShift(p, dynamic_range, bnorm)
    b = compShift(b, dynamic_range, bnorm)
    y = compShift(y, dynamic_range, bnorm)

    return p, b, y


class TensorBoardBmode(tf.keras.callbacks.TensorBoard):
    """
    TensorBoardBmode extends tf.keras.callbacks.TensorBoard, adding custom processing
    upon setup and after every epoch to store properly processed ultrasound images.
    """

    def __init__(self, val_data, *args, **kwargs):
        # Use base class initialization
        self.val_data = val_data  # Validation data to be used for TensorBoard
        super().__init__(*args, **kwargs)

    def set_model(self, model):
        """ Override set_model function to add image TensorBoard summaries. """
        super().set_model(model)

        # Make ground truth, NN B-mode, and DAS B-mode images
        dynamic_range = [-60, 0]
        bimg = make_bmode_tf(self.model.inputs[0])  # DAS B-mode image
        yhat = self.model.outputs[0]  # Predictions
        ytgt = self.val_data[1]  # Use passed ground truth directly from validation data

        # Adjust the shape of ground truth to match prediction if necessary
        szy = tf.shape(yhat)
        ytgt = tf.reshape(ytgt, szy)

        # Process images for TensorBoard display
        yhat, bimg, ytgt = make_tensorboard_images(dynamic_range, yhat, bimg, ytgt)

        # Prepare summaries
        self.bsumm = tf.summary.image("Bmode", bimg)
        self.ysumm = tf.summary.image("Target", ytgt)
        self.psumm = tf.summary.image("Output", yhat)

    def on_epoch_end(self, epoch, logs=None):
        """ At the end of each epoch, add prediction images to TensorBoard."""
        super().on_epoch_end(epoch, logs)
        # Prepare data for summaries
        if epoch == 0:
            bs, ys = self._sess.run([self.bsumm, self.ysumm], feed_dict={self.model.input: self.val_data[0]})
            self.writer.add_summary(bs, 0)
            self.writer.add_summary(ys, 0)

        # Add the predictions every 10 epochs
        if epoch % 10 == 9:
            ps = self._sess.run(self.psumm, feed_dict={self.model.input: self.val_data[0]})
            self.writer.add_summary(ps, epoch + 1)

        self.writer.flush()

    def _sess(self):
        return tf.keras.backend.get_session()
