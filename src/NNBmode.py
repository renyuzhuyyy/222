import tensorflow as tf
from tensorflow.keras import layers


def NNBmode(inputs, nlayers, filters, kernel_size):
    x = inputs
    # Convolution, activation, batch normalization
    for _ in range(nlayers):
        print(x.get_shape())
        x = layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            data_format="channels_first",
            activation="relu",
        )(x)
        # x = layers.BatchNormalization(axis=1)(x)

    # Concatenate input to end of convolution blocks
    print(x.get_shape())
    x = layers.Concatenate(axis=1)([x, inputs])

    # Convert filters into a single filter
    print(x.get_shape())
    x = layers.Conv2D(filters=1, kernel_size=1, data_format="channels_first")(x)

    # Square to enforce non-negative
    print(x.get_shape())
    x = layers.Lambda(lambda z: z * z + 1e-32)(x)
    return x