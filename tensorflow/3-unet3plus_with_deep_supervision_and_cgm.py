import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import tensorflow.keras.layers as L

def conv_block(x, num_filters, act=True):
    x = L.Conv2D(num_filters, kernel_size=3, padding="same")(x)

    if act == True:
        x = L.BatchNormalization()(x)
        x = L.Activation("relu")(x)

    return x

def encoder_block(x, num_filters):
    x = conv_block(x, num_filters)
    x = conv_block(x, num_filters)

    p = L.MaxPool2D((2, 2))(x)
    return x, p

def unet3plus(input_shape, num_classes=1, deep_sup=True):
    """ Inputs """
    inputs = L.Input(input_shape, name="input_layer")

    """ Encoder """
    e1, p1 = encoder_block(inputs, 64)
    e2, p2 = encoder_block(p1, 128)
    e3, p3 = encoder_block(p2, 256)
    e4, p4 = encoder_block(p3, 512)

    """ Bottleneck """
    e5 = conv_block(p4, 1024)
    e5 = conv_block(e5, 1024)

    """ Classification """
    cls = L.Dropout(0.5)(e5)
    cls = L.Conv2D(2, kernel_size=1, padding="same")(cls)
    cls = L.GlobalMaxPooling2D()(cls)
    cls = L.Activation("sigmoid")(cls)
    cls = tf.argmax(cls, axis=-1)
    cls = cls[..., tf.newaxis]
    cls = tf.cast(cls, dtype=tf.float32)

    """ Decoder 4 """
    e1_d4 = L.MaxPool2D((8, 8))(e1)
    e1_d4 = conv_block(e1_d4, 64)

    e2_d4 = L.MaxPool2D((4, 4))(e2)
    e2_d4 = conv_block(e2_d4, 64)

    e3_d4 = L.MaxPool2D((2, 2))(e3)
    e3_d4 = conv_block(e3_d4, 64)

    e4_d4 = conv_block(e4, 64)

    e5_d4 = L.UpSampling2D((2, 2), interpolation="bilinear")(e5)
    e5_d4 = conv_block(e5_d4, 64)

    d4 = L.Concatenate()([e1_d4, e2_d4, e3_d4, e4_d4, e5_d4])
    d4 = conv_block(d4, 64*5)

    """ Decoder 3 """
    e1_d3 = L.MaxPool2D((4, 4))(e1)
    e1_d3 = conv_block(e1_d3, 64)

    e2_d3 = L.MaxPool2D((2, 2))(e2)
    e2_d3 = conv_block(e2_d3, 64)

    e3_d3 = conv_block(e3, 64)

    d4_d3 = L.UpSampling2D((2, 2), interpolation="bilinear")(d4)
    d4_d3 = conv_block(d4_d3, 64)

    e5_d3 = L.UpSampling2D((4, 4), interpolation="bilinear")(e5)
    e5_d3 = conv_block(e5_d3, 64)

    d3 = L.Concatenate()([e1_d3, e2_d3, e3_d3, d4_d3, e5_d3])
    d3 = conv_block(d3, 64*5)

    """ Decoder 2 """
    e1_d2 = L.MaxPool2D((2, 2))(e1)
    e1_d2 = conv_block(e1_d2, 64)

    e2_d2 = conv_block(e2, 64)

    d3_d2 = L.UpSampling2D((2, 2), interpolation="bilinear")(d3)
    d3_d2 = conv_block(d3_d2, 64)

    d4_d2 = L.UpSampling2D((4, 4), interpolation="bilinear")(d4)
    d4_d2 = conv_block(d4_d2, 64)

    e5_d2 = L.UpSampling2D((8, 8), interpolation="bilinear")(e5)
    e5_d2 = conv_block(e5_d2, 64)

    d2 = L.Concatenate()([e1_d2, e2_d2, d3_d2, d4_d2, e5_d2])
    d2 = conv_block(d2, 64*5)

    """ Decoder 1 """
    e1_d1 = conv_block(e1, 64)

    d2_d1 = L.UpSampling2D((2, 2), interpolation="bilinear")(d2)
    d2_d1 = conv_block(d2_d1, 64)

    d3_d1 = L.UpSampling2D((4, 4), interpolation="bilinear")(d3)
    d3_d1 = conv_block(d3_d1, 64)

    d4_d1 = L.UpSampling2D((8, 8), interpolation="bilinear")(d4)
    d4_d1 = conv_block(d4_d1, 64)

    e5_d1 = L.UpSampling2D((16, 16), interpolation="bilinear")(e5)
    e5_d1 = conv_block(e5_d1, 64)

    d1 = L.Concatenate()([e1_d1, d2_d1, d3_d1, d4_d1, e5_d1])
    d1 = conv_block(d1, 64*5)

    """ Deep Supervision and CGM (Classification Guided Module) """
    if deep_sup == True:
        y1 = L.Conv2D(num_classes, kernel_size=3, padding="same")(d1)
        y1 = y1 * cls
        y1 = L.Activation("sigmoid")(y1)

        y2 = L.Conv2D(num_classes, kernel_size=3, padding="same")(d2)
        y2 = L.UpSampling2D((2, 2), interpolation="bilinear")(y2)
        y2 = y2 * cls
        y2 = L.Activation("sigmoid")(y2)

        y3 = L.Conv2D(num_classes, kernel_size=3, padding="same")(d3)
        y3 = L.UpSampling2D((4, 4), interpolation="bilinear")(y3)
        y3 = y3 * cls
        y3 = L.Activation("sigmoid")(y3)

        y4 = L.Conv2D(num_classes, kernel_size=3, padding="same")(d4)
        y4 = L.UpSampling2D((8, 8), interpolation="bilinear")(y4)
        y4 = y4 * cls
        y4 = L.Activation("sigmoid")(y4)

        y5 = L.Conv2D(num_classes, kernel_size=3, padding="same")(e5)
        y5 = L.UpSampling2D((16, 16), interpolation="bilinear")(y5)
        y5 = y5 * cls
        y5 = L.Activation("sigmoid")(y5)

        outputs = [y1, y2, y3, y4, y5, cls]

    else:
        y1 = L.Conv2D(num_classes, kernel_size=3, padding="same")(d1)
        y1 = L.Activation("sigmoid")(y1)
        outputs = [y1]


    model = tf.keras.Model(inputs, outputs)
    return model


if __name__ == "__main__":
    input_shape = (256, 256, 3)
    model = unet3plus(input_shape)
    model.summary()
