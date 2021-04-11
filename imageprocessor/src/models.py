""" Using DNN for feature extraction

"""
import numpy as np
from keras.models import Model, Sequential
from keras.layers import BatchNormalization, add, GlobalAveragePooling2D
from keras.applications.xception import Xception, preprocess_input
from keras.layers import Activation, Input, concatenate, Dropout, Dense

import tensorflow as tf
import params as cfg


class XceptiponModel():
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.drop_out = cfg.dropout_rate
        self.model = Xception(weights='imagenet', include_top=False, pooling = 'avg',
                                input_tensor=Input(shape=self.input_shape))

    def xception_gap(self):
        """ Global Average Pooling on Xception network
        """
        c1 = self.model.layers[16].output
        c1 = GlobalAveragePooling2D()(c1)

        c2 = self.model.layers[26].output
        c2 = GlobalAveragePooling2D()(c2)

        c3 = self.model.layers[36].output
        c3 = GlobalAveragePooling2D()(c3)

        c4 = self.model.layers[126].output
        c4 = GlobalAveragePooling2D()(c4)

        con = concatenate([c2, c3, c4])

        bottleneck_final_model = Model(inputs=self.model.input, outputs=con)

        # Classification model

        model = Sequential()
        model.add(bottleneck_final_model)
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(self.drop_out))
        model.add(Dense(self.num_classes, activation="softmax"))

        return model


    def xception_alone(self):

        logits = Dense(self.num_classes)(self.model.layers[-1].output)
        output = Activation('softmax')(logits)
        model = Model(self.model.input, output)

        return model

""" test """

# xception = XceptiponModel(input_shape=cfg.input_shape, num_classes=cfg.num_classes)
# model = xception.xception_gap()

# model.summary()



