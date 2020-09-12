import keras
from keras import layers
from keras import models
import tensorflow as tf
import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt
from keras.utils import Sequence

from keras.applications import MobileNet
from keras import backend as K


# The generator for training
class Generator(Sequence):
    def __init__(self, images, batch_size, insize=64, outsize=240, factor=4, pretrained=None):

        self.image_set = images
        self.batch_size = batch_size
        self.len = int(np.ceil(len(self.image_set) / self.batch_size))
        self.insize = (insize, insize)
        self.model_in = (insize * factor, insize * factor)
        self.start = int((insize * factor - outsize) / 2)
        self.end = self.start + outsize
        self.pretrained = pretrained

    def __len__(self):
        return int(self.len)

    def __getitem__(self, idx):
        y = []
        x = []
        current_list = self.image_set[int(idx * self.batch_size): int((idx + 1) * self.batch_size)]
        for img in current_list:
            image = cv2.imread(img)
            in_image = cv2.resize(image, self.insize)
            in_image = cv2.resize(in_image, self.model_in)
            out_image = cv2.resize(image, self.model_in)
            out_image = out_image[self.start:self.end, self.start:self.end, :]
            x.append(in_image.astype(np.float32))
            y.append(out_image.astype(np.float32))

        x = np.asarray(x)
        y = np.asarray(y)
        if self.pretrained is not None:
            outputs = self.pretrained(y)
            # y = {}
            # for i,layer in enumerate(pretrained_multi_output.output_names):
            #     y[layer] = outputs[i]
            y = outputs

        return x / 255.0, y / 255.0



# The pretrained model for perception loss
pretrained = MobileNet(input_shape = (None,None,3),
                       include_top = False,
                       weights = 'imagenet')
layer_for_perception_loss = [4,7,14,67,-1] # mobilenet layers
pretrained_multi_output = models.Model(pretrained.input,outputs = [pretrained.layers[i].output for i in layer_for_perception_loss],name = 'perception')
pretrained_multi_output.trainable = False
def percep_loss(y_true,y_pred):
    c_true = pretrained_multi_output(y_true)
    c_pred = pretrained_multi_output(y_pred)
    loss = 0
    for p,t in zip(c_true,c_pred):
        loss+=K.mean(K.square(p-t))
    return loss



