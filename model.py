
from keras import layers
from keras import models


class SuperResolution:
    def __init__(self,n1,f1,n2,f2,f3,pretrained = None):
        self.model = self.create_model(n1,f1,n2,f2,f3)
        if pretrained!=None:
            inp = layers.Input(shape = (None,None,3),name = 'SR_input')
            out = self.model(inp)
            percep_out = pretrained(out)
            self.model_w_percep = models.Model(inp,outputs = [out,percep_out],name = "SR_w_percep")

    def create_patch(self,n,f,name,tensor):
        x = layers.Conv2D(filters=n,kernel_size=(f,f),name = name,
                          padding='valid',use_bias=True)(tensor)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        return x

    def create_model(self,n1,f1,n2,f2,f3):
        inp = layers.Input(shape = (None,None,3)) # input

        x = self.create_patch(n1,f1,'patch_extraction',inp)

        x = self.create_patch(n2,f2,'nonlinear_mapping',x)

        # reconstruction needs none of the relu stuff
        out = layers.Conv2D(filters=3,kernel_size=(f3,f3),name ='reconstruction',
                          padding='valid',use_bias=True,activation='sigmoid')(x)

        return models.Model(inp,out,name = "SR")