
__author__ = "Adrian Arnaiz-Rodriguez"
__email__ = "aarnaizr@uoc.edu"
__version__ = "1.0.0"


from tensorflow import Tensor
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, ReLU, BatchNormalization, Add, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

def build_skcon_cae(input_shape, batch_size=None, ker_reg=False):
    #INPUT
    input_img = Input(shape = input_shape, batch_size=batch_size) #128x128x1

    ker_reg = l2(1e-5) if ker_reg else None
    
    #ENCODER
    x = Conv2D(32, (3,3), strides= 2, padding="same", kernel_regularizer=ker_reg, name='Conv1')(input_img) #64x64x32
    x64_64_32 = BatchNormalization()(x)
    x =  ReLU(name='eReLu1')(x64_64_32)

    x = Conv2D(64, (3,3), strides= 2, padding="same", kernel_regularizer=ker_reg, name='Conv2')(x) #32x32x64
    x32_32_64 = BatchNormalization()(x)
    x =  ReLU(name='eReLu2')(x32_32_64)

    x = Conv2D(128, (3,3), strides= 2, padding="same", kernel_regularizer=ker_reg, name='Conv3')(x) #16x16x128
    x = BatchNormalization()(x)
    latent =  ReLU(name='eReLu3')(x)

    
    #DECODER
    y = Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', name='DConv3')(latent) #32*32*64
    y = BatchNormalization()(y) 
    y = Add(name='SKIP_CONN1')([y, x32_32_64])
    y =  ReLU(name='dReLu3')(y)

    y = Conv2DTranspose(32, (3,3), strides=(2,2), padding='same', name='DConv2')(y) #64*64*32
    y = BatchNormalization()(y) 
    y = Add(name='SKIP_CONN2')([y, x64_64_32])
    y =  ReLU(name='dReLu2')(y)

    y = Conv2DTranspose(16, (3,3), strides=(2,2), padding='same', name='DConv1')(y) #128*128*16
    y = BatchNormalization()(y) 
    y =  ReLU(name='dReLu1')(y)

    decoded = Conv2D(1, (3,3), activation='sigmoid', padding='same', name='DConvCH')(y)
    return Model(input_img, decoded)

if __name__ == "__main__":
    from tensorflow.keras.utils import plot_model
    model = build_skcon_cae((128,128,1))
    print(model.summary())
    plot_model(model, to_file="EXAMPLE.png", show_shapes=True, show_layer_names=True, rankdir="LR")