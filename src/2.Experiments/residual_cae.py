
__author__ = "Adrian Arnaiz-Rodriguez"
__email__ = "aarnaizr@uoc.edu"
__version__ = "1.0.0"


from tensorflow import Tensor
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, ReLU, BatchNormalization, Add, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

def relu_bn(inputs: Tensor, name='RB') -> Tensor:
    y = BatchNormalization(name=name+'inner_BN')(inputs) 
    return ReLU(name=name+'_innerReLu')(y)

def residual_block(x: Tensor, filters, ks = (3,3), stride = 2, name='RB'):
    y = Conv2D(filters= filters,
               kernel_size= ks,
               strides= stride,              
               padding="same",
               name=name+'_C1')(x)
    
    y = relu_bn(y, name=name)
    
    y = Conv2D(filters= filters,
               kernel_size= ks,
               strides= 1,               
               padding="same",
               name=name+'_C2')(y)
    y = BatchNormalization(name=name+'_BN')(y) 
    
    if stride !=1:
        x = Conv2D(filters = filters,
                   kernel_size= (1,1),
                   strides= stride,
                   padding="same",
                   name=name+'_CAdjust')(x)
        x = BatchNormalization(name=name+'_BNAdjust')(x) 
    
    y = Add(name=name+'_ResSUM')([x,y])
    y =  ReLU(name=name+'_ReLu')(y)
    return y

def upsampling_block(x: Tensor, filters, ks=(3,3), name='UP'):
    '''y = Conv2D(filters, ks, activation='relu', padding='same', name=name+'_C1')(x) #474k
    y = BatchNormalization(name=name+'_BN')(y) 
    y = UpSampling2D((2, 2), name=name+'_Up')(y)'''
    
    y = Conv2DTranspose(filters, ks, strides=(2,2), padding='same', name=name+'_C1')(x) #474k
    y = BatchNormalization(name=name+'_BN')(y) 
    y =  ReLU(name=name+'_ReLu')(y)
    return y
    

def build_res_encoder(input_shape, batch_size=None):
    #INPUT
    input_img = Input(shape = input_shape, batch_size=batch_size)
    
    #ENCODER
    x = Conv2D(32, (3,3), strides= 2, padding="same", name='Conv1')(input_img) #/2*/2*32
    x = residual_block(x, 64, name='RB1') #64*64*64
    x = residual_block(x, 64, stride=1, name='RB2_same_dim') #/4*/4*64
    x = MaxPooling2D((2,2))(x) #/8*/8*64
    x = residual_block(x, 128, stride=2, name='RB3') #/16*/16*128
    
    #DECODER
    y = upsampling_block(x,64, name='UP1') #/8*/8*64
    y = upsampling_block(y,32, name='UP2') #/4*/4*32
    y = upsampling_block(y,32, name='UP3') #/2*/2*16
    y = upsampling_block(y,32, name='UP4') #1.*1.*8
    decoded = Conv2D(1, (3,3), activation='sigmoid', padding='same')(y)
    return Model(input_img, decoded)



