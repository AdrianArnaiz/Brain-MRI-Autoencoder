
__author__ = "Adrian Arnaiz-Rodriguez"
__email__ = "aarnaizr@uoc.edu"
__version__ = "1.0.0"


from tensorflow import Tensor
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, ReLU, BatchNormalization, Add, SpatialDropout2D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

def relu_bn(inputs: Tensor, name='RB') -> Tensor:
    y = BatchNormalization(name=name+'inner_BN')(inputs) 
    return ReLU(name=name+'_innerReLu')(y)

def full_pre_residual_block(x: Tensor, filters, ks = (3,3), stride = 1, name='FPRB'):
    '''
    https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035
    [7]. K. He, X. Zhang, S. Ren, and J. Sun. Identity Mappings in Deep Residual Networks. arXiv preprint arXiv:1603.05027v3,2016.
    '''

    y = BatchNormalization(name=name+'_BN1')(x)
    y = relu_bn(y, name=name+'_ReLu1')
    y = Conv2D(filters= filters,
               kernel_size= ks,
               strides= stride,              
               padding="same",
               name=name+'_C1')(y)
     
    y = BatchNormalization(name=name+'_BN2')(y)
    y = relu_bn(y, name=name+'_ReLu2')
    y = Conv2D(filters= filters,
               kernel_size= ks,
               strides= 1,               
               padding="same",
               name=name+'_C2')(y)
    
    y = Add(name=name+'_ResSUM')([x,y])
    return y


def upsampling_block(x: Tensor, filters, ks=(3,3), name='UP'):    
    y = Conv2DTranspose(filters, ks, strides=(2,2), padding='same', name=name+'_C1')(x) #474k
    y = BatchNormalization(name=name+'_BN')(y) 
    y =  ReLU(name=name+'_ReLu')(y)
    return y
    

def build_myronenco_cae(input_shape, batch_size=None, pooling=False, block_type='original'):
    #INPUT
    input_img = Input(shape = input_shape, batch_size=batch_size) #128x128x1
    
    #ENCODER
    x = Conv2D(32, (3,3), strides= 2, padding="same", name='Conv1')(input_img) #64x64x32
    x = SpatialDropout2D(0.2)(x)

    x = residual_block(x, 64, name=bname+'1') #32x32x64
    x = residual_block(x, 64, stride=1, name=bname+'2_same_dim') #32x32x64
    latent = residual_block(x, 128, stride=2, name=bname+'3') #16x16x128
    
    #DECODER
    y = upsampling_block(latent,64, name='UP1') #/4*/4*64
    y = upsampling_block(y,32, name='UP2') #/4*/4*32
    y = upsampling_block(y,16, name='UP3') #/2*/2*16
    decoded = Conv2D(1, (3,3), activation='sigmoid', padding='same')(y)
    return Model(input_img, decoded)


