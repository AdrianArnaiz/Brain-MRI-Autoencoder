
__author__ = "Adrian Arnaiz-Rodriguez"
__email__ = "aarnaizr@uoc.edu"
__version__ = "1.0.0"


from tensorflow import Tensor
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, ReLU, BatchNormalization, Add, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.regularizers import l2

def relu_bn(inputs: Tensor, name='RB') -> Tensor:
    y = BatchNormalization(name=name+'inner_BN')(inputs) 
    return ReLU(name=name+'_innerReLu')(y)

def original_residual_block(x: Tensor, filters, ks = (3,3), stride = 2, name='RB',ker_reg=None):
    y = Conv2D(filters= filters,
               kernel_size= ks,
               strides= stride,              
               padding="same",
               kernel_regularizer=ker_reg,
               name=name+'_C1')(x)
    y = relu_bn(y, name=name)
    
    y = Conv2D(filters= filters,
               kernel_size= ks,
               strides= 1,               
               padding="same",
               kernel_regularizer=ker_reg,
               name=name+'_C2')(y)
    y = BatchNormalization(name=name+'_BN')(y) 
    
    if stride !=1:
        x = Conv2D(filters = filters,
                   kernel_size= (1,1),
                   strides= stride,
                   padding="same",
                   kernel_regularizer=ker_reg,
                   name=name+'_CAdjust')(x)
    
    y = Add(name=name+'_ResSUM')([x,y])
    y =  ReLU(name=name+'_ReLu')(y)
    return y

def full_pre_residual_block(x: Tensor, filters, ks = (3,3), stride = 2, name='FPRB', ker_reg=None):
    '''
    https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035
    [7]. K. He, X. Zhang, S. Ren, and J. Sun. Identity Mappings in Deep Residual Networks. arXiv preprint arXiv:1603.05027v3,2016.
    '''
    y = relu_bn(x, name=name+'_BN_R1')
    y = Conv2D(filters= filters,
               kernel_size= ks,
               strides= stride,              
               padding="same",
               kernel_regularizer= ker_reg,
               name=name+'_C1')(y)
     
    y = relu_bn(y, name=name+'_BN_R2')
    y = Conv2D(filters= filters,
               kernel_size= ks,
               strides= 1,               
               padding="same",
               kernel_regularizer=ker_reg,
               name=name+'_C2')(y)
    
    if stride !=1:
        x = Conv2D(filters = filters,
                   kernel_size= (1,1),
                   strides= stride,
                   padding="same",
                   kernel_regularizer=ker_reg,
                   name=name+'_CAdjust')(x) 
    
    y = Add(name=name+'_ResSUM')([x,y])
    return y


def upsampling_block(x: Tensor, filters, ks=(3,3), name='UP'):
    '''y = Conv2D(filters, ks, activation='relu', padding='same', name=name+'_C1')(x) #474k
    y = BatchNormalization(name=name+'_BN')(y) 
    y = UpSampling2D((2, 2), name=name+'_Up')(y)'''
    
    y = Conv2DTranspose(filters, ks, strides=(2,2), padding='same', name=name+'_C1')(x) #474k
    y = relu_bn(y, name=name+'_BN_RUP')
    return y
    

def build_res_encoder(input_shape, batch_size=None, block_type='original', ker_reg = False):
    #INPUT
    input_img = Input(shape = input_shape, batch_size=batch_size) #128x128x1
    
    if block_type == 'original':
        residual_block = original_residual_block
        bname = 'RB'
    elif block_type == 'full_pre':
        residual_block = full_pre_residual_block
        bname = 'FP_RB'
    else:
        raise Exception('Not implemented block')

    ker_reg = l2(1e-5) if ker_reg else None
    print('-------------')
    print(ker_reg)
    print('-------------')

    #ENCODER
    x = Conv2D(32, (3,3), strides= 2, padding="same", name='Conv1', kernel_regularizer=ker_reg)(input_img) #64x64x32
    x = residual_block(x, 64, name=bname+'1', ker_reg=ker_reg) #32x32x64
    x = residual_block(x, 64, stride=1, name=bname+'2_same_dim', ker_reg=ker_reg) #32x32x64
    latent = residual_block(x, 128, stride=2, name=bname+'3', ker_reg=ker_reg) #16x16x128
    
    #DECODER
    y = upsampling_block(latent,64, name='UP1') #/4*/4*64
    y = upsampling_block(y,32, name='UP2') #/2*/2*32
    y = upsampling_block(y,16, name='UP3') #1*1*16
    decoded = Conv2D(1, (3,3), activation='sigmoid', padding='same', kernel_regularizer=ker_reg)(y)
    return Model(input_img, decoded)



