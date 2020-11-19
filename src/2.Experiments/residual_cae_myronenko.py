
__author__ = "Adrian Arnaiz-Rodriguez"
__email__ = "aarnaizr@uoc.edu"
__version__ = "1.0.0"


from tensorflow import Tensor
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, ReLU, BatchNormalization, Add, SpatialDropout2D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

def relu_bn(inputs: Tensor, name='RB') -> Tensor:
    y = BatchNormalization(name=name+'inner_BN')(inputs) 
    return ReLU(name=name+'_innerReLu')(y)


def full_pre_residual_block(x: Tensor, filters, ks = (3,3), stride = 1, name='FPRB', ker_reg=None):
    """Residual block explained in [1]
    https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035
    [1] K. He, X. Zhang, S. Ren, and J. Sun. Identity Mappings in Deep Residual Networks. arXiv preprint arXiv:1603.05027v3,2016.
    Args:
        x (Tensor): [description]
        filters ([type]): [description]
        ks (tuple, optional): [description]. Defaults to (3,3).
        stride (int, optional): [description]. Defaults to 1.
        name (str, optional): [description]. Defaults to 'FPRB'.

    Returns:
        [type]: [description]
    """
    input_tensor = Conv2D(filters=filters,
                          kernel_size=(1,1),
                          kernel_regularizer=ker_reg,
                          strides=1)(x)

    y = relu_bn(x, name=name+'_ReLu1')
    y = Conv2D(filters= filters,
               kernel_size= ks,
               strides= stride,              
               padding="same",
               kernel_regularizer=ker_reg,
               name=name+'_C1')(y)
     
    y = relu_bn(y, name=name+'_ReLu2')
    y = Conv2D(filters= filters,
               kernel_size= ks,
               strides= 1,               
               padding="same",
               kernel_regularizer=ker_reg,
               name=name+'_C2')(y)
    
    y = Add(name=name+'_ResSUM')([input_tensor,y])
    return y


def upsampling_block(x: Tensor, filters, ks=(3,3), name='UP'):    
    y = Conv2DTranspose(filters, ks, 
                        strides=(2,2), 
                        padding='same', 
                        name=name+'_C1')(x) #474k
    y = relu_bn(y, name=name+'_ReLu_UP')
    return y
    

def build_myronenko_cae(input_shape, batch_size=None, ker_reg=False):
    #INPUT
    input_img = Input(shape = input_shape, batch_size=batch_size) #128x128x1

    ker_reg = l2(1e-5) if ker_reg else None
  
    
    #ENCODER
    bname = 'GB'
    #Convolution to input
    x = Conv2D(32, (3,3), strides= 1, padding="same", name='Conv1')(input_img) #128x128x32
    x = SpatialDropout2D(0.1)(x)

    #FPRB1 (Out: 64x64x32)
    y = full_pre_residual_block(x, 32, ker_reg=ker_reg, name=bname+'_32_1') #128x128x32
    y = Conv2D(filters= 32,
               kernel_size= (3,3),
               strides = 2,               
               padding="same",
               kernel_regularizer=ker_reg,
               name='Conv_Downsample_1')(y) #64x64x32

    #GFPRB2 (Out: 32x32x64)
    y = full_pre_residual_block(y, 64, ker_reg=ker_reg, name=bname+'_64_1') #64x64x64
    y = full_pre_residual_block(y, 64, ker_reg=ker_reg, name=bname+'_64_2') #64x64x64
    y = Conv2D(filters= 64,
               kernel_size= (3,3),
               strides = 2,               
               padding="same",
               kernel_regularizer=ker_reg,
               name='Conv_Downsample_2')(y) #32x32x64

    #GFPRB3 (Out: 32x32x64)
    y = full_pre_residual_block(y, 128, ker_reg=ker_reg, name=bname+'_128_1') #32x32x128
    y = full_pre_residual_block(y, 128, ker_reg=ker_reg, name=bname+'_128_2') #32x32x128
    y = Conv2D(filters= 128,
               kernel_size= (3,3),
               strides = 2,               
               padding="same",
               kernel_regularizer=ker_reg,
               name='Conv_Downsample_3')(y) #16x16x128
    
    #y = full_pre_residual_block(y, 256, ker_reg=ker_reg,name=bname+'_256_1') #16x16x256

        
    #DECODER
    y = upsampling_block(y,64, name='UP1') #/4*/4*64
    y = upsampling_block(y,32, name='UP2') #/2*/2*32
    y = upsampling_block(y,16, name='UP3') #1.*/1.*16
    decoded = Conv2D(1, (3,3),
                    activation='sigmoid',
                    padding='same',
                    name='OUTPUT')(y)
    return Model(input_img, decoded)



if __name__ == "__main__":
    from tensorflow.keras.utils import plot_model
    model = build_myronenko_cae((128,128,1))
    print(model.summary())
    plot_model(model, to_file="EXAMPLE.png", show_shapes=True, show_layer_names=True, rankdir="TD")



