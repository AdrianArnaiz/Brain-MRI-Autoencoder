
__author__ = "Adrian Arnaiz-Rodriguez"
__email__ = "aarnaizr@uoc.edu"
__version__ = "1.0.0"

from tensorflow.config.experimental import list_physical_devices, set_memory_growth
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import MSE
from tensorflow.keras.utils import plot_model
from tensorflow import math as tfmath
from tensorflow import image as tfimage
import glob
import os
import random
import time
#My modules and classes
from residual_cae import build_res_encoder
from residual_cae_myronenko import build_myronenko_cae
from skip_connection_cae import build_skcon_cae
from res_skip_cae import build_res_skip_cae
#Data Loader
from my_tf_data_loader_optimized import tf_data_png_loader

#Custom tf execution
physical_devices = list_physical_devices('GPU')
set_memory_growth(physical_devices[0], True)

#EXPERIMENT CONFIGURABLE OPTIONS
NETWORK_ARCHITECTURE = 'res_skip_cae' #See architecture options
AUGMENT = True
METRIC = 'DSSIM' #See loss options
KERNEL_REGULARIZATION = False #L2
REDUCE_LR_PLATEAU = True #Min_improvement dynamic satted dependeds on METRIC used for loss
BUILDING_BLOCK = 'full_pre' #only relevant in small_res_cae - Se block options

MODEL_NAME = NETWORK_ARCHITECTURE+'_'+METRIC

EPOCHS = 100
BATCH_SIZE = 32
train_percentage = 0.85
INPUT_SHAPE = (128,128)

#############################
# Check experiment options

block_options = ['original',
                 'full_pre'
]
architecure_options = ['small_res_cae',
                       'myronenko_cae',
                       'skip_con_cae',
                       'res_skip_cae'
]

def DSSIM(y_true, y_pred):
    return tfmath.divide(tfmath.subtract(1.0,tfimage.ssim(y_true, y_pred, max_val=1.0)),2.0)

def PSNR(y_true, y_pred):
    return tfimage.psnr(y_true, y_pred, max_val=1.0)

loss_options = ['MSE',
                'DSSIM',
                'PSNR'
]

assert METRIC in loss_options,'Loss does not belong to the possible ones'
loss_function = eval(METRIC)
loss_options.remove(METRIC)
loss_options = [eval(i) for i in loss_options]

assert NETWORK_ARCHITECTURE in architecure_options,'Network does not belong to the possible ones'
assert BUILDING_BLOCK in block_options,'Bulinding block not implemented'

##########################
#Results PATH
reduce_lr_str = '_LRPlat' if REDUCE_LR_PLATEAU else '_NoPlat'
kreg_str = '_L2KReg' if KERNEL_REGULARIZATION else '_NoKReg'
block_str = '_'+BUILDING_BLOCK if NETWORK_ARCHITECTURE=='small_res_cae' else ''
augment_str = '_AUG' if AUGMENT else ''
MODEL_NAME+= block_str+augment_str+kreg_str+reduce_lr_str

RES_PATH = 'results'+os.path.sep+MODEL_NAME+'_T'+time.strftime('%d_%m_%y__%H_%M') 
if not os.path.exists(RES_PATH):
    os.mkdir(RES_PATH) 

########################
#Data Splitting
TRAIN_img_PATH = '..'+os.path.sep+'IXI-T1'+os.path.sep+'PNG'+os.path.sep+'train_val_folder'+os.path.sep+'train_and_val'
TEST_img_PATH = '..'+os.path.sep+'IXI-T1'+os.path.sep+'PNG'+os.path.sep+'test_folder'+os.path.sep+'test'

#Load train paths
trainval_img_files = glob.glob(TRAIN_img_PATH+os.path.sep+'*.png')
random.shuffle(trainval_img_files)

#Split train_val dataset
lim = int(len(trainval_img_files)*train_percentage)
train_img_files = trainval_img_files[:lim]
validation_img_files = trainval_img_files[lim:]

#Create data loaders
params = {'batch_size': BATCH_SIZE,
          'cache':False,
          'shuffle_buffer_size':1000,
          'resize':INPUT_SHAPE
         }
#train         
train_loader = tf_data_png_loader(train_img_files, **params, augment=AUGMENT)
train_ds = train_loader.get_tf_ds_generator()
#validation
validation_loader = tf_data_png_loader(validation_img_files, **params, augment=False)
validation_ds = validation_loader.get_tf_ds_generator()

#Train parameters for model.fit with generators
STEP_SIZE_TRAIN = len(train_img_files) // train_loader.batch_size
STEP_SIZE_VALID = len(validation_img_files) // validation_loader.batch_size

###############################
#Callbacks Parameters
if METRIC == 'MSE':
    stopping_min_delta = 2e-7
    reducer_min_delta = 1e-7
elif METRIC == 'DSSIM':
    stopping_min_delta = 5e-5
    reducer_min_delta = 2e-5
#Callbacks
my_callbacks = [CSVLogger(RES_PATH+os.path.sep+MODEL_NAME+'.csv', separator=";", append=False),
                ModelCheckpoint(filepath=RES_PATH+os.path.sep+MODEL_NAME+'.h5', #.{epoch:02d}-{val_loss:.2f}
                                monitor='val_loss',
                                mode='min',
                                save_best_only=True),
                EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20, min_delta=stopping_min_delta)
                ]
#Learninrg Rate reducer
if REDUCE_LR_PLATEAU:
    my_callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                          patience=4, min_lr=1e-7, 
                                          min_delta=reducer_min_delta,
                                          verbose=1))

#MODEL FIT
if NETWORK_ARCHITECTURE == 'small_res_cae':
    autoencoder =  build_res_encoder(INPUT_SHAPE+(1,), block_type=BUILDING_BLOCK, ker_reg=KERNEL_REGULARIZATION) #,  params.get('batch_size'))
elif NETWORK_ARCHITECTURE == 'myronenko_cae':
    autoencoder =  build_myronenko_cae(INPUT_SHAPE+(1,), ker_reg=KERNEL_REGULARIZATION)
elif NETWORK_ARCHITECTURE == 'skip_con_cae':
    autoencoder = build_skcon_cae(INPUT_SHAPE+(1,), ker_reg=KERNEL_REGULARIZATION)
elif NETWORK_ARCHITECTURE == 'res_skip_cae':
    autoencoder = build_res_skip_cae(INPUT_SHAPE+(1,), block_type=BUILDING_BLOCK, ker_reg=KERNEL_REGULARIZATION)
else:
    raise('Architecture not implemented')

#Compile, save diagram and fit
autoencoder.compile(loss=loss_function, 
                    optimizer=RMSprop(),
                    metrics=loss_options)
plot_model(autoencoder, to_file=RES_PATH+os.path.sep+MODEL_NAME+".png", show_shapes=True, show_layer_names=True, rankdir="TD")
history = autoencoder_train = autoencoder.fit(train_ds,
                                              epochs=100,
                                              #batch_size=train_loader.batch_size,
                                              steps_per_epoch = STEP_SIZE_TRAIN,
                                              validation_data = validation_ds, 
                                              validation_steps = STEP_SIZE_VALID,
                                              verbose=1,
                                              callbacks = my_callbacks,
                                              max_queue_size = 50
                                             )