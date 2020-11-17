
__author__ = "Adrian Arnaiz-Rodriguez"
__email__ = "aarnaizr@uoc.edu"
__version__ = "1.0.0"

from tensorflow.config.experimental import list_physical_devices, set_memory_growth
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import MSE
from tensorflow.keras.utils import plot_model
import glob
import os
import random
import time
#My modules and classes
from residual_cae import build_res_encoder
from residual_cae_myronenko import build_myronenko_cae
from my_tf_data_loader_optimized import tf_data_png_loader
#Custom tf execution
physical_devices = list_physical_devices('GPU')
set_memory_growth(physical_devices[0], True)


NETWORK_ARCHITECTURE = 'myronenko_cae' # 'small_res_cae', 'myronenko_cae'
MODEL_NAME = 'Myronenko_ls128_MinMax'
EPOCHS = 100
BATCH_SIZE = 32
train_percentage = 0.85
INPUT_SHAPE = (128,128)

##########################
RES_PATH = 'results'+os.path.sep+MODEL_NAME+'_e'+str(EPOCHS)+'_b'+str(BATCH_SIZE)+'_is'+str(INPUT_SHAPE[0])+'_T'+time.strftime('%d_%m_%y__%H_%M') 
if not os.path.exists(RES_PATH):
    os.mkdir(RES_PATH) 

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
train_loader = tf_data_png_loader(train_img_files, **params)
train_ds = train_loader.get_tf_ds_generator()
#validation
validation_loader = tf_data_png_loader(validation_img_files, **params)
validation_ds = validation_loader.get_tf_ds_generator()

#Train parameters for model.fit with generators
STEP_SIZE_TRAIN = len(train_img_files) // train_loader.batch_size
STEP_SIZE_VALID = len(validation_img_files) // validation_loader.batch_size

###############################
#Callbacks
my_callbacks = [CSVLogger(RES_PATH+os.path.sep+MODEL_NAME+'_csvlogger.csv', separator=";", append=False),
                ModelCheckpoint(filepath=RES_PATH+os.path.sep+MODEL_NAME+'_model.h5', #.{epoch:02d}-{val_loss:.2f}
                                monitor='val_loss',
                                mode='min',
                                save_best_only=True),
                EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15, min_delta=1e-7)
                ]                

#MODEL FIT
if NETWORK_ARCHITECTURE == 'small_res_cae':
    autoencoder =  build_res_encoder(INPUT_SHAPE+(1,), block_type=None) #,  params.get('batch_size'))
elif NETWORK_ARCHITECTURE == 'myronenko_cae':
    autoencoder =  build_myronenko_cae(INPUT_SHAPE+(1,))
else:
    raise('Architecture not implemented')

autoencoder.compile(loss=MSE, optimizer=RMSprop())
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