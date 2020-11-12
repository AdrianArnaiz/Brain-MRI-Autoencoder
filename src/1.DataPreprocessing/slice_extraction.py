""" Python script for extract slices from IXI volumes.
We will use DeepBrainSliceExtractor class
"""

__author__ = "Adrian Arnaiz-Rodriguez"
__email__ = "aarnaizr@uoc.edu"

# Path improvement configuration
from os.path import dirname
import sys
import numpy as np
import pickle as pkl


script_path = dirname(__file__)
sys.path.append(script_path)

from deep_brain_slice_extractor import DeepBrainSliceExtractor


with open(script_path+'/deepbrain_image_data.pickle', 'rb') as f:
    db_image_data = pkl.load(f)

with open(script_path+'/../2.Experiments/data_test_volumes_df.pkl', 'rb') as f:
    test_vols = pkl.load(f)

with open(script_path+'/../2.Experiments/data_train_val_volumes_df.pkl', 'rb') as f:
    train_val_vols = pkl.load(f)

test_vols = test_vols.IXI_ID.values
train_val_vols = train_val_vols.IXI_ID.values

se = DeepBrainSliceExtractor(volume_folder = script_path+'/../IXI-T1/*.gz',
                             save_img_path = script_path+'/../IXI-T1/img/',
                             pretrained=True, 
                             img_data=db_image_data,
                             trainval_ids=train_val_vols,
                             test_ids=test_vols)

se.transform()