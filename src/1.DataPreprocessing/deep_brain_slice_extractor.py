

__author__ = "Adrian Arnaiz-Rodriguez"
__email__ = "aarnaizr@uoc.edu"
__version__ = "1.0.0"


import warnings
warnings.filterwarnings("ignore")

import pickle as pkl
import glob
import numpy as np
import pandas as pd
from deepbrain import Extractor

import warnings
warnings.filterwarnings("default")




class DeepBrainSliceExtractor:

    def __init__(self, volume_folder:str, pretrained:bool = False, img_data = None):

        self.volume_folder = volume_folder
        self.all_volume_files = glob.glob(self.volume_folder)
        self.pretrained = pretrained
        self.img_data = img_data

        if self.pretrained:
            if isinstance(img_data, str):
                self.path_img_data = img_data
                with open(path_img_data, 'rb') as handle:
                    img_data = pkl.load(handle)

            assert(isinstance(img_data, pd.DataFrame))

    def fit(self):
        if self.pretrained:
            raise Exception("Brain data already extracted on img_data. For fitting, use 'pretrained'=False and img_data=None (Default)")

    def transform(self):
        pass


if __name__ == "__main__":
    print('--')


            
        
