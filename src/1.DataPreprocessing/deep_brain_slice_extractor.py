

__author__ = "Adrian Arnaiz-Rodriguez"
__email__ = "aarnaizr@uoc.edu"
__version__ = "1.0.0"


import warnings
warnings.filterwarnings("ignore")

import pickle as pkl
import glob
import numpy as np
import pandas as pd
import nibabel as nib
from deepbrain import Extractor
import os
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("default")




class DeepBrainSliceExtractor:

    def __init__(self, 
                volume_folder:str, 
                save_img_path:str,
                pretrained:bool = False, 
                img_data = None, 
                trainval_ids = None, 
                test_ids = None,
                out_format = 'npy'):
        """[summary]

        Args:
            volume_folder (str): [description]
            save_img_path (str): [description]
            pretrained (bool, optional): [description]. Defaults to False.
            img_data ([DataFrame], optional): [description]. Defaults to None.
            trainval_ids ([iterable:int], optional): [description]. Defaults to None.
            test_ids ([iterable:int], optional): [description]. Defaults to None.
        """

        self.volume_folder = volume_folder
        self.save_img_path = save_img_path

        self.all_volume_files = glob.glob(self.volume_folder)
        self.pretrained = pretrained
        self.img_data = img_data
        self.trainval_ids = trainval_ids
        self.test_ids = test_ids

        self.out_format = out_format

        if self.pretrained:
            if isinstance(img_data, str):
                self.path_img_data = img_data
                with open(path_img_data, 'rb') as handle:
                    img_data = pkl.load(handle)

            assert(isinstance(img_data, pd.DataFrame))

    def fit(self):
        if self.pretrained:
            raise Exception("Brain data already extracted on img_data. For fitting, use 'pretrained'=False and img_data=None (Default)")

    def transform(self, verbose = True):
        counttrain, counttest = 0, 0
        
        for f in self.all_volume_files:
            innercount = 0

            name_vol = f.split('\\')[-1][:-7]
            ixi_id = int(name_vol[3:6])

            if ixi_id in self.trainval_ids:
                split = 'train_and_val/'
            elif ixi_id in self.test_ids:
                split = 'test/'
            else:
                raise Exception('Volume DO NOT BELONG to any partition')

            if not os.path.isdir(self.save_img_path+split):
                os.mkdir(self.save_img_path+split)

            vol_np = nib.load(f).get_fdata()
            for id_sag_slice in range(vol_np.shape[2]):
                
                name_slice = name_vol + '_' + str(id_sag_slice)

                brain_q = int(self.img_data[self.img_data['ID']==name_slice]['BRAIN_QUANTITY'])
                if brain_q>3000:
                    innercount += 1
                    img_slice = np.rot90(vol_np[:,:,id_sag_slice])
                    assert(img_slice.shape==(256,256))

                    if self.out_format == 'npy':
                        np.save(self.save_img_path+split+name_slice, img_slice)
                    else:
                        plt.imsave(self.save_img_path+split+name_slice+'.'+self.out_format,
                                   img_slice, format = self.out_format,
                                   cmap='gray')

            if split == 'train_and_val/':
                counttrain += innercount
            else:
                counttest += innercount

            if verbose:
                print(ixi_id,'-', split,'-', name_vol)
                print('\tRelevant vol. slices:', innercount)
                print('\tTotal Train and Val:', counttrain)
                print('\tTotal Test:', counttest)
                print()
                print('--------------') 

            



if __name__ == "__main__":
    print('--')


            
        
