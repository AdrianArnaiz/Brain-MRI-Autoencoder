
__author__ = "Adrian Arnaiz-Rodriguez"
__email__ = "aarnaizr@uoc.edu"
__version__ = "1.0.0"

import glob
import tensorflow as tf
from my_tf_data_loader_optimized import tf_data_png_loader
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
pd.set_option("display.precision", 10)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

class TestMetricWrapper():

    def __init__(self, models_folders_paths, test_files_path):

        self.models_folders_paths = models_folders_paths
        self.test_files_path = test_files_path

        params = {'batch_size': 8,
          'resize':(128,128)
         }
        self.test_ds = tf_data_png_loader(self.test_files_path,
                                          **params,
                                          train=False
                                          ).get_tf_ds_generator()
        
        self.df_t_loss = None
        self.df_v_loss = None
        self.keras_evaluation = None

    def get_training_df(self):
        """
        Returns:
            df_train, df_validation[tuple]: returns losses in train and validation by model and epoch
        """
        df_v_loss = pd.DataFrame(index=range(100))
        df_t_loss = pd.DataFrame(index=range(100))
        for model_folder in self.models_folders_paths:
            #get_csv_logger
            csv = glob.glob(model_folder+'\*.csv')[0]
            name = csv.split('\\')[-1][:-4]
            df = pd.read_csv(csv,sep=';', float_precision='round_trip')
            df_v_loss[name] = df['val_loss']
            df_t_loss[name] = df['loss']
        
        self.df_t_loss = df_t_loss
        self.df_v_loss = df_v_loss
        return df_t_loss, df_v_loss
    
    def get_min_validation_loss_df(self):
        df = pd.DataFrame((self.df_v_loss.min(),self.df_v_loss.apply(lambda x: x.argmin()))).transpose().sort_values(0)
        return df.rename(columns={0:'Val_loss', 1:'Epoch'})

    def plot_val_loss(self, ylimit=0.008, epochs=100, rolling_window=1):
        df = self.df_v_loss.fillna(self.df_v_loss.min())
        df.rolling(rolling_window, axis=0).sum().plot(ylim=(0,ylimit), xlim=(0,epochs)).legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    
    def get_keras_evaluation(self, return_type='dict', verbose=2):
        keras_evaluation = {}
        for model_folder in self.models_folders_paths:
            model_path = glob.glob(model_folder+'\\*.h5')[0]
            model_trained = tf.keras.models.load_model(model_path)
            model_name = model_path.split('\\')[-1][:-3]
            print(model_name, end=' - ')
            result = model_trained.evaluate(self.test_ds, verbose=verbose)
            result = result if isinstance(result, list) else [result]
            keras_evaluation[model_name] = dict(zip(model_trained.metrics_names, result))

        self.keras_evaluation  = keras_evaluation

        if return_type=='dict': 
            return keras_evaluation
        else:
            return pd.DataFrame.from_dict(keras_evaluation, orient='index')

    def get_custom_evaluation(self, verbose=False, return_type='dict'):
        """Returns mean and std of MSE, DSSIM and PSNR of all models on test set

        Args:
            verbose (bool, optional): verbose.
            return_type(str, optional): data type of return.

        Returns:
            custom_evaluation[dict|type]: dictionary or dataframe with metric of test evaluations
        """
        custom_evaluation = dict()
        for model_folder in self.models_folders_paths:
            model_path = glob.glob(model_folder+'\\*.h5')[0]
            model = tf.keras.models.load_model(model_path)
            model_name = model_path.split('\\')[-1][:-3]
            if verbose: print(model_name, end=' - ')

            #get predicted images
            predicted = model.predict(self.test_ds)
            i=0
            mse_metrics = []
            dssim_metrics = []
            psnr_metrics = []
            for batchx, batchy in self.test_ds:
                for x,y in zip(batchx, batchy):
                    mse_metrics.append(self._mserror(y, predicted[i]).numpy())
                    dssim_metrics.append(self._dssim(y, predicted[i]).numpy())
                    psnr_metrics.append(self._psnr(y, predicted[i]).numpy())
                    i+=1

            custom_evaluation[model_name]['mse_mean'] = mean_mse = np.mean(mse_metrics)
            custom_evaluation[model_name]['mse_std'] = std_mse = np.std(mse_metrics)

            custom_evaluation[model_name]['dssim_mean'] = mean_dssim = np.mean(dssim_metrics)
            custom_evaluation[model_name]['dssim_std'] = std_dssim = np.std(dssim_metrics)

            custom_evaluation[model_name]['psnr_mean'] = mean_psnr = np.mean(psnr_metrics)
            custom_evaluation[model_name]['psnr_std'] = std_psnr =np.std(psnr_metrics)

            if verbose:
                print( "MSE: {:.2e}+-{:.2e} - DSSIM: {:.2e}+-{:.2e} - PSNR: {:.2e}+-{:.2e}".format(mean_mse, std_mse,
                                                                                            mean_dssim, std_dssim,
                                                                                            mean_psnr, std_psnr
                                                                                            ))
                print()

        self.custom_evaluation = custom_evaluation
        if return_type=='dict': 
            return custom_evaluation
        else:
            return pd.DataFrame.from_dict(custom_evaluation, orient='index')

    def plot_custom_metrics(self,figsize=(20,5)):
        fig, axs = plt.subplots(1,3, figsize=figsize)
        df = pd.DataFrame.from_dict(self.custom_evaluation, orient='index')
        colors = plt.cm.Paired(np.arange(len(df)))
        for i,m in enumerate(['mse','dssim','psnr']):
            df[m+'_mean'].plot(ax = axs[i], kind='bar',
                                            yerr = df[m+'_std'],
                                            color = colors)
        return fig

    def _dssim(self, x, y):
        """
        We calculate the Structural Dissimilarity between 2 images.
        """
        return tf.math.divide(tf.subtract(1,tf.image.ssim(x, y, max_val=1.0)), 2)

    def _psnr(self, x, y):
        return tf.image.psnr(x, y, max_val=1.0)

    def _mserror(self, x, y):
        return tf.math.reduce_mean(tf.keras.losses.MSE(x, y))




