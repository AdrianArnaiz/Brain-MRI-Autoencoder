import numpy as np
from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=8, dim=(256, 256), n_channels=1, 
                 shuffle=True, std_normalization=False, to_fit=True, f_aug=None):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.samples = len(list_IDs)
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.std_normalization = std_normalization
        self.to_fit = to_fit
        self.f_aug = f_aug
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        batch_x = self.__data_generation(list_IDs_temp)
        
        if self.to_fit:
            batch_x, batch_y = batch_x
        
            return np.stack([
                self._augment(image=x) for x in batch_x
            ], axis=0), np.array(batch_y)
        else:
            return np.stack([
                self._augment(image=x) for x in batch_x
            ], axis=0)


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        
         # Generate data
        for i, ID in enumerate(list_IDs_temp):
            #Load
            img = np.load(ID)
            img = np.expand_dims(img, axis=2)
            #Preprocess Sample
            if self.std_normalization:
                img = (img-img.mean())/img.std()
            
            # Store sample 
            assert img.shape==(256,256,1),"BAD INPUT IMAGE:"+str(img.shape)
            X[i,] = img
            
        if self.to_fit:
            return X, X.copy()
        else:
            return X    
        
    
    def _augment(self, image):
        #image = resize(image, (image.shape[0] // 2, image.shape[1] // 2, 1), anti_aliasing=True)
        #TBD
        if self.f_aug:
            image = self.f_aug(image)
        assert image.shape==(256,256,1),"BAD INPUT IMAGE:"+str(image.shape)
        return image