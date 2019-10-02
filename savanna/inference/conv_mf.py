import time
from multiprocessing import cpu_count

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from rerf.RerF import fastRerF

class ConvMF(object):
    def __init__(self, num_trees = 1000, tree_type = 'S-RerF'):
        self.num_trees = num_trees;
        self.tree_type = tree_type;
        self.time_taken = {"load": 0, "test_chop": 0, "train": 0, "fit": 0, "train_post": 0, "test": 0, "test_post": 0}

    def fit(self, images, labels):
        #Confused by this... ask Bijan
        def approx_predict_proba_sample_wise(sample):
            return np.array(self.forest.predict_post(sample.tolist())[1] / float(self.num_trees))


        batch_size, length, width, _ = images.shape
        convolved_image = np.zeros((images.shape[0], length, width, 1))


        all_sub_images = sub_images.reshape(batch_size*length*width, -1)
        all_sub_labels = sub_labels.reshape(batch_size*length*width, -1)

        self.forest = rerf.Rerf.FastRerf(X=all_sub_images,
                                         Y=all_sub_labels,
                                         forestType=self.tree_type,
                                         trees=self.num_trees,
                                         numCores=cpu_count() - 1,
                                         imageHeight=length,
                                         imageWidth=width,
                                         patchHeightMin=1,
                                         patchWidthMin=1,
                                         patchHeightMax=3,
                                         patchWidthMax=3,)
        #Is this necessary
        for i in range(length):
            for j in range(width):
                MF_image[:, i, j] = np.array([approx_predict_proba_sample_wise(
                    sample) for sample in sub_images[:, i, j]])[..., np.newaxis]

        return MF_image


    def predict(self, images):
        if not self.forest:
            raise Exception("Should fit training data before  predicting")

        batch_size, length, width, _ = images.shape
        kernel_predictions = np.zeros((images.shape[0], length, width, 1))

        for i in range(length):
            for j in range(width):
                def approx_predict_proba_sample_wise(sample):
                    return np.array(self.forest.predict_post(sample.tolist())[1] / float(self.num_trees))
                kernel_predictions[:, i, j] = np.array([approx_predict_proba_sample_wise(
                    sample) for sample in images[:, i, j]])[..., np.newaxis]

        return kernel_predictions
