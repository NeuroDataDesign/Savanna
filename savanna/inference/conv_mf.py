import time
from multiprocessing import cpu_count

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from rerf.rerfClassifier import rerfClassifier

class ConvMF(object):
    def __init__(self, type = 'native', num_trees = 1000, tree_type = 'S-RerF', patch_height_min = 1, patch_width_min = 1, patch_height_max = 5, patch_width_max = 5):
        self.num_trees = num_trees;
        self.tree_type = tree_type;
        self.type = type;
        self.patch_height_min = patch_height_min
        self.patch_height_max = patch_height_max
        self.patch_width_max = patch_width_max
        self.patch_width_min = patch_width_min
        self.time_taken = {"load": 0, "test_chop": 0, "train": 0, "fit": 0, "train_post": 0, "test": 0, "test_post": 0}

    def fit(self, images, labels):
        MF_image = np.zeros(5)
        if self.type == 'native':
            batch_size, length, width,_ = images.shape


            reshaped_images = images.reshape(batch_size, length*width)


            self.forest = rerfClassifier(projection_matrix="S-RerF",
                                             n_estimators=self.num_trees,
                                             n_jobs=cpu_count() - 1,
                                             image_height=length,
                                             image_width=width,
                                             patch_height_min=self.patch_height_min,
                                             patch_width_min=self.patch_width_min,
                                             patch_height_max=self.patch_height_max,
                                             patch_width_max=self.patch_height_min)
            self.forest.fit(reshaped_images, labels)
            #Is this necessary
            #for i in range(length):
            #    for j in range(width):
            #        x = 1
            #        MF_image[:, i, j] = np.array([approx_predict_proba_sample_wise(
            #            sample) for sample in images[:, i, j]])[..., np.newaxis]

            MF_image = self.forest.predict_proba(reshaped_images)

        return MF_image


    def predict(self, images):
        if not self.forest:
            raise Exception("Should fit training data before  predicting")

        batch_size, length, width, _ = images.shape
        reshaped_images = images.reshape(batch_size, length*width)
        kernel_predictions = np.zeros((images.shape[0], length, width, 1))

        if self.type == 'native':
            kernel_predictions = self.forest.predict_proba(reshaped_images)

        return kernel_predictions


    def final_predict(self, images):
        if not self.forest:
            raise Exception("Should fit training data before  predicting")

        batch_size, length, width, _ = images.shape
        reshaped_images = images.reshape(batch_size, length*width)
        kernel_predictions = np.zeros((images.shape[0], length, width, 1))

        if self.type == 'native':
            kernel_predictions = self.forest.predict(reshaped_images)

        return kernel_predictions
