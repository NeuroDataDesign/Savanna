import time
from multiprocessing import cpu_count

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from rerf.rerfClassifier import rerfClassifier

class ConvMF(object):
    def __init__(self, type = 'native', kernel_size = 5, stride = 2, num_trees = 1000, num_split_trees = 100, tree_type = 'S-RerF', patch_height_min = 1, patch_width_min = 1, patch_height_max = 5, patch_width_max = 5, max_depth = None):
        self.kernel_size = kernel_size
        self.stride = stride
        self.num_trees = num_trees;
        self.tree_type = tree_type;
        self.type = type;
        self.patch_height_min = patch_height_min
        self.patch_height_max = patch_height_max
        self.patch_width_max = patch_width_max
        self.patch_width_min = patch_width_min
        self.num_split_trees = num_split_trees
        self.max_depth = max_depth
        self.time_taken = {"load": 0, "test_chop": 0, "train": 0, "fit": 0, "train_post": 0, "test": 0, "test_post": 0}

    def _convolve_chop(self, images, labels=None, flatten=False):

        batch_size, in_dim, _, num_channels = images.shape

        #20 x 20


        out_dim = int((in_dim - self.kernel_size) / self.stride) + 1  # calculate output dimensions

        # create matrix to hold the chopped images
        out_images = np.zeros((batch_size, out_dim, out_dim,
                               self.kernel_size, self.kernel_size, num_channels))
        out_labels = None

        curr_y = out_y = 0
        # move kernel vertically across the image
        while curr_y + self.kernel_size <= in_dim:
            curr_x = out_x = 0
            # move kernel horizontally across the image
            while curr_x + self.kernel_size <= in_dim:
                # chop images
                out_images[:, out_x, out_y] = images[:, curr_x:curr_x +
                                                     self.kernel_size, curr_y:curr_y+self.kernel_size, :]
                curr_x += self.stride
                out_x += 1
            curr_y += self.stride
            out_y += 1

        if flatten:
            out_images = out_images.reshape(batch_size, out_dim, out_dim, -1)

        if labels is not None:
            out_labels = np.zeros((batch_size, out_dim, out_dim))
            out_labels[:, ] = labels.reshape(-1, 1, 1)

        return out_images, out_labels



    def fit(self, images, labels):
        MF_image = np.zeros(5)
        self.num_classes = len(np.unique(labels))
        if self.type == 'native':
            batch_size, length, width = images.shape


            reshaped_images = images.reshape(batch_size, length*width)


            self.forest = rerfClassifier(projection_matrix=self.tree_type,
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

        elif self.type == 'kernel_patches':
            sub_images, sub_labels = self._convolve_chop(images, labels=labels, flatten=True)
            batch_size, out_dim, _ = sub_images.shape
            MF_image = np.zeros((images.shape[0], out_dim, out_dim, self.num_classes))
            self.forest = np.zeros((out_dim, out_dim), dtype=np.int).tolist()

            for i in range(out_dim):
                for j in range(out_dim):
                    self.forest[i][j] = rerfClassifier(projection_matrix=self.tree_type,
                                                     n_estimators=self.num_trees,
                                                     n_jobs=cpu_count() - 1,
                                                     image_height=self.kernel_size,
                                                     image_width=self.kernel_size,
                                                     patch_height_min=self.patch_height_min,
                                                     patch_width_min=self.patch_width_min,
                                                     patch_height_max=self.patch_height_max,
                                                     patch_width_max=self.patch_height_min)

                    self.forest[i][j].fit(sub_images[:, i, j], sub_labels[:, i, j])
                    MF_image[:, i, j] = self.forest[i][j].predict_proba(
                        sub_images[:, i, j])[..., 1][..., np.newaxis]

        elif self.type == 'split_forest':
            self.forest = []

            batch_size, length, width = images.shape
            reshaped_images = images.reshape(batch_size, length*width)

            MF_image = np.zeros((batch_size, self.num_trees, self.num_classes))

            for n in range(self.num_trees):
                self.forest.append(rerfClassifier(projection_matrix=self.tree_type,
                                             n_estimators=self.num_split_trees,
                                             n_jobs=cpu_count() - 1,
                                             image_height=length,
                                             image_width=width,
                                             patch_height_min=self.patch_height_min,
                                             patch_width_min=self.patch_width_min,
                                             patch_height_max=self.patch_height_max,
                                             patch_width_max=self.patch_height_min,
                                             max_depth=self.max_depth));
                self.forest[n].fit(reshaped_images, labels);
                MF_image[:,n] = self.forest[n].predict_proba(reshaped_images)

        return MF_image


    def predict(self, images):
        kernel_predictions = []
        if not self.forest:
            raise Exception("Should fit training data before  predicting")

        if self.type == 'native':
            batch_size, length, width = images.shape
            reshaped_images = images.reshape(batch_size, length*width)
            kernel_predictions = np.zeros((images.shape[0], length, width, 1))
            kernel_predictions = self.forest.predict_proba(reshaped_images)

        elif self.type == 'kernel_patches':
            sub_images, _ = self._convolve_chop(images, flatten = True)
            batch_size, out_dim, _ = sub_images.shape
            kernel_predictions = np.zeros((images.shape[0], out_dim, out_dim, self.num_classes))
            for i in range(out_dim):
                for j in range(out_dim):
                    kernel_predictions[:, i, j] = self.forest[i][j].predict_proba(
                            sub_images[:, i, j])

        elif self.type == 'split_forest':
            batch_size, length, width = images.shape
            reshaped_images = images.reshape(batch_size, length*width)
            kernel_predictions = np.zeros((batch_size, self.num_trees, self.num_classes))
            for n in range(self.num_trees):
                kernel_predictions[:,n] = self.forest[n].predict_proba(reshaped_images)

        return kernel_predictions


    def final_predict(self, images):
        if not self.forest:
            raise Exception("Should fit training data before  predicting")

        kernel_predictions = []

        if self.type == 'native':
            batch_size, length, width = images.shape
            reshaped_images = images.reshape(batch_size, length*width)
            kernel_predictions = np.zeros((images.shape[0], length, width, 1))
            kernel_predictions = self.forest.predict(reshaped_images)

        if self.type == 'kernel_patches':
            sub_images, _ = self._convolve_chop(images, flatten = True)
            batch_size, out_dim, _= sub_images.shape
            predictions = np.zeros((images.shape[0], self.num_classes))
            for i in range(out_dim):
                for j in range(out_dim):
                        predictions[:,] = predictions[:,] + self.forest[i][j].predict_proba(
                            sub_images[:, i, j])
            kernel_predictions = np.argmax(predictions, axis = 1)

        if self.type == 'split_forest':
            batch_size, length, width = images.shape
            reshaped_images = images.reshape(batch_size, length*width)
            predictions = np.zeros((batch_size, self.num_classes))
            for n in range(self.num_trees):
                predictions = predictions + self.forest[n].predict_proba(reshaped_images)
            kernel_predictions = np.argmax(predictions, axis = 1)

        return kernel_predictions
