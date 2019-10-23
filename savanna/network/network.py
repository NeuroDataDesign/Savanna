from savanna.inference.conv_mf import ConvMF

class Network(object):

	def __init__(self):
		self.layers = []

	def add_convMF(self, SUBTYPE = 'native', NUM_TREES = 1000, TREE_TYPE = 'S-RerF', PATCH_HEIGHT_MIN = 1, PATCH_WIDTH_MIN = 1, PATCH_HEIGHT_MAX = 5, PATCH_WIDTH_MAX = 5):
		self.layers.append(ConvMF(type = SUBTYPE,
									num_trees = NUM_TREES,
									tree_type = TREE_TYPE,
									patch_height_min = PATCH_HEIGHT_MIN,
									patch_width_min = PATCH_WIDTH_MIN,
									patch_height_max = PATCH_HEIGHT_MAX,
									patch_width_max = PATCH_WIDTH_MAX))

	def add_NN(nn_object):
			self.layers.append(nn_object)

	def fit(self, images, labels):
		prev = images
		for layer in self.layers:
			if isinstance(layer, ConvMF):
				prev = layer.fit(prev, labels)
			else:
				 #NEED TO FIX THIS



	def predict(self, images):
		prediction = []

		prev = images
		for i in range(len(self.layers)):
			if i != len(self.layers) - 1:
				prev = self.layers[i].predict(prev)
			else:
				prediction = self.layers[i].final_predict(prev)

		return prediction
