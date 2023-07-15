import tensorflow as tf

from tensorflow.keras import layers, backend as K

from src.Utils.utils import distance_to_similarity
from src.VisualConcept.vcl_interpretation import closest_patch_to_vc


class VCL(layers.Layer):
    def __init__(self, num_vcl=80, p_h=2, p_w=2, **kwargs):
        """
        :param num_vcl: the number of the VC for the VCL = num_classes * vc_per_class
        """
        super(VCL, self).__init__(**kwargs)
        self.num_vc = num_vcl
        self.p_h = p_h
        self.p_w = p_w

    def build(self, input_shape):
        """
        Create a trainable weight variable for this layer.
        :param input_shape: the shape of the input tensor [batch_size, height, width, depth]
        """
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.p_h, self.p_w, int(input_shape[-1]), self.num_vc),
                                      initializer='uniform',
                                      trainable=True)

        self.ones = K.ones_like(self.kernel)

        return super(VCL, self).build(input_shape)

    def call(self, inputs):
        """
        Output always positive (distance --> similarity)
        :param inputs: input tensor [batch_size, height, width, depth]
        """
        batch_size = K.shape(inputs)[0]
        height = K.shape(inputs)[1]
        image_height = 224
        inputs_square = inputs ** 2
        inputs_patch_sum = K.conv2d(inputs_square, kernel=self.ones, padding='same')
        p2 = self.kernel ** 2
        p2 = K.tile(K.expand_dims(K.sum(p2, axis=2), axis=0), [batch_size, 1, 1, 1])
        if self.p_h != 1 and self.p_w != 1:
            p2 = K.expand_dims(K.expand_dims(K.sum(p2, axis=[1, 2]), axis=1), axis=1)
        xp = K.conv2d(inputs, kernel=self.kernel, padding='same')
        intermediate_result = - 2 * xp + p2  # use broadcast
        # get the distances for each visual concept with convolutional patch output
        self.distances = K.relu(inputs_patch_sum + intermediate_result)
        min_distances = -K.pool2d(-self.distances, pool_size=(self.distances.shape[1], self.distances.shape[2]))
        min_distances = K.reshape(min_distances, [-1, self.num_vc])
        # The activations are the similarity values for the set of visual concepts for each image in the batch
        vcl_activations = distance_to_similarity(min_distances)
        # the vc_weight is the actual visual concept representation that we are learning
        self.vc_weight = tf.tile(K.expand_dims(self.kernel, axis=0), [batch_size, 1, 1, 1, 1])
        # Get the coordinates of the patches in the scale of the image (x, y)
        x_y = closest_patch_to_vc(batch_size, self.num_vc, self.distances)
        # the tf.minimum helps not to not exceed the limit of the image boundary
        # height and width are equal
        self.coordinates = tf.minimum(tf.concat([x_y, x_y + self.p_h], axis=2), height)
        # up-sample the coordinates to the image size.
        self.coordinates = tf.cast(self.coordinates * (image_height // height), dtype=tf.int32)
        return [vcl_activations, self.vc_weight, inputs, self.distances, self.coordinates]

    def compute_output_shape(self, input_shape):
        return [(input_shape[0], self.num_vc), self.vc_weight.shape, input_shape, self.distances.shape, self.coordinates.shape]

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_vc': self.num_vc,
            'p_h': self.p_h,
            'p_w': self.p_w,
        })
        return config
