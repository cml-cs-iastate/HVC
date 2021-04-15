import tensorflow as tf

from tensorflow.keras import backend as K


def cluster_separation_cost(label, vc_distances):
    """
    take one input tensor at a time and measure the clustering cost with the vc that belong to the same category
    :param label: int label index of the input sample class label       # shape = [1]
    :param vc_distances: TensorFlow vc tensor                           # shape = [height, width, num_vc]
    :return: float clustering and separation cost
    """
    num_vc = tf.shape(vc_distances)[-1]
    num_classes = tf.shape(label)[0]
    vc_per_class = num_vc // num_classes
    # Get starting and ending index of the VC weights
    label = tf.cast(tf.argmax(label), dtype=tf.int32)
    start = label * vc_per_class
    end = (label + 1) * vc_per_class
    cluster_distance = vc_distances[:, :, start:end]
    cluster_cost = tf.reduce_min(cluster_distance)
    return [[cluster_cost]]


def diversity_cost(vc_weights, num_classes):
    """
    :param vc_weights: TensorFlow vc tensor                             # shape = [p_h, p_w, depth, num_vc]
    :param num_classes: TensorFlow integer tensor                       # shape = [1]
    https://stackoverflow.com/questions/37009647/compute-pairwise-distance-in-a-batch-without-replicating-tensor-in-tensorflow
    """
    num_vc = tf.shape(vc_weights)[-1]
    vc_per_class = num_vc // num_classes
    vc_weights_ = tf.reshape(vc_weights, [-1, num_vc])

    def condition(idx_, d_loss):
        return tf.less(idx_, num_vc)

    def body(idx_, d_loss):
        p = tf.transpose(vc_weights_[:, idx_:idx_+vc_per_class])
        r = tf.reduce_sum(p * p, 1)
        # turn r into column vector
        r = tf.reshape(r, [-1, 1])
        c = r - 2 * tf.matmul(p, tf.transpose(p)) + tf.transpose(r)
        # Because of some small values, we get distance values in negative, this just means it equals 0
        c = K.relu(c)
        # Replace the diagonal value of 0 (distance between the vector and itself) with the max value for each row
        cost_ = tf.reduce_min(tf.linalg.set_diag(c, tf.reduce_max(c, axis=1)))
        d_loss = tf.concat([d_loss, [cost_]], axis=0)
        return [tf.add(idx_, vc_per_class), d_loss]

    _, diversity_distance = tf.while_loop(condition,
                                          body,
                                          loop_vars=[tf.constant(0), tf.constant([0.])],
                                          shape_invariants=[tf.constant(0).get_shape(), tf.TensorShape([None])])
    divers_cost = tf.reduce_mean(diversity_distance[1:])
    return divers_cost


def generate_patches(label, inputs_image, coordinates, next_coordinates, gap, n_gap, vcl_activation, p_h, p_w):
    """
    take one input tensor at a time and measure the clustering cost with the vc that belong to the same category
    :param label: int label index of the input sample class label           # shape = [num_labels]
    :param inputs_image: the input image tensor                             # shape = [img_height, img_width, img_depth]
    :param coordinates: the coordinates of the current patch                # shape = [num_vc, 4] 4 for [x, y, x2, y2]
    :param next_coordinates: the (x,y) coordinates of the patch             # shape = [num_vc, 4] 4 for [n_x, n_y, n_x2, n_y2]
    :param gap: the gap of patches of the VCL
    :param n_gap: the gap of the patches of the next VCL
    :param vcl_activation: Tensorflow vc similarity tensor                  # shape = [num_vc]
    :param p_h: height of VC
    :param p_w: width of VC
    """
    num_vc = tf.shape(next_coordinates)[0]
    num_classes = tf.shape(label)[0]
    # Get starting and ending index of the vc weights
    label = tf.cast(tf.argmax(label), dtype=tf.int32)
    vc_per_class = num_vc // num_classes
    start = label * vc_per_class
    end = (label + 1) * vc_per_class
    # Get the coordinates the correspond to the target class
    target_class_next_coordinates = next_coordinates[start:end, :]
    target_class_coordinates = coordinates[start:end, :]
    target_class_vc_activation = vcl_activation[start:end]
    # Get the next_patches and the current coordinates from the input image

    condition = lambda start_, end_, n_p, p: tf.less(start_, end_)

    def body(start_, end_, n_p, p):
        n_t = target_class_next_coordinates[start_]
        n_x1, n_y1, n_x2, n_y2 = n_t[0], n_t[1], n_t[2], n_t[3]
        t = target_class_coordinates[start_]
        x1, y1, x2, y2 = t[0], t[1], t[2], t[3]
        next_patch = inputs_image[n_x1:n_x2, n_y1:n_y2, :]
        patch = inputs_image[x1:x2, y1:y2, :]

        if p_h != 1 and p_w != 1:
            # Solve the problem when the patches are in the borders and don't complete a standard patch size
            def f1(): return tf.tile(tf.reshape(next_patch, [-1, n_gap//2, 3]), [1, 2, 1])
            def f2(): return tf.tile(tf.reshape(next_patch, [n_gap//2, n_gap, 3]), [2, 1, 1])
            def f3(): return next_patch
            next_patch = tf.cond(tf.not_equal(tf.shape(next_patch)[1], n_gap), f1, f3)
            next_patch = tf.cond(tf.not_equal(tf.shape(next_patch)[0], n_gap), f2, f3)
            def f1(): return tf.tile(tf.reshape(patch, [-1, gap//2, 3]), [1, 2, 1])
            def f2(): return tf.tile(tf.reshape(patch, [gap//2, gap, 3]), [2, 1, 1])
            def f3(): return patch
            patch = tf.cond(tf.not_equal(tf.shape(patch)[1], gap), f1, f3)
            patch = tf.cond(tf.not_equal(tf.shape(patch)[0], gap), f2, f3)

        n_p = tf.concat([n_p, tf.expand_dims(next_patch, axis=0)], axis=0)
        p = tf.concat([p, tf.expand_dims(patch, axis=0)], axis=0)
        return [tf.add(start_, 1), end_, n_p, p]

    _, _, next_patches, patches = tf.while_loop(condition,
                                                body,
                                                loop_vars=[tf.constant(0), vc_per_class, tf.zeros(shape=[1, n_gap, n_gap, 3]), tf.zeros(shape=[1, gap, gap, 3])],
                                                shape_invariants=[tf.constant(0).get_shape(), vc_per_class.get_shape(), tf.TensorShape([None, n_gap, n_gap, 3]), tf.TensorShape([None, gap, gap, 3])])
    return [patches[1:], next_patches[1:], target_class_vc_activation]


def membership_cost_fun(patches, next_patches, vc_per_class, target_class_vc_activation, loss_=True):
    """
    :param patches: the patches of the current VC layer                             # shape [vc_per_class, h, w, d]
    :param next_patches: the patches of the next (higher) VC layer                  # shape [vc_per_class, h_next, w_next, next_d]
    :param vc_per_class: int number of vc per class
    :param target_class_vc_activation: the similarity scores of the target class    # shape [vc_per_class]
    :param loss_: if True, this function will be used to calculate the loss
    """
    patches = tf.transpose(patches, perm=[1, 2, 3, 0])
    # Calculate the minimum distance of each patch with all the next patches next PC layer of the same class
    patches_square = next_patches ** 2
    patches_square_sum = K.conv2d(patches_square, kernel=K.ones_like(patches), padding='same')
    current_patches_square = patches ** 2
    current_patches_square = K.tile(K.expand_dims(K.sum(current_patches_square, axis=2), axis=0), [vc_per_class, 1, 1, 1])
    current_patches_square = K.expand_dims(K.expand_dims(K.sum(current_patches_square, axis=[1, 2]), axis=1), axis=1)
    xp = K.conv2d(next_patches, kernel=patches, padding='same')
    intermediate_result = - 2 * xp + current_patches_square
    distances = K.relu(patches_square_sum + intermediate_result)
    # Multiply the distances of each patch with the patch of the parent patch by (1 - similarity_score) of the parent patch vc
    weighted_distances = (distances * (1 - tf.reshape(target_class_vc_activation, shape=[vc_per_class, 1, 1, 1])))
    # global min pooling
    min_distances = -K.pool2d(-distances, pool_size=(distances.shape[1], distances.shape[2]))
    min_distances = K.reshape(min_distances, [-1, vc_per_class])

    if not loss_:
        # link each vc to the parent vc that closest to by membership_distance
        membership_distance = tf.reduce_min(min_distances, axis=0)
        # get the index of the parent vc that's closest to the current vc
        membership_index = tf.argmin(min_distances, axis=0)
        return [[membership_distance], [membership_index]]

    membership_cost = tf.reduce_min(min_distances, axis=1)
    membership_cost = tf.reduce_mean(membership_cost)
    return [membership_cost]


# Tested
def custom_loss(vc_distances, vc_weights, batch_size=tf.constant(32), num_classes=8, inputs_images=None, next_coordinates=None,
                current_coordinates=None, vc_per_class=0, vcl_activations=None, gap=0, n_gap=0, p_h=1, p_w=1):
    """
    the gathering and membership loss functions that take the input samples in the batch with their labels
    :param vc_distances: TensorFlow vc tensor                                   # shape = [batch_size, height, width, num_vc]
    :param vc_weights: TensorFlow vc tensor                                     # shape = [p_h, p_w, depth, num_vc]
    :param batch_size: TensorFlow constant
    :param num_classes:
    :param inputs_images: the inputs images                                     # shape = [batch_size, height, width, depth]
    :param next_coordinates: coordinates information of next PC layer           # shape = [batch_size, num_vc, 4] [x, y, x2, y2]
    :param current_coordinates: coordinates information of current PC layer     # shape = [batch_size, num_vc, 4] [x, y, x2, y2]
    :param vc_per_class: int number of vc per class
    :param vcl_activations: Tnesorflow vc similarity tensor                     # shape = [batch_size, num_vc]
    :param gap: the height/width of the image patch for the current vc
    :param n_gap: the height/width of the image patch for the next vc
    :param p_h: the height of the vc
    :param p_w: the width of the vc
    the coordinates represent the closest patch in the input sample to each vc along with their similarity score.
    The input type is a list of shape [batch_size, [x, y, p_h, p_w, similarity_score]] where (x, y) is
    the x and y indices of the image where the patch starts, p_h and p_w is the x2 and y2 indices where the image patch
    ends, and the similarity score is the score of how similar that patch to its correlated vc.
    """
    def loss(y_true, y_pred):

        def condition(start_, end_, g_loss, m_loss):
            return tf.less(start_, end_)

        def body(start_, end_, g_loss, m_loss):
            # Gathering Cost
            loss_ = cluster_separation_cost(y_true[start_, :], vc_distances[start_])
            g_loss = tf.concat([g_loss, loss_[0]], axis=0)

            # Membership Cost
            if next_coordinates is not None and current_coordinates is not None:
                patches, next_patches, target_class_vc_activation = generate_patches(y_true[start_, :],
                                                                                     inputs_images[start_, :, :, :],
                                                                                     current_coordinates[start_, :],
                                                                                     next_coordinates[start_, :],
                                                                                     gap,
                                                                                     n_gap,
                                                                                     vcl_activations[start_, :],
                                                                                     p_h,
                                                                                     p_w)
                loss_membership = membership_cost_fun(patches, next_patches, vc_per_class, target_class_vc_activation)
                m_loss = tf.concat([m_loss, loss_membership], axis=0)
            else:
                m_loss = tf.concat([m_loss, tf.constant([0.])], axis=0)

            return [tf.add(start_, 1), end_, g_loss, m_loss]

        _, _, gather_, membership_ = tf.while_loop(condition,
                                                   body,
                                                   loop_vars=[tf.constant(0), batch_size, tf.constant([0.]), tf.constant([0.])],
                                                   shape_invariants=[tf.constant(0).get_shape(), batch_size.get_shape(), tf.TensorShape([None]), tf.TensorShape([None])])

        gather_ = gather_[1:]
        membership_ = membership_[1:]
        gathering_cost = tf.reduce_mean(gather_)
        membership_cost = tf.reduce_mean(membership_)
        # Diversity Cost
        divers_cost = - diversity_cost(vc_weights, num_classes)
        crossentropy = K.mean(K.categorical_crossentropy(y_true, y_pred))
        final_loss = 1e-4 + crossentropy + (0.8 * gathering_cost) + (0.05 * membership_cost) + (0.2 * divers_cost)
        return final_loss

    return loss
