from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications.vgg16 import VGG16

from src.Utils.utils import custom_weights_init
from src.VisualConcept.vcl import VCL


def VGG_16(num_classes, num_vc=0, dropout_rate=0, layer_to_vcl=5, p_h=1, p_w=1, total_sub_models_num=5):
    m = VGG16(classes=num_classes, input_shape=(224, 224, 3), weights='imagenet', include_top=False)
    layer_to_vcl_dict = {1: 'block1_pool', 2: 'block2_pool', 3: 'block3_pool', 4: 'block4_pool', 5: 'block5_pool'}
    layer_ = m.get_layer(layer_to_vcl_dict[layer_to_vcl]).output

    if num_vc > 0:
        layer_ = layers.Dropout(rate=dropout_rate, name='dropout')(layer_)
        additional_layer_1 = layers.Conv2D(128, (1, 1), activation='relu', padding='same', name='additional_layer_1')(layer_)
        additional_layer_2 = layers.Conv2D(128, (1, 1), activation='sigmoid', padding='same', name='additional_layer_2')(additional_layer_1)
        vcl = VCL(num_vcl=num_vc, p_h=p_h, p_w=p_w)(additional_layer_2)
        output = layers.Dense(num_classes,
                              activation='softmax',
                              name='sub_model_predictions',
                              kernel_initializer=custom_weights_init(weights_shape=[vcl[1].shape[-1], num_classes]),
                              trainable=False,
                              use_bias=False)(vcl[0])
        vcl_next = None
        if layer_to_vcl == total_sub_models_num:
            model = models.Model(inputs=m.input, outputs=output, name='vgg16')
        else:
            layer_to_next_vcl = m.get_layer(layer_to_vcl_dict[layer_to_vcl + 4]).output  # + 1
            additional_layer_1_next = layers.Conv2D(128, (1, 1), activation='relu', padding='same', name='additional_layer_1_next')(layer_to_next_vcl)
            additional_layer_2_next = layers.Conv2D(128, (1, 1), activation='sigmoid', padding='same', name='additional_layer_2_next')(additional_layer_1_next)
            vcl_next = VCL(num_vcl=num_vc, p_h=p_h, p_w=p_w)(additional_layer_2_next)
            output2 = layers.Dense(num_classes,
                                   activation='softmax',
                                   name='sub_model_predictions2',
                                   kernel_initializer=custom_weights_init(weights_shape=[vcl[1].shape[-1], num_classes]),
                                   trainable=False,
                                   use_bias=False)(vcl_next[0])
            model = models.Model(inputs=m.input, outputs=[output, output2], name='vgg16')
        return model, vcl, m.input, vcl_next

    else:
        x = layers.Flatten(name='flatten')(m.layers[-1].output)
        x = layers.Dense(4096, activation='relu', name='fc1')(x)
        x = layers.Dropout(rate=dropout_rate)(x)
        x = layers.Dense(4096, activation='relu', name='fc2')(x)
        x = layers.Dropout(rate=dropout_rate)(x)
        output = layers.Dense(num_classes, activation='softmax', name='predictions', kernel_regularizer=l2(0.01))(x)
        model = models.Model(inputs=m.input, outputs=output, name='vgg16')
        return model
