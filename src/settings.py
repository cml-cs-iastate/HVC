import os
import tensorflow as tf

from src.NeuralNetworks.VGG16 import VGG_16
from src.NeuralNetworks.ResNet50 import ResNet_50


print(f'TensorFlow Version: {tf.__version__}')
print(f'List of GPU devices being utilized: {tf.config.list_physical_devices("GPU")}')

# ----------------------------------------------------------------------------------------------------------------------
# Data

dataset = 'kvasir'

data_path = '../../../Data/'

training_path = os.path.join(data_path, dataset, 'train')
val_path = os.path.join(data_path, dataset, 'validate')
testing_path = os.path.join(data_path, dataset, 'test')

if dataset == 'kvasir':
    num_classes = 8
else:
    num_classes = 1000

# ----------------------------------------------------------------------------------------------------------------------
# Model

model_name = 'VGG_16'
model_dict = {'VGG_16': VGG_16, 'ResNet_50': ResNet_50}

# hyper-parameters
neural_network = model_dict[model_name]
batch_size = 32
epochs = 100
image_width = 224
image_height = 224
drop_off = 0.5

plot_model = True

# ----------------------------------------------------------------------------------------------------------------------
# HVC Model hyper-parameters
total_iteration = epochs
training_epochs = 5
vc_per_class = 10
num_vc = num_classes * vc_per_class
total_sub_models_num = 5
p_h = 1
p_w = 1
p_d = 128

# ----------------------------------------------------------------------------------------------------------------------
# Model path
output_directory = os.path.join('../Output', dataset, model_name)
sub_model_directory = os.path.join(output_directory, 'SubModels')
base_model_name = os.path.join(output_directory, 'base_model.h5')

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

if not os.path.exists(sub_model_directory):
    os.makedirs(sub_model_directory)

membership_dict_name = os.path.join(sub_model_directory, 'membership_dict.pkl')
