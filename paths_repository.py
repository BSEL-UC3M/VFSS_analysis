import os
from os.path import join

# Data file paths. CHANGE IF NEEDED

current_directory = os.path.dirname(os.path.abspath(__file__))
data_dir = join(current_directory, 'data')
raw_VFSS_dir = join(current_directory, 'data', 'raw_VFSS')
output_dir = join(current_directory, 'data', 'output_data')
train_data_dir = join(current_directory, 'data', 'train_data')
models_dir = join(current_directory, 'models')