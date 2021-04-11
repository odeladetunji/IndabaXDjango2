""" Configuration file for an experiment
"""

random_seed = 24
train_dir = 'dataset/train/'
test_dir = "dataset/test/"

model = "xception_alone"

classnames = ['Benign', 'InSitu', 'Invasive', 'Normal']

features_dir = "./extracted_features/"
train_features = "./extracted_features/features_train.npy"
valid_features = "./extracted_features/features_validate.npy"
test_features = "./extracted_features/features_test.npy"

checkpoint_name = "./checkpoints/checkpoint.h5"
model_path = checkpoint_name +'.h5'
result_file = "history.txt"
train_batch_size = 8
val_batch_size = 8
test_batch_size = 4
img_height, img_width = 512, 512
input_shape = (img_height, img_width, 3)
epochs = 30

num_classes = 4
lr = 0.001
beta_1 = 0.6
beta_2 = 0.8

save_feature = True
dropout_rate = 0.5
