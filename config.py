# Data locations
model_dir = '/Users/koen/Workspace/Autonomous-Ai-drone-scripts/data/models/trained_best_model_full_set_inception_tranfer_SIGMOID.h5'
eval_dir = '/Users/koen/Workspace/Autonomous-Ai-drone-scripts/data/eval_data.npy'
data_dir = "/Users/koen/Workspace/Autonomous-Ai-drone-scripts/dataset_POC/Training"

# Training variables
training_size = 17571
validation_size = 7529
batch_size = 32

# Transfer learning
transfer_learning = False

train_dir = '/Users/koen/Workspace/Autonomous-Ai-drone-scripts/dataset_POC/Training'
val_dir = '/Users/koen/Workspace/Autonomous-Ai-drone-scripts/dataset_POC/Testing'
test_dir = '/Users/koen/Workspace/Autonomous-Ai-drone-scripts/data/eval_data.npy'

# Visualizer
playback_speed = 0.75
visualizer_folder = "/home/drone/Desktop/dataset_POC/Testing/Run1"


# Saving trained model
model_name = 'inceptionv3_new_preprocessor__sigmoid_lr_0.001.h5'