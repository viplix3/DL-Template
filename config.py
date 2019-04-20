num_parallel_calls = 10
input_shape = 608 # input shape of the model
# for data augmentation
jitter = 0.3 
hue = 0.1
sat = 1.5
cont = 0.8
bri = 0.1

# decay parameters
use_warm_up = False # for warm up training so that the model training goes smoothly without gradient explosion
burn_in_epochs = 1 # number of iterations to be used for burn-in
norm_decay = 0.9 # batch_norm momentum
weight_decay = 0.9 # weight decay for l2-regularization of convolution kernels
norm_epsilon = 1e-3 # for avoiding division by zero error during batch normalization
num_classes = 1 # number of classes in the dataset
init_learning_rate = 1e-4 # initial learning rate for carrying out the burn in process
learning_rate = 1e-4 # learning rate for training the model
learning_rate_lower_bound = 1e-7 # lower bound for learning rate
momentum = 0.9 # momentum for the optimizer
train_batch_size = 16 # batch size to be used during training
subdivisions = 1 # for splitting the training data into this many minibtches for processing
val_batch_size = 16 # batch size to be used during validation
train_num = 4500 # number of images to be used for training
val_num = 510 # number of images to be used for validation
Epoch = 300 # number of epochs for training
warm_up_lr_scheduler = 'polynomial' # learning rate scheduler to be used during burn-in (linear, exponential, polynomial)
lr_scheduler = "cosine" # learning rate scheduler to be used after burn-in (linear, exponential, polynomial)
gpu_num = "0" # gpu bus id to be used for all the processes
logs_dir = './logs/' # path for saving the training/validation logs
data_dir = './dataset/' # base path in which the dataset has been kept
model_dir = './converted/' # path for saving the model
classes_path = './model_data/pd_classes.txt' # path for the text file containing the classes of the dataset
train_annotations_file = './train.txt' # path for the text file containing the training image annotations
val_annotations_file = './val.txt' # path for text file containing the validation image annotations
output_dir = './tfrecords/' # path for saving the tfrecords
model_export_path = './protobuf_model/' # path for saving the protobuf model for production purposes