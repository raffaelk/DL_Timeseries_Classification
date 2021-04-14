"""
This script will create visualizations for a pre trained model.

"""


import torch

from torch.utils.data import DataLoader

# my own functions
from utils.utils import UCRDataset, UCRTorchTrainer, importance_per_class


######## specify parameter #################################################

# index to visualize
IDX = 13

# name of model and dataset
NAME = 'models/gunpoint.pth'
DATA_SET = 'Gun_Point'


# path to datasets
DATA_PATH = './UCR_archive/UCR_TS_Archive_2015'


############################################################################

# create train and test set
train_set = UCRDataset(f'{DATA_PATH}/{DATA_SET}/{DATA_SET}_TRAIN')
test_set = UCRDataset(f'{DATA_PATH}/{DATA_SET}/{DATA_SET}_TEST')

# load model to use
net = torch.load(NAME, map_location=torch.device('cpu'))

# create data loaders for the test data only
testdata_loader = DataLoader(
    dataset=test_set,
    batch_size=1,
    shuffle=False
    )

# model with model trainer
model_trainer = UCRTorchTrainer(
    model=net,
    criterion=torch.nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam(net.parameters())
    )

# show plots for the test data
if 'Fcn' in str(net.type):
    model_trainer.show_cam(testdata_loader, IDX)
    model_trainer.show_gradients(testdata_loader, IDX)

elif 'RESNET' in str(net.type):
    model_trainer.show_cam(testdata_loader, IDX)
    model_trainer.show_gradients(testdata_loader, IDX)
    importance_per_class(model_trainer, testdata_loader, mode='CAM')
    
elif 'UnivarTransformer' in str(net.type):
    model_trainer.show_attention(testdata_loader, IDX, block=0, show=True)
    model_trainer.show_gradients(testdata_loader, IDX)
    importance_per_class(model_trainer, testdata_loader, mode='attention')
