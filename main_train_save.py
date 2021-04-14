"""
This script will train and save a model.

"""


import torch

from utils.utils import UCRDataset, UCRTorchTrainer
from utils.scheduled_optimizer import ScheduledOptim

from nets.resnet import RESNET
from nets.fcn import Fcn
from nets.transformer_model import UnivarTransformer

######## specify parameter #################################################

# chose model: resnet, fcn, transformer-postLN, transformer-preLN
MODEL = 'resnet'

# the model is saved under this name
NAME = 'models/gunpoint.pth'

# the log is saved under this name
LOGNAME = 'logs/test.csv'

# chose data set to use
DATA_SET = 'Gun_Point'

# relative path to dataset
DATA_PATH = './UCR_archive/UCR_TS_Archive_2015'


# define some training related hyper-parameters
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 0.001

# large value will turn off early stopping
EARLY_STOPPING = 1000000

# warm up time for transformer-postLN
WARMUP = 50

#############################################################################


# print parameters
print(f'MODEL: {MODEL}; NAME: {NAME}; DATA_SET: {DATA_SET}; ' +
      f'BATCH_SIZE: {BATCH_SIZE}; LEARNING_RATE: {LEARNING_RATE}')

# create train and test set
train_set = UCRDataset(f'{DATA_PATH}/{DATA_SET}/{DATA_SET}_TRAIN')
test_set = UCRDataset(f'{DATA_PATH}/{DATA_SET}/{DATA_SET}_TEST')

# define model to use

if MODEL == 'resnet':
    net = RESNET(train_set.siglen, train_set.n_target_classes)
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

elif MODEL == 'fcn':
    net = Fcn(train_set.siglen, train_set.n_target_classes)
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

elif MODEL == 'transformer-postLN':
    net = UnivarTransformer(train_set.siglen, train_set.n_target_classes, heads=1, depth=1, emb_dim=50, layer_order='postLN')
    optimizer = ScheduledOptim(torch.optim.Adam(net.parameters()), LEARNING_RATE, WARMUP)

elif MODEL == 'transformer-preLN':
    net = UnivarTransformer(train_set.siglen, train_set.n_target_classes, heads=1, depth=1, emb_dim=50, layer_order='preLN')
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

# create data loaders
traindata_loader = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=BATCH_SIZE,
    shuffle=True
    )

testdata_loader = torch.utils.data.DataLoader(
    dataset=test_set,
    batch_size=BATCH_SIZE,
    shuffle=False
    )

# train model with model trainer
model_trainer = UCRTorchTrainer(
    model=net,
    criterion=torch.nn.CrossEntropyLoss(),
    optimizer=optimizer
    )

model_trainer.train(
    traindata_loader,
    EPOCHS,
    testdata_loader=testdata_loader,
    early_stopping=EARLY_STOPPING,
    save_as=NAME,
    save_log_as=LOGNAME
    )
