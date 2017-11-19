import os
import time
import shutil
import sys
from datetime import timedelta
from data_providers.utils import get_data_provider_by_name
import numpy as np
import tensorflow as tf
from metric_learning.densenet import DenseNet

train_params_cifar = {
    'batch_size': 64,
    'n_epochs': 60,
    'initial_learning_rate': 0.1,
    'reduce_lr_epoch_1': 30,  # epochs * 0.5
    'reduce_lr_epoch_2': 45,  # epochs * 0.75
    'validation_set': True, 
    'validation_split': None,  # None or float
    'shuffle': 'every_epoch',  # None, once_prior_train, every_epoch
    'normalization': 'by_chanels',  # None, divide_256, divide_255, by_chanels
}

densenet_params={
    'growth_rate':12,
    'depth':40,
    'total_blocks':3,
    'keep_prob':0.5, #keep probability for dropout. If keep_prob = 1, dropout will be disables
    'weight_decay':1e-4,
    'nesterov_momentum':0.9, #momentum for Nesterov optimizer
    'model_type':'DenseNet',
    'bc_mode': False,
    'dataset' : 'C100+',
    'should_save_model':True,
    'reduction':0.5,
    'renew_logs': False,
    'embedding_dim':64,
    'display_iter':100,
    'save_path':'/home/weilin/Downloads/densenet/saves/dense_12_40.chkpt',
    'logs_path':'/home/weilin/Downloads/densenet/saves/dense_12_40',
    'margin_multiplier':1.0,
}
data_provider = get_data_provider_by_name(densenet_params['dataset'], train_params_cifar)
print("Data provider train images: ", data_provider.train.num_examples)
model = DenseNet(data_provider, densenet_params)
model.train_all_epochs(train_params_cifar)

