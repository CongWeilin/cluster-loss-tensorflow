import numpy as np
from data_providers.utils import get_data_provider_by_name
from metric_learning.densenet import DenseNet

train_params_cifar = {
    'batch_size': 128,
    'n_epochs': 100,
    'initial_learning_rate': 0.1,
    'reduce_lr_epoch_1': 50,  # epochs * 0.5
    'reduce_lr_epoch_2': 75,  # epochs * 0.75
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
    'model_type':'DenseNet-BC',
    'bc_mode': True,
    'dataset' : 'C100+',
    'should_save_model':True,
    'reduction':0.5,
    'renew_logs': False,
    'embedding_dim':64,
    'display_iter':100,
    'save_path':'/home/weilin/Downloads/densenet/saves/DenseNet-BC_growth_rate=12_depth=40_dataset_C100+/metric_learning.chkpt',
    'logs_path':'/home/weilin/Downloads/densenet/saves/DenseNet-BC_growth_rate=12_depth=40_dataset_C100+/metric_learning',
    'pretrained_model':'/home/weilin/Downloads/densenet/pretrained_model/DenseNet-BC_growth_rate=12_depth=40_dataset_C100+/model.chkpt',
    'margin_multiplier':1.0,
}

data_provider = get_data_provider_by_name(densenet_params['dataset'], train_params_cifar)
model = DenseNet(data_provider, densenet_params)

training = True
feature_extract = False

if training:
    print("Data provider train images: ", data_provider.train.num_examples)
    model.load_pretrained_model()
    model.train_all_epochs(train_params_cifar)
    
if feature_extract:
    if not training:
        model.load_model()
    print("Data provider test images: ", data_provider.test.num_examples)
    feature_embeddings = model.feature_extracting(data_provider.test, batch_size=100)
    print(feature_embeddings.shape)
    np.save('feature_embedding.npy',feature_embeddings)


