# Cluster-loss-Tensorflow
This repository is an implementation of Deep Metric Learning via Facility Location on tensorflow. We build this on Cifar100 and Densenet-40. This paper is available [here](https://arxiv.org/pdf/1612.01213.pdf). For the loss layer implementation, look at [here](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/losses/python/metric_learning). For the Densenet implementation, look at [here](https://github.com/ikhlestov/vision_networks).
 
	@inproceedings{songCVPR17,
	  Author    = {Hyun Oh Song and Stefanie Jegelka and Vivek Rathod and Kevin Murphy},
	  Title     = {Deep Metric Learning via Facility Location},
	  Booktitle = {Computer Vision and Pattern Recognition (CVPR)},
	  Year      = {2017}
	}
    
## Installation
1. Install prerequsites for `tensorflow` (see: [tensorflow-gpu installation instructions](https://www.tensorflow.org/install/install_linux)).
2. Run `pip install -r requirements.txt` get required support.

## Training Procedure
1. Modify `metric_learning_densenet.py` for training-params and densenet-params. We pick Cifar100 as our training data, because it's tiny, save GPU-memory (when batch size 64, it cost about 4.6G GPU-Memory) and good for doing research.
2. Run `python metric_learning_densenet.py`, the `data_provider` with automaticlly handle data download and process. After that, start Densenet-Cluster-loss training.
3. Download Downsampled Imagenet with size 32x32 from [here](https://patrykchrabaszcz.github.io/Imagenet32/). Modify `metric_learning_densenet.py` train on Imagenet.

## Feature Extraction after Training
1. Modify `metric_learning_densenet.py` extract feature embeddings on cifar test set, the embeddings is saved with `.npy` format used for evaluation process.

## Clustering and Retrieval Evaluation
1. Run `python visualization/tsne.py` can plot and save the cluster result on Cifar database.
![tSNE](evaluation/tsne.jpg)


## Repository Information
- [x] Densenet tensorflow training code
- [x] Deep metric learning cluster loss code
- [ ] Evaludation
    - [ ] NMI, Recall@K code
    - [x] feature extraction code
    - [x] feature visulization code (tSNE)
- [x] Dataset support
    - [x] cifar-10
    - [x] cifar-100
    - [x] imagenet-32x32
