# cluster-loss-tensorflow
This repository is an implementation of Deep Metric Learning via Facility Location on tensorflow. We build this on Cifar100 and Densenet-40. This paper is available [here](https://arxiv.org/pdf/1612.01213.pdf). For the loss layer implementation, look at [here](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/losses/python/metric_learning). For the Densenet implementation, look at [here](https://github.com/ikhlestov/vision_networks).

## installation
1. Install prerequsites for `tensorflow` (see: [tensorflow-gpu installation instructions](https://www.tensorflow.org/install/install_linux)).
2. Run `pip install -r requirements.txt` get required support.
3. Run `python metric_learning_densenet_cifar100.py` start Densenet-Cluster-loss training.