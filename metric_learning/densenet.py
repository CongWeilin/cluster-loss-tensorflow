import os
import time
import shutil
import sys
from datetime import timedelta
from data_providers.utils import get_data_provider_by_name
import numpy as np
import tensorflow as tf
import metric_loss_ops as metric_loss

class DenseNet:
    def __init__(self, data_provider, densenet_params):

        self.data_provider = data_provider
        self.data_shape = data_provider.data_shape
        self.n_classes = data_provider.n_classes
        
        self.depth = densenet_params['depth']
        self.growth_rate = densenet_params['growth_rate']
        self.first_output_features = self.growth_rate * 2
        self.total_blocks = densenet_params['total_blocks']
        self.layers_per_block = (self.depth - (self.total_blocks + 1)) // self.total_blocks
        self.bc_mode = densenet_params['bc_mode']
        self.reduction = densenet_params['reduction']
        self.keep_prob = densenet_params['keep_prob']
        self.weight_decay = densenet_params['weight_decay']
        self.nesterov_momentum = densenet_params['nesterov_momentum']
        self.model_type = densenet_params['model_type']
        self.dataset_name = densenet_params['dataset']
        self.should_save_model = densenet_params['should_save_model']
        self.save_path = densenet_params['save_path']
        self.embedding_dim = densenet_params['embedding_dim']
        self.display_iter = densenet_params['display_iter']
        self.logs_path = densenet_params['logs_path']
        self.margin_multiplier = densenet_params['margin_multiplier']
        self.batches_step = 0
        
        if not self.bc_mode:
            print("Build %s model with %d blocks, "
                  "%d composite layers each." % (
                      self.model_type, self.total_blocks, self.layers_per_block))
        if self.bc_mode:
            self.layers_per_block = self.layers_per_block // 2
            print("Build %s model with %d blocks, "
                  "%d bottleneck layers and %d composite layers each." % (
                      self.model_type, self.total_blocks, self.layers_per_block,
                      self.layers_per_block))
        print("Reduction at transition layers: %.1f" % self.reduction)

        self._define_inputs()
        self._build_graph()
        self._initialize_session()
        self._count_trainable_params()
        

    def _initialize_session(self):
        """Initialize session, variables, saver"""
        config = tf.ConfigProto()
        # restrict model GPU memory utilization to min required
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        tf_ver = int(tf.__version__.split('.')[1])
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter(self.logs_path)

    def _count_trainable_params(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parametes = 1
            for dim in shape:
                variable_parametes *= dim.value
            total_parameters += variable_parametes
        print("Total training params: %.1fM" % (total_parameters / 1e6))


    def save_model(self, global_step=None):
        self.saver.save(self.sess, self.save_path, global_step=global_step)

    def log_loss(self, loss, epoch, prefix, should_print=True):
        if should_print:
            print("%s, epoch %d, mean cross_entropy: %f" % (prefix, epoch, loss))
        summary = tf.Summary(value=[
            tf.Summary.Value(
                tag='loss_%s' % prefix, simple_value=float(loss)),
        ])
        self.summary_writer.add_summary(summary, epoch)

    def _define_inputs(self):
        shape = [None]
        shape.extend(self.data_shape)
        self.images = tf.placeholder(
            tf.float32,
            shape=shape,
            name='input_images')
        self.labels = tf.placeholder(
            tf.float32,
            shape=[None,],
            name='labels')
        self.learning_rate = tf.placeholder(
            tf.float32,
            shape=[],
            name='learning_rate')
        self.is_training = tf.placeholder(tf.bool, shape=[])

    def composite_function(self, _input, out_features, kernel_size=3):
        with tf.variable_scope("composite_function"):
            # BN
            output = self.batch_norm(_input)
            # ReLU
            output = tf.nn.relu(output)
            # convolution
            output = self.conv2d(
                output, out_features=out_features, kernel_size=kernel_size)
            # dropout(in case of training and in case it is no 1.0)
            output = self.dropout(output)
        return output

    def bottleneck(self, _input, out_features):
        with tf.variable_scope("bottleneck"):
            output = self.batch_norm(_input)
            output = tf.nn.relu(output)
            inter_features = out_features * 4
            output = self.conv2d(
                output, out_features=inter_features, kernel_size=1,
                padding='VALID')
            output = self.dropout(output)
        return output

    def add_internal_layer(self, _input, growth_rate):
        # call composite function with 3x3 kernel
        if not self.bc_mode:
            comp_out = self.composite_function(
                _input, out_features=growth_rate, kernel_size=3)
        elif self.bc_mode:
            bottleneck_out = self.bottleneck(_input, out_features=growth_rate)
            comp_out = self.composite_function(
                bottleneck_out, out_features=growth_rate, kernel_size=3)
        # concatenate _input with out from composite function
        output = tf.concat(axis=3, values=(_input, comp_out))
        return output

    def add_block(self, _input, growth_rate, layers_per_block):
        output = _input
        for layer in range(layers_per_block):
            with tf.variable_scope("layer_%d" % layer):
                output = self.add_internal_layer(output, growth_rate)
        return output

    def transition_layer(self, _input):
        # call composite function with 1x1 kernel
        out_features = int(int(_input.get_shape()[-1]) * self.reduction)
        output = self.composite_function(
            _input, out_features=out_features, kernel_size=1)
        # run average pooling
        output = self.avg_pool(output, k=2)
        return output

    def final_output(self, _input):
        # BN
        output = self.batch_norm(_input)
        # ReLU
        output = tf.nn.relu(output)
        # average pooling
        last_pool_kernel = int(output.get_shape()[-2])
        output = self.avg_pool(output, k=last_pool_kernel)
        # FC
        features_total = int(output.get_shape()[-1])
        output = tf.reshape(output, [-1, features_total])
        W = self.weight_variable_xavier([features_total, self.embedding_dim], name='W')
        bias = self.bias_variable([self.embedding_dim])
        output = tf.matmul(output, W) + bias
        output = tf.nn.l2_normalize(output,1)
        return output

    def conv2d(self, _input, out_features, kernel_size,
               strides=[1, 1, 1, 1], padding='SAME'):
        in_features = int(_input.get_shape()[-1])
        kernel = self.weight_variable_msra(
            [kernel_size, kernel_size, in_features, out_features],
            name='kernel')
        output = tf.nn.conv2d(_input, kernel, strides, padding)
        return output

    def avg_pool(self, _input, k):
        ksize = [1, k, k, 1]
        strides = [1, k, k, 1]
        padding = 'VALID'
        output = tf.nn.avg_pool(_input, ksize, strides, padding)
        return output

    def batch_norm(self, _input):
        output = tf.contrib.layers.batch_norm(
            _input, scale=True, is_training=self.is_training,
            updates_collections=None)
        return output

    def dropout(self, _input):
        if self.keep_prob < 1:
            output = tf.cond(
                self.is_training,
                lambda: tf.nn.dropout(_input, self.keep_prob),
                lambda: _input
            )
        else:
            output = _input
        return output

    def weight_variable_msra(self, shape, name):
        return tf.get_variable(
            name=name,
            shape=shape,
            initializer=tf.contrib.layers.variance_scaling_initializer())

    def weight_variable_xavier(self, shape, name):
        return tf.get_variable(
            name,
            shape=shape,
            initializer=tf.contrib.layers.xavier_initializer())

    def bias_variable(self, shape, name='bias'):
        initial = tf.constant(0.0, shape=shape)
        return tf.get_variable(name, initializer=initial)

    def _build_graph(self):
        growth_rate = self.growth_rate
        layers_per_block = self.layers_per_block
        # first - initial 3 x 3 conv to first_output_features
        with tf.variable_scope("Initial_convolution"):
            output = self.conv2d(
                self.images,
                out_features=self.first_output_features,
                kernel_size=3)

        # add N required blocks
        for block in range(self.total_blocks):
            with tf.variable_scope("Block_%d" % block):
                output = self.add_block(output, growth_rate, layers_per_block)
            # last block exist without transition layer
            if block != self.total_blocks - 1:
                with tf.variable_scope("Transition_after_block_%d" % block):
                    output = self.transition_layer(output)

        with tf.variable_scope("Transition_to_classes"):
            embeddings = self.final_output(output)
        
        # Losses
        loss = metric_loss.cluster_loss(self.labels,embeddings,self.margin_multiplier,print_losses=True)
        self.loss = loss
        l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])

        # optimizer and train step
        optimizer = tf.train.MomentumOptimizer(self.learning_rate, self.nesterov_momentum, use_nesterov=True)
        self.train_step = optimizer.minimize(loss + l2_loss * self.weight_decay)

    def train_all_epochs(self, train_params):
        n_epochs = train_params['n_epochs']
        learning_rate = train_params['initial_learning_rate']
        batch_size = train_params['batch_size']
        reduce_lr_epoch_1 = train_params['reduce_lr_epoch_1']
        reduce_lr_epoch_2 = train_params['reduce_lr_epoch_2']
        total_start_time = time.time()
        for epoch in range(1, n_epochs + 1):
            print ("Train epoch: %d " % epoch)
            start_time = time.time()
            if epoch == reduce_lr_epoch_1 or epoch == reduce_lr_epoch_2:
                learning_rate = learning_rate / 10
                self.margin_multiplier = 0.94*self.margin_multiplier
                print("Decrease learning rate, new lr = %f" % learning_rate)


            loss = self.train_one_epoch(self.data_provider.train, batch_size, learning_rate)

            self.log_loss(loss, epoch, prefix='train')

            if train_params.get('validation_set', False):
                loss = self.test(self.data_provider.validation, batch_size)
                self.log_loss(loss, epoch, prefix='valid')

            time_per_epoch = time.time() - start_time
            seconds_left = int((n_epochs - epoch) * time_per_epoch)
            print("Time per epoch: %s, Est. complete in: %s" % (
                str(timedelta(seconds=time_per_epoch)),
                str(timedelta(seconds=seconds_left))))

            if self.should_save_model:
                self.save_model()

        total_training_time = time.time() - total_start_time
        print("\nTotal training time: %s" % str(timedelta(
            seconds=total_training_time)))

    def train_one_epoch(self, data, batch_size, learning_rate):
        num_examples = data.num_examples
        total_loss = []
        for i in range(num_examples // batch_size):
            batch = data.next_batch(batch_size)
            images, labels = batch
            feed_dict = {
                self.images: images,
                self.labels: labels,
                self.learning_rate: learning_rate,
                self.is_training: True,
            }
            fetches = [self.train_step, self.loss]
            result = self.sess.run(fetches, feed_dict=feed_dict)
            _, loss = result
            total_loss.append(loss)
            self.batches_step += 1
            self.log_loss(loss, self.batches_step, prefix='per_batch',should_print=False)
        mean_loss = np.mean(total_loss)
        return mean_loss

    def test(self, data, batch_size):
        num_examples = data.num_examples
        total_loss = []
        for i in range(num_examples // batch_size):
            batch = data.next_batch(batch_size)
            feed_dict = {
                self.images: batch[0],
                self.labels: batch[1],
                self.is_training: False,
            }
            fetches = [self.loss]
            loss = self.sess.run(fetches, feed_dict=feed_dict)
            total_loss.append(loss)
        mean_loss = np.mean(total_loss)
        return mean_loss

