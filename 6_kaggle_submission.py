import gc
from sklearn.utils import shuffle
from PIL import Image
import time
import re
import random
from glob import glob
from matplotlib import pyplot as plt
import os
import sys
import numpy as np
import tensorflow as tf
import sklearn
import pandas as pd
from PIL import Image
from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope
from keras.losses import binary_crossentropy, categorical_crossentropy
import keras.backend as K
import cv2
import tensorflow as tf
from tensorflow.python.ops import array_ops
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'




class train_datatrain_da():
    def __init__(self, filepath):
        self.filepath = filepath
        assert '.npy' in filepath
        if not os.path.exists(filepath):
            print("[!] Data file not exists")
            sys.exit(1)

    def __enter__(self):
        print("[*] Loading data...")
        self.data = np.load(self.filepath)

        # np.random.shuffle(self.data)
        print("[*] Load successfully...")
        return self.data

    def __exit__(self, type, value, trace):
        del self.data
        gc.collect()
        print("In __exit__()")


# data augmentation options# data a
def data_augmentation(image, mode):
    if mode == 0:
        # original
        return image
    elif mode == 1:
        # flip up and down
        return np.flipud(image)
    elif mode == 2:
        # rotate 180 degree
        return np.rot90(image, k=2)
    elif mode == 3:
        # flip left and right
        return np.flipud(image)


def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)


def load_images(filelist):
    # pixel value range 0-255
    if not isinstance(filelist, list):
        im = cv2.imread(filelist, 0)

        #im = Image.open(filelist)#.convert('L')
        return(im) #np.array(im).reshape(1, im.size[1], im.size[0], 1)
    data = []
    for file in filelist:

        im = cv2.imread(file, 0)

        #im = Image.open(file)#.convert('L')
        data.append(im) #(np.array(im).reshape(1, im.size[1], im.size[0], 1))
    return data


def multi_label_hot(prediction, threshold=0.5):
    prediction = tf.cast(prediction, tf.float32)
    threshold = float(threshold)
    return tf.cast(tf.greater(prediction, threshold), tf.float32)


def focal_loss(prediction_tensor, target_tensor, weights=None, alpha=0.25, gamma=2):
    r"""Compute focal loss for predictions.
        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    Args:
     prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
     target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
     weights: A float tensor of shape [batch_size, num_anchors]
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """
    sigmoid_p = tf.nn.sigmoid(prediction_tensor)
    zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)

    # For poitive prediction, only need consider front part loss, back part is 0;
    # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
    pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)

    # For negative prediction, only need consider back part loss, front part is 0;
    # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
    return per_entry_cross_ent


def f1_loss(y_true, y_pred):
    tp = K.sum(y_true * y_pred, axis=0)
    tn = K.sum((1 - y_true) * (1 - y_pred), axis=0)
    fp = K.sum((1 - y_true) * y_pred, axis=0)
    fn = K.sum(y_true * (1 - y_pred), axis=0)
    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    # 1 - K.mean(f1)
    return 1 - f1


momentum = 0.9
cardinality = 8 # how many split ?
blocks = 3 # 3 res_block ! (split + transition)
depth = 64  # out channel

reduction_ratio = 4
batch_size = 32  # 128

img_channels = 1
class_num = 28
weight_decay = 0.0005


def conv_layer(input, filter, kernel, stride, padding='SAME', layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, use_bias=False, filters=filter, kernel_size=kernel, strides=stride,
                                   padding=padding)
        return network


def Global_Average_Pooling(x):
    return global_avg_pool(x, name='Global_avg_pooling')


def Average_pooling(x, pool_size=[2, 2], stride=2, padding='SAME'):
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


def Batch_Normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True):
        return tf.cond(training,
                       lambda: batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda: batch_norm(inputs=x, is_training=training, reuse=True))


def Relu(x):
    return tf.nn.relu(x)


def Sigmoid(x):
    return tf.nn.sigmoid(x)


def Concatenation(layers):
    return tf.concat(layers, axis=3)


def Fully_connected(x, units=class_num, layer_name='fully_connected'):
    with tf.name_scope(layer_name):
        return tf.layers.dense(inputs=x, use_bias=False, units=units)


def first_layer(x, scope, training):
    with tf.name_scope(scope):
        x = conv_layer(x, filter=64, kernel=[3, 3], stride=1, layer_name=scope + '_conv1')
        x = Batch_Normalization(x, training=training, scope=scope + '_batch1')
        x = Relu(x)

        return x


def transform_layer(x, stride, scope, training=True):
    with tf.name_scope(scope):
        x = conv_layer(x, filter=depth, kernel=[1, 1], stride=1, layer_name=scope + '_conv1')
        x = Batch_Normalization(x, training=training, scope=scope + '_batch1')
        x = Relu(x)

        x = conv_layer(x, filter=depth, kernel=[3, 3], stride=stride, layer_name=scope + '_conv2')
        x = Batch_Normalization(x, training=training, scope=scope + '_batch2')
        x = Relu(x)
        return x


def transition_layer(x, out_dim, scope, training=True):
    with tf.name_scope(scope):
        x = conv_layer(x, filter=out_dim, kernel=[1, 1], stride=1, layer_name=scope + '_conv1')
        x = Batch_Normalization(x, training=training, scope=scope + '_batch1')
        # x = Relu(x)

        return x


def split_layer(input_x, stride, layer_name, training=True):
    with tf.name_scope(layer_name):
        layers_split = list()
        for i in range(cardinality):
            splits = transform_layer(input_x, stride=stride, scope=layer_name + '_splitN_' + str(i), training=training)
            layers_split.append(splits)

        return Concatenation(layers_split)


def squeeze_excitation_layer(input_x, out_dim, ratio, layer_name):
    with tf.name_scope(layer_name):
        squeeze = Global_Average_Pooling(input_x)

        excitation = Fully_connected(squeeze, units=out_dim / ratio, layer_name=layer_name + '_fully_connected1')
        excitation = Relu(excitation)
        excitation = Fully_connected(excitation, units=out_dim, layer_name=layer_name + '_fully_connected2')
        excitation = Sigmoid(excitation)

        excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])
        scale = input_x * excitation

        return scale


def residual_layer(input_x, out_dim, layer_num, res_block=blocks, training=True):
    # split + transform(bottleneck) + transition + merge
    # input_dim = input_x.get_shape().as_list()[-1]

    for i in range(res_block):
        input_dim = int(np.shape(input_x)[-1])

        if input_dim * 2 == out_dim:
            flag = True
            stride = 2
            channel = input_dim // 2
        else:
            flag = False
            stride = 1

        x = split_layer(input_x, stride=stride, layer_name='split_layer_'+layer_num+'_'+str(i), training = training )
        x = transition_layer(x, out_dim=out_dim, scope='trans_layer_' + layer_num + '_' + str(i),
                             training=training)
        x = squeeze_excitation_layer(x, out_dim=out_dim, ratio=reduction_ratio,
                                     layer_name='squeeze_layer_' + layer_num + '_' + str(i))

        if flag is True:
            pad_input_x = Average_pooling(input_x)
            pad_input_x = tf.pad(pad_input_x,
                                 [[0, 0], [0, 0], [0, 0], [channel, channel]])  # [?, height, width, channel]
        else:
            pad_input_x = input_x

        input_x = Relu(x + pad_input_x)

    return input_x


# the model
def SE_ResNeXt(input, is_training=True, output_channels=1):
    training = is_training

    input = first_layer(input, scope='first_layer', training=training)

    x = residual_layer(input, out_dim=64, layer_num='1', training=training)  # out_dim=64
    x = residual_layer(x, out_dim=128, layer_num='2', training = training)
    x = residual_layer(x, out_dim=256, layer_num='3' , training = training)

    x = Global_Average_Pooling(x)
    x = flatten(x)

    x = Fully_connected(x, layer_name='final_fully_connected')
    return x


def roc_auc_score(y_true, y_pred):
    """ ROC AUC Score.
    Approximates the Area Under Curve score, using approximation based on
    the Wilcoxon-Mann-Whitney U statistic.
    Yan, L., Dodier, R., Mozer, M. C., & Wolniewicz, R. (2003).
    Optimizing Classifier Performance via an Approximation to the Wilcoxon-Mann-Whitney Statistic.
    Measures overall performance for a full range of threshold levels.
    Arguments:
        y_pred: `Tensor`. Predicted values.
        y_true: `Tensor` . Targets (labels), a probability distribution.
    """
    with tf.name_scope("RocAucScore"):
        pos = tf.boolean_mask(y_pred, tf.cast(y_true, tf.bool))
        neg = tf.boolean_mask(y_pred, ~tf.cast(y_true, tf.bool))
        pos = tf.expand_dims(pos, 0)
        neg = tf.expand_dims(neg, 1)
        # original paper suggests performance is robust to exact parameter choice
        gamma = 0.2
        p = 3
        difference = tf.zeros_like(pos * neg) + pos - neg - gamma
        masked = tf.boolean_mask(difference, difference < 0.0)
        return tf.reduce_sum(tf.pow(-masked, p))

class denoiser(object):
    def __init__(self, sess, input_c_dim=1, batch_size=32):
        self.sess = sess
        input_c_dim = input_c_dim

        self.is_training = tf.placeholder(tf.bool, name='is_training')

        # input images
        #self.image_size = 512
        self.img_channels = 1
        self.weight_decay = 0.0005

        self.class_num = 28
        self.X = tf.placeholder(tf.float32, shape=[None, None, None , self.img_channels],
                                name='input_image')

        self.p = tf.placeholder(tf.float32, shape=[None, self.class_num], name='actual_classes')
        # actual classes
        self.one_hot_prediction = tf.placeholder(tf.float32, shape=[None, self.class_num], name='one_hot_prediction')

        # predicted classes
        self.logit_q = tf.placeholder(tf.float32, shape=[None, self.class_num], name='input_classes_logit')
        # self.q = tf.placeholder(tf.float32, shape=[None, self.class_num],  name='input_classes')

        # predicted probabilities
        self.logit_q = SE_ResNeXt(self.X, is_training=self.is_training)

        # learning rate
        self.lr = tf.placeholder(tf.float32, name='learning_rate')

        # prediction after softmax
        self.one_hot_prediction = multi_label_hot(self.logit_q)
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.p, logits=self.logit_q))

      

        # f1_score

        # self.eva_f1_score = sklearn.metrics.f1_score(self.p, self.q, labels=None, pos_label=1, average='micro') # or average='weighted' and 'samples'

        self.eva_f1_score = f1(self.p, self.one_hot_prediction)
        # adam optimizer
        # default variables
        # beta one 0.9
        # beta two 0.999
        # Epsilon 10^-8
        # beta1=0.9, beta2=0.999, epsilon=1e-08
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr,
                                           name='AdamOptimizer')

        # returns the list of values in the collection with the given name
        # UPDATE_OPS is a collection of ops (operations performed when the
        # graph runs, like multiplication, ReLU, etc.), not variables.
        # Specifically, this collection maintains a list of ops which
        # need to run after every training step.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # min loss
            self.train_op = optimizer.minimize(self.loss)
        # initialize variables
        init = tf.global_variables_initializer()
        self.sess.run(init)
        print("[*] Initialize model successfully...")

    def evaluate(self, iter_num, test_data, test_datay, sample_dir,
                 summary_merged, summary_writer):
        # assert test_data value range is 0-255
        outstreval = "[*] Evaluating...\n"

        print("[*] Evaluating...")
        f1_score_sum = 0

        for idx in range(len(test_data)):
            input_image = test_data[idx].astype(np.int8) / 255.0

            actual_classes = test_datay.iloc[[idx]].values.reshape((1, 28))



            output_classes, input_image, one_hot_prediction, f1_score_summary = self.sess.run(
                [self.logit_q, self.X, self.one_hot_prediction, summary_merged],

                feed_dict={self.p: actual_classes, self.X: input_image,
                           self.is_training: False})

            summary_writer.add_summary(f1_score_summary, iter_num)
            # feed_dict={self.Y_: actual_classes,
            # self.is_training: False})

            # np.clip
            # Given an interval, values outside the interval are clipped to
            # the interval edges.
            # For example, if an interval of [0, 1] is specified,
            # values smaller than 0 become 0, and values larger than 1 become 1.

            one_hot_prediction = one_hot_prediction.astype('uint8')

            groundtruth_classes = actual_classes.astype('uint8')
            #inputimage = np.clip(input_image, 0, 255).astype('uint8')

            # calculate f1_score
            groundtruth_classes = np.ndarray.transpose(groundtruth_classes)
            one_hot_prediction = np.ndarray.transpose(one_hot_prediction)

            f1_score = sklearn.metrics.f1_score(groundtruth_classes, one_hot_prediction, labels=None, average='macro')

            print('img ', str(idx + 1), ' f1_score:', str(f1_score))

            outstreval = outstreval + ' img ' + str(idx + 1) + ' f1_score: ' + str(
                f1_score) + '\n'

            f1_score_sum += f1_score

        avg_f1_score = f1_score_sum / len(test_data)

        print('--- Test ---- Average f1_score %.2f ---- ' % (avg_f1_score))

        # add average f1_score to tensorboard

        outstreval = outstreval + '--- Test ---- Average f1_score ' + str(
            avg_f1_score) + ' ---\n'

        filename = 'outstreval' + str(iter_num) + '.txt'
        file = open(filename, 'w')
        file.write(outstreval)
        file.close()

    def train(self, data, datay, eval_data, eval_datay, batch_size, ckpt_dir,
              epoch, lr, sample_dir, eval_every_epoch=1):
        # assert data range is between 0 and 1
        numBatch = int((len(data) + 1) / batch_size)
        print('numBatch', (len(data)) / batch_size, len(data), batch_size)

        # if pretrained model exists - load pretrained model
        # else train new model
        # ckpt_dir=checkpoint
        load_model_status, global_step = self.load(ckpt_dir)
        if load_model_status:
            iter_num = global_step
            start_epoch = global_step // numBatch
            start_step = global_step % numBatch
            print("[*] Model restore success!")
        else:
            iter_num = 0
            start_epoch = 0
            start_step = 0
            print("[*] Did not find pretrained model!")
        # make summary
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('lr', self.lr)
        writer = tf.summary.FileWriter('./logs', self.sess.graph)
        merged = tf.summary.merge_all()
        summary_f1_score = tf.summary.scalar('eva_f1_score', self.eva_f1_score)

        outstrtr = "Save output here.\n"

        outstrtr = outstrtr + "[*] Start training, with start epoch " + str(start_epoch) + " start iter " + str(
            iter_num) + "\n"

        print("[*] Start training, with start epoch %d start iter %d : " % (
            start_epoch, iter_num))
        start_time = time.time()

        # evaluate
        # eval_data value range is 0-255
        self.evaluate(iter_num, eval_data, eval_datay, sample_dir=sample_dir,
                      summary_merged=summary_f1_score,
                      summary_writer=writer)

        for epoch in range(start_epoch, epoch):

            for batch_id in range(start_step, numBatch):
                # data=load(img_clean_pats)
                # print(type(data))
                # print(type(datay))
                # <class 'list'>
                # <class 'pandas.core.frame.DataFrame'>
                # print(len(data))
                # print(len(datay))
                batch_images = data[batch_id * batch_size:(
                                                                  batch_id + 1) * batch_size,
                               :, :, :]

                batch_images = batch_images.astype(
                    np.int8) / 255.0  # normalize the data to 0-1

                # batch_images = data[batch_id].astype(np.float32)
                # batch_imagesy = datay[batch_id].reshape((1, 28))
                batch_imagesy = datay[batch_id * batch_size:(
                                                                    batch_id + 1) * batch_size,
                                :]
                batch_imagesy = batch_imagesy.astype(
                    np.int8)


                _, loss, summary = self.sess.run(
                    [self.train_op, self.loss, merged],
                    feed_dict={self.p: batch_imagesy, self.X: batch_images,
                               self.lr: lr[epoch],
                               self.is_training: True})

                # add f1_score as well
                outstrtr = outstrtr + "Epoch: " + str(epoch + 1) + " [" + str(
                    batch_id + 1) + "/" + str(numBatch) + "] " + "time: " + str(
                    time.time() - start_time) + "loss: " + str(loss) + "\n"

                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f"

                      % (epoch + 1, batch_id + 1, numBatch,
                         time.time() - start_time, loss))

                iter_num += 1
                writer.add_summary(summary, iter_num)
            if np.mod(epoch + 1, eval_every_epoch) == 0:
                self.evaluate(iter_num, eval_data, eval_datay,
                              sample_dir=sample_dir,
                              summary_merged=summary_f1_score,
                              summary_writer=writer)  # eval_data value range is 0-255
                self.save(iter_num, ckpt_dir)
        print("[*] Finish training.")
        filename = 'outstrtr' + str(epoch + 1) + '.txt'
        file = open(filename, 'w')
        file.write(outstrtr)
        file.close()

    def save(self, iter_num, ckpt_dir, model_name='CNN-tensorflow'):
        saver = tf.train.Saver()
        checkpoint_dir = ckpt_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        print("[*] Saving model...")
        saver.save(self.sess,
                   os.path.join(checkpoint_dir, model_name),
                   global_step=iter_num)

    def load(self, checkpoint_dir):
        
		print("[*] Reading checkpoint...")
		saver = tf.train.Saver()
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			full_path = tf.train.latest_checkpoint(checkpoint_dir)
			global_step = int(full_path.split('/')[-1].split('-')[-1])
			saver.restore(self.sess, full_path)
			return True, global_step
		else:
			return False, 0
        '''
        print("[*] Reading checkpoint...")
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = './checkpoint/CNN-tensorflow-7500'
            # full_path = tf.train.import_meta_graph('./checkpoint/CNN-tensorflow-121848')
            print(full_path)

            global_step = int(full_path.split('/')[-1].split('-')[-1])
            saver.restore(self.sess, full_path)
            return True, global_step
        else:
            return False, 0
		'''




    def test(self, test_files, ckpt_dir, save_dir):

        """Test CNN"""
        # init variables
        tf.initialize_all_variables().run()
        assert len(test_files) != 0, 'No testing data x!'



        load_model_status, global_step = self.load(ckpt_dir)
        assert load_model_status == True, '[!] Load weights FAILED...'
        print(" [*] Load weights SUCCESS...")
        print("[*] " + " start testing...")


        input_image = np.zeros((1, 512, 512, 1), dtype="uint8")


        #test_data = load_images(test_files[:500])
        outstrtr = 'Id,Predicted'+  "\n"
        for idx in range(len(test_files)):

            in_image = cv2.imread(test_files[idx], 0)
            input_image[0,:,:,0] = in_image



            output_classes, one_hot_prediction = self.sess.run([self.logit_q, self.one_hot_prediction],
                                                               feed_dict={

                                                                   self.X: input_image,
                                                                   self.is_training: False})

            one_hot_prediction = one_hot_prediction.astype('uint8')


            # calculate f1_score
            one_hot_prediction = np.ndarray.transpose(one_hot_prediction)


            if idx % 500 == 0:
                print(idx)

            outstrtr = outstrtr + str(test_files[idx].split("test/",1)[1].split("_",1)[0]) + "," + str(
                re.sub(',', ' ', str([i for i, e in enumerate(one_hot_prediction) if e == 1]))) +  "\n"
            print(str(test_files[idx].split("test/",1)[1].split("_",1)[0]) + "," + str(
                re.sub(',', ' ', str([i for i, e in enumerate(one_hot_prediction) if e == 1]))))
        #imgplot = plt.imshow(input_image[0,:,:,0])
        #plt.show()
        filename = str('kaggle-submission.txt')
        file = open(filename, 'w')
        file.write(outstrtr)
        file.close()




def make_image_row(image):
    image = np.reshape(np.array(image, dtype="uint8"),
                       (image.size[0], image.size[1], 1))

    return image


def make_title(file_id, train_labels):
    label_names = {
        0: "Nucleoplasm",
        1: "Nuclear membrane",
        2: "Nucleoli",
        3: "Nucleoli fibrillar center",
        4: "Nuclear speckles",
        5: "Nuclear bodies",
        6: "Endoplasmic reticulum",
        7: "Golgi apparatus",
        8: "Peroxisomes",
        9: "Endosomes",
        10: "Lysosomes",
        11: "Intermediate filaments",
        12: "Actin filaments",
        13: "Focal adhesion sites",
        14: "Microtubules",
        15: "Microtubule ends",
        16: "Cytokinetic bridge",
        17: "Mitotic spindle",
        18: "Microtubule organizing center",
        19: "Centrosome",
        20: "Lipid droplets",
        21: "Plasma membrane",
        22: "Cell junctions",
        23: "Mitochondria",
        24: "Aggresome",
        25: "Cytosol",
        26: "Cytoplasmic bodies",
        27: "Rods & rings"
    }
    file_targets = train_labels.loc[
        train_labels.Id == file_id, ["Nucleoplasm", "Nuclear membrane", "Nucleoli", "Nucleoli fibrillar center",
                                     "Nuclear speckles", "Nuclear bodies", "Endoplasmic reticulum", "Golgi apparatus",
                                     "Peroxisomes", "Endosomes", "Lysosomes", "Intermediate filaments",
                                     "Actin filaments", "Focal adhesion sites", "Microtubules", "Microtubule ends",
                                     "Cytokinetic bridge", "Mitotic spindle", "Microtubule organizing center",
                                     "Centrosome", "Lipid droplets", "Plasma membrane", "Cell junctions",
                                     "Mitochondria", "Aggresome", "Cytosol", "Cytoplasmic bodies",
                                     "Rods & rings"]].values[0]
    title = " - "
    for n in file_targets:
        title += label_names[n] + " - "
    return title


class train_data():
    def __init__(self, filepath):
        self.filepath = filepath
        assert '.npy' in filepath
        if not os.path.exists(filepath):
            print("[!] Data file not exists")
            sys.exit(1)

    def __enter__(self):
        print("[*] Loading data...")
        self.data = np.load(self.filepath)

        # np.random.shuffle(self.data)
        print("[*] Load successfully...")
        return self.data

    def __exit__(self, type, value, trace):
        del self.data
        gc.collect()
        print("In __exit__()")


def load_data(filepath):
    return train_data(filepath=filepath)


# denoiser_train
def denoiser_train(denoiser, lr):
    with load_data(
            filepath='./dataset/transformed_data/protein_image_patches_pat_size_300_bat_size_32_1001_10000.npy') \
            as data:
        with load_data(
                filepath='./dataset/transformed_data/protein_image_classes_pat_size_300_bat_size_32_1001_10000.npy') as datay:
            # number of images in batch
            batch_size = 32
            # models are saved here
            ckpt_dir = './checkpoint'
            epoch = 100
            # samples are saved here
            sample_dir = './sample'

            lr = 0.1
            lr = lr * np.ones([epoch])

            lr[30:] = lr[0] / 10.0
            lr[60:] = lr[0] / 100.0
            lr[90:] = lr[0] / 1000.0

            train_set = 'train'

            train_labels = pd.read_csv(
                './dataset/transformed_data/one-hot-matrix-classes-data-augmentation-oversampling.csv')
            train_labels.sort_values(['Row'])

            eval_datay = train_labels.loc[:100]

            eval_datay = eval_datay.iloc[:, 3:]

            # print(train_labels.head())

            # get image id
            eval_data_im_id = train_labels.loc[:100, "Id"]

            # get each image channel as a greyscale image (second argument 0 in imread)
            eval_files = []
            for eval_im_id in eval_data_im_id:
                eval_files.append(
                    glob('./dataset/train/{}_green.png'.format(eval_im_id)))
            eval_files.sort()

            eval_files = [''.join(x) for x in eval_files]

            eval_data = load_images(eval_files)

            denoiser.train(data, datay, eval_data, eval_datay,
                           batch_size=batch_size, ckpt_dir=ckpt_dir,
                           epoch=epoch, lr=lr,
                           sample_dir=sample_dir)


# denoiser_test
# dataset for testing

# this is ok
def denoiser_test(denoiser):
    # models are saved here
    ckpt_dir = './checkpoint'
    # test sample are saved here
    test_dir = './test'

    test_set = 'test'

    # get each image channel as a greyscale image (second argument 0 in imread)
    test_files = glob('./dataset/{}/*_green.png'.format(test_set))
    print('testfiles:', len(test_files))
    denoiser.test(test_files,  ckpt_dir=ckpt_dir, save_dir=test_dir)


######## Start training here

# initial learning rate for adam
lr = 0.1

# number of epochs

epoch = 100
# use_gpu=0: use tensorflow cpu
# use_gpu=1: use tensorflow gpu
use_gpu = 0

# train or test
phase = 'test'
ckpt_dir = './checkpoint'
sample_dir = './sample'
test_dir = './test'

if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

lr = lr * np.ones([epoch])

lr[30:] = lr[0] / 10.0
lr[60:] = lr[0] / 100.0
lr[90:] = lr[0] / 1000.0

tf.reset_default_graph()
if use_gpu:
    # added to control the gpu memory
    print("GPU\n")
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model = denoiser(sess)
        if phase == 'train':
            denoiser_train(model, lr=lr)
        elif phase == 'test':
            denoiser_test(model)
        else:
            print('[!]Unknown phase')
            exit(0)
else:
    print("CPU\n")
    with tf.Session() as sess:
        model = denoiser(sess)
        if phase == 'train':
            denoiser_train(model, lr=lr)
        elif phase == 'test':
            denoiser_test(model)
        else:
            print('[!]Unknown phase')
            exit(0)
