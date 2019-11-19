from textwrap import wrap
import re
import itertools
import tensorflow as tf
import tfplot
from tensorflow.contrib.tensorboard.plugins import projector
import matplotlib
from  matplotlib.figure import Figure
import datetime
import time
import math
import numpy as np
import os
from PIL import Image

def confusion_metric(labels, predictions, num_classes, name=None):
    """
    Function to create confusion metric which can be plugged into confusion matrix plot in tensorboard.

    Parameters
    ----------
        labels: batch labels
        predictions: batch of predicted labels
        num_classes: number of different classes
        name: name of scope under which variables are created.

    Returns
    -------
        metric: tuple with variable and assign_add operation which aggregates the values.
    """

    with tf.name_scope(name) as scope:
        # Compute evaluation metrics.
        batch_confusion = tf.confusion_matrix(
          tf.squeeze(labels),
          predictions,
          num_classes=num_classes, 
          name='batch_confusion',
          dtype=tf.int64
        )

        # Create an accumulator variable to hold the counts
        confusion = tf.Variable(
          tf.zeros([num_classes, num_classes], dtype=tf.int64),
          name='confusion'
        )

        # Create the update op for doing a "+=" accumulation on the batch
        metric = (confusion, confusion.assign_add(batch_confusion))
    return metric


def plot_confusion_matrix(cm, label_names, title='Confusion matrix', tensor_name = 'confusion_matrix/image', normalize=False):
    ''' 
    Parameters:
        cm                              : The aggregated confusion matrix as tf.confusion_matrix
        label_names                     : This is a lit of labels which will be used to display the axix labels
        title='Confusion matrix'        : Title for your matrix
        tensor_name = 'MyFigure/image'  : Name for the output summay tensor

    Returns:
        summary: TensorFlow summary 

    Other itema to note:
        - Depending on the number of category and the data , you may have to modify the figzie, font sizes etc. 
        - Currently, some of the ticks dont line up due to rotations.
    '''
    
    if normalize == True:
        cm = tf.to_float(cm)
        cm = cm / tf.math.reduce_sum(cm, axis=0)

    
    np.set_printoptions(precision=3)

    def create_matplotlib_figure(cm):
        fig = Figure(figsize=(3.5, 3.5), dpi=320, facecolor='w', edgecolor='k')
        #fig, ax = tfplot.subplots(figsize=(3.5, 3.5), dpi=320, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(1, 1, 1)

        im = ax.imshow(cm, cmap='Oranges')

        classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in label_names]
        classes = ['\n'.join(wrap(l, 40)) for l in classes]

        tick_marks = np.arange(len(classes))

        ax.set_xlabel('Predicted', fontsize=7)
        ax.set_xticks(tick_marks)
        c = ax.set_xticklabels(classes, fontsize=4, rotation=-90,  ha='center')
        ax.xaxis.set_label_position('bottom')
        ax.xaxis.tick_bottom()

        ax.set_ylabel('True Label', fontsize=7)
        ax.set_yticks(tick_marks)
        c = ax.set_yticklabels(classes, fontsize=4, va='center')
        ax.yaxis.set_label_position('left')
        ax.yaxis.tick_left()

        # Fix for matplotlib version 3.1.1
        if matplotlib.__version__ == '3.1.1':
            bottom, top = ax.get_ylim()
            ax.set_ylim(bottom + 0.5, top - 0.5)
        

        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize == True:
                ax.text(j, i, format(cm[i, j], '.3f'), horizontalalignment='center', fontsize=5, verticalalignment='center', color= 'black')
            else:
                ax.text(j, i, format(cm[i, j], 'd'), horizontalalignment='center', fontsize=5, verticalalignment='center', color= 'black')
        fig.set_tight_layout(True)
        return fig

    plot_op = tfplot.autowrap(create_matplotlib_figure)(cm)
    return tf.summary.image(tensor_name, tf.expand_dims(plot_op, axis=0))

def get_weight_histogram_summary(name):
    return tf.summary.histogram(name + '_hist', tf.get_collection(tf.GraphKeys.VARIABLES, name)[0])

class EmbeddingSaverHook(tf.train.SessionRunHook):

    def __init__(self, graph, params, values_name, values, images_name, labels_name, captions):
        super(EmbeddingSaverHook, self).__init__()

        self._classes = captions
        self._model_dir = params['model_dir']
        self._stain_code_size = params['stain_code_size']

        self.var = None

        self._labels = None
        self._images = None

        self._emb_values = []
       
        self._emb_labels = []
        self._emb_captions = []

        self._values_name = values_name
        self._labels_name = labels_name
        self._images_name = images_name
        self._graph = graph

        self._w = params['width']
        self._h = params['height']

        # 8192? Largest multiple of 2 below 10,000, so if image size is in 2^x, this will be an even number
        self._num_sprites_x = math.floor(8192 / self._w)
        self._num_sprites_y = math.floor(8192 / self._h)
        self._num_sprites = self._num_sprites_x * self._num_sprites_y


        self._emb_values_placeholder = tf.placeholder(tf.float32, name='emb_values_placeholder')
        self._emb_stain_placeholder = tf.placeholder(tf.float32, name='emb_stain_placeholder')
        self._emb_structure_placeholder = tf.placeholder(tf.float32, name='emb_structure_placeholder')

        self._embedding_var = tf.Variable(tf.zeros([self._num_sprites, values.shape[1]], dtype=tf.float32), name='emb_values')
        self._embedding_var_stain = tf.Variable(tf.zeros([self._num_sprites, self._stain_code_size], dtype=tf.float32), name='emb_stain')
        self._embedding_var_structure = tf.Variable(tf.zeros([self._num_sprites, values.shape[1] - self._stain_code_size], dtype=tf.float32), name='emb_structure')
        
        self._assign_op_var = self._embedding_var.assign(self._emb_values_placeholder,read_value=False)
        self._assign_op_stain = self._embedding_var_stain.assign(self._emb_stain_placeholder,read_value=False)
        self._assign_op_structure = self._embedding_var_structure.assign(self._emb_structure_placeholder,read_value=False)

        self._saver = tf.train.Saver([self._embedding_var, self._embedding_var_stain, self._embedding_var_structure])       

        self._master = Image.new(mode='RGB', size=(self._num_sprites_x*self._w, self._num_sprites_y*self._h), color=(0,0,0))
        self._x = 0
        self._y = 0
        

    def begin(self):
        self.var = self._graph.get_tensor_by_name(self._values_name)
        self._labels = self._graph.get_tensor_by_name(self._labels_name)
        self._images = self._graph.get_tensor_by_name(self._images_name)

    def before_run(self, run_context):
        return tf.train.SessionRunArgs([self.var, self._labels, self._images])

    def after_run(self, run_context, run_values):
        if len(self._emb_values) <= self._num_sprites:
            self._emb_values.extend(run_values[0][0])
    
            self._emb_labels.extend(run_values[0][1])
            self._emb_captions.extend([self._classes[x[0]] for x in run_values[0][1]])         
    
            if self._y < self._num_sprites_y:
                for image in run_values[0][2]:
                    if self._y < self._num_sprites_y:
                        x = self._x * self._w
                        y = self._y * self._h
                        image = np.interp(image, (image.min(), image.max()), (0, 255))
                        image = np.asarray(image, dtype=np.uint8)
                        image = Image.fromarray(image, mode='RGB')
                        self._master.paste(image, (x,y))
                        if self._x < self._num_sprites_x - 1:
                            self._x += 1
                        else:
                            self._x = 0
                            self._y += 1
            


    def end(self, session):
        global_step = tf.train.get_global_step(graph=self._graph)
        global_step_val = global_step.value().eval(session=session)
        metadata_filename = os.path.join(self._model_dir, 'projector/metadata-' + str(global_step_val)  + '.tsv')

        if not os.path.isdir(os.path.join(self._model_dir, 'projector')):
            os.makedirs(os.path.join(self._model_dir, 'projector'))
        with open(metadata_filename, 'w+') as f:
            f.write('Index\tCaption\tLabel\n')
            for idx in range(self._num_sprites):
                f.write('{:05d}\t{}\t{}\n'
                        .format(idx, self._emb_captions[idx], self._emb_labels[idx]))
            f.close()

        #Add sprites
        image_sprite_filename = os.path.join(self._model_dir, 'projector/' + 'sprites_10k_' + str(global_step_val)  + '.jpg')
        self._master.save(image_sprite_filename)        
        
        data = np.squeeze(self._emb_values[:self._num_sprites])
        session.run(self._assign_op_var, feed_dict={self._emb_values_placeholder: data})
        session.run(self._assign_op_stain, feed_dict={self._emb_stain_placeholder: data[:,:self._stain_code_size]})
        session.run(self._assign_op_structure, feed_dict={self._emb_structure_placeholder: data[:,self._stain_code_size:]})

        config = projector.ProjectorConfig()
        
        embedding_var = config.embeddings.add()
        embedding_var.tensor_name = self._embedding_var.name

        embedding_stain = config.embeddings.add()
        embedding_stain.tensor_name = self._embedding_var_stain.name

        embedding_structure = config.embeddings.add()
        embedding_structure.tensor_name = self._embedding_var_structure.name

        # Add metadata to the log
        embedding_var.metadata_path = metadata_filename
        embedding_var.sprite.image_path = image_sprite_filename
        embedding_var.sprite.single_image_dim.extend([self._w, self._h])

        # Add metadata to the log
        embedding_stain.metadata_path = metadata_filename
        embedding_stain.sprite.image_path = image_sprite_filename
        embedding_stain.sprite.single_image_dim.extend([self._w, self._h])

        # Add metadata to the log
        embedding_structure.metadata_path = metadata_filename
        embedding_structure.sprite.image_path = image_sprite_filename
        embedding_structure.sprite.single_image_dim.extend([self._w, self._h])
       

        writer = tf.summary.FileWriter(os.path.join(self._model_dir, 'projector/'), self._graph)
        
        projector.visualize_embeddings(writer, config)

        self._saver.save(session, os.path.join(self._model_dir, "projector/model_emb.ckpt"), global_step=global_step)
        pass

    def get_embeddings(self):
        return { 'values': self._emb_values, 'labels': self._emb_labels, 'captions': self._emb_captions }