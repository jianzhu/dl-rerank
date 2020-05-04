import json
import os

import tensorflow as tf

from modeling.mtl.expert import Expert
from modeling.mtl.gate import Gate


class Tasks(tf.keras.layers.Layer):
    """Multi-gate Mixture-of-Experts

        ref:
         1) Recommending What Video to Watch Next: A Multitask Ranking System
         2) Multi-Task Deep Neural Networks for Natural Language Understanding
    """

    def __init__(self, config_dir, dropout_rate, gate_dropout_rate):
        super(Tasks, self).__init__()

        mtl_config = self.parse_mtl(config_dir)
        self.task_num = len(mtl_config['tasks'])
        self.experts = []
        for _ in range(self.task_num):
            self.experts.append(Expert(mtl_config['expert'], dropout_rate))

        self.create_gates(self.task_num, gate_dropout_rate)
        self.create_task(mtl_config)

    def call(self, inputs, training=False):
        """inputs contains following two tensor

           input:
              input tensor: shape (B, T, H)
              input sequence mask tensor: shape (B, T)
              tasks labels list: each label's shape (B, T, 1)

           output:
              output tensor: shape (B, T, H)
        """
        # apply experts on top of shared bottom layer
        # each expert output shape: (B, T, H')
        shared_bottom = inputs[0]
        expert_xs = []
        for expert in self.experts:
            expert_xs.append(expert(shared_bottom, training=training))

        # apply gates on top of shared bottom layer
        # each gate output shape: (B, T, H')
        gate_ws = []
        for gate in self.gates:
            gate_inputs = {'shared_bottom': shared_bottom, 'experts': expert_xs}
            gate_ws.append(gate(gate_inputs, training=training))

        # (B, T, 1)
        sequence_mask = inputs[1]
        weights = tf.expand_dims(tf.cast(sequence_mask, dtype=shared_bottom.dtype), axis=-1)
        # apply tasks on top of gates output layer
        tasks_predictions = {}
        tasks_loss = {}
        total_loss = 0

        labels = inputs[2]
        all_predictions = None
        for i, task in enumerate(self.tasks):
            # output shape: (B, T, 1)
            predictions = task['dense'](gate_ws[i])
            tasks_predictions[task['name']] = predictions
            if training:
                # label shape: (B, T, 1), weights shape: (B, T, 1)
                loss = tf.compat.v1.losses.log_loss(labels=labels[i], predictions=predictions, weights=weights)
                tasks_loss[task['name']] = loss
                total_loss += loss * task['weight']
            if all_predictions is None:
                all_predictions = task['weight'] * predictions
            else:
                all_predictions += task['weight'] * predictions
        tasks_predictions['predictions'] = all_predictions

        return [tasks_predictions, tasks_loss, total_loss]

    def parse_mtl(self, config_dir):
        path = os.path.join(os.path.join(config_dir, 'multi_task'), 'mtl.json')
        with tf.io.gfile.GFile(path) as f:
            mtl_config = json.loads(''.join([line for line in f.readlines()]))
        return mtl_config

    def create_gates(self, expert_num, gate_dropout_rate):
        self.gates = []
        for _ in range(expert_num):
            self.gates.append(Gate(expert_num, gate_dropout_rate))

    def create_task(self, mtl_config):
        self.tasks = []
        for task in mtl_config['tasks']:
            units = task['units']
            if units != 1:
                raise ValueError("Invalid task ouput units: {}".format(units))
            self.tasks.append({'dense': tf.keras.layers.Dense(units=units, activation='sigmoid'),
                                'weight': task['weight'], 'name': task['name']})


# import numpy as np
#
# config_dir = '/Users/zhujian/Work/Compute/dl-rerank/resources/config'
# tasks = Tasks(config_dir, 0.3, 0.1)
#
# shared_bottom = np.random.rand(2, 3, 4)
# sequence_mask = tf.sequence_mask(tf.constant([2, 3], dtype=tf.int64))
# labels = [np.random.randint(2, size=(2, 3, 1)),
#           np.random.randint(2, size=(2, 3, 1)),
#           np.random.randint(2, size=(2, 3, 1))]
#
# inputs = [shared_bottom, sequence_mask, labels]
#
# tasks_predictions, tasks_loss, total_loss = tasks(inputs)
# print('-------------task predictions----------------')
# print(tasks_predictions)
#
# print('---------------task loss-----------------')
# print(tasks_loss)
#
# print('total loss: {}'.format(total_loss))
