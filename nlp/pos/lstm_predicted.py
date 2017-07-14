# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import tensorflow as tf
import numpy as np
import glob
import nlp.util.data_read as reader
import nlp.pos.reader as pos_reader
import nlp.pos.model as pos_model
from nlp.pos.config import LargeConfigChinese
from nlp.pos.config import data_config

class ModelLoader(object):
    def __init__(self,dataConfig):
        self.dataConfig=dataConfig
        print("Starting new Tensorflow session...")
        self.session = tf.Session()
        print("Initializing pos_tagger class...")
        self.model = self._init_pos_model(self.session, self.dataConfig)

    def predict(self, words):
        '''
        Coding: utf-8 for Chinese Characters
        Return tuples of [(word, tag),...]
        '''
        tagging = self._predict_pos_tags(self.session, self.model, words, self.dataConfig)
        return tagging

    def _init_pos_model(self, session, dataConfig):
        config = LargeConfigChinese
        config.batch_size = 1
        config.num_steps = 10 # iterator one token per time

        with tf.variable_scope("pos_var_scope"):  # Need to Change in Pos_Tagger Save Function
            model = pos_model.POSTagger(is_training=False, config=config)  # save object after is_training

        if len(glob.glob(dataConfig.ckpt_path + '.data*')) > 0:  # file exist with pattern: 'pos.ckpt.data*'
            print("Loading model parameters from %s" % dataConfig.ckpt_path)
            all_vars = tf.global_variables()
            model_vars = [k for k in all_vars if k.name.startswith("pos_var_scope")]
            tf.train.Saver(model_vars).restore(session, dataConfig.ckpt_path)
        else:
            print("Model not found, created with fresh parameters.")
            session.run(tf.global_variables_initializer())
        return model

    def _predict_pos_tags(self, session, model, words, dataConfig):

        word_data = pos_reader.sentence_to_word_ids(dataConfig.data_path, words)
        tag_data = [0] * len(word_data)
        #       state = session.run(model.initial_state)

        predict_id = []
        for step, (x, y) in enumerate(reader.predicted_iterator(word_data, tag_data, model.batch_size, model.num_steps)):
            fetches = [model.cost, model.logits, tf.no_op()]  # eval_op define the m.train_op or m.eval_op
            feed_dict = {}
            feed_dict[model.input_data] = x
            feed_dict[model.targets] = y
            cost, logits, _ = session.run(fetches, feed_dict)
            x,y=logits.shape
            if x>1:
                for s in logits:
                    predict_id.append(int(np.argmax(s)))
            else:
                predict_id.append(int(np.argmax(logits)))
        predict_tag = pos_reader.word_ids_to_sentence(dataConfig.data_path, predict_id[:len(word_data)])
        #        return zip(words, predict_tag)
        return (words, predict_tag)


def load_model():
    dataConfig = data_config()
    return ModelLoader(dataConfig)

if __name__ == '__main__':
    test = load_model()
    word = ['我们', '都是', '好孩子']
    result=test.predict(word)
    print(result)
