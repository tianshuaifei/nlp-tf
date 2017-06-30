# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import tensorflow as tf
import numpy as np
import glob
import nlp.util.data_read as reader
import nlp.seg.data_reader as seg_reader
import nlp.seg.bilstm_model as seg_model
from nlp.seg.data_Config import data_config
from nlp.seg.data_Config import LargeConfigChinese


class ModelLoader(object):
    def __init__(self,dataConfig):
        self.dataConfig=dataConfig

        print("Starting new Tensorflow session...")
        self.session = tf.Session()
        print("Initializing seg_tagger class...")
        self.model = self._init_seg_model(self.session,self.dataConfig)

    def predict(self, words):
        '''
        Coding: utf-8 for Chinese Characters
        Return tuples of [(word, tag),...]
        '''
        tagging = self._predict_seg_tags(self.session, self.model, words,self.dataConfig)
        return tagging

    def _init_seg_model(self, session, dataConfig):
        config = LargeConfigChinese
        config.batch_size = 1
        config.num_steps = 30 # iterator one token per time

        with tf.variable_scope("seg_var_scope"):  # Need to Change in Pos_Tagger Save Function
            model = seg_model.SegTagger(is_training=False, config=config)  # save object after is_training

        if len(glob.glob(dataConfig.ckpt_path + '.data*')) > 0:  # file exist with pattern: 'pos.ckpt.data*'
            print("Loading model parameters from %s" % dataConfig.ckpt_path)
            all_vars = tf.global_variables()
            model_vars = [k for k in all_vars if k.name.startswith("seg_var_scope")]
            tf.train.Saver(model_vars).restore(session, dataConfig.ckpt_path)
        else:
            print("Model not found, created with fresh parameters.")
            session.run(tf.global_variables_initializer())
        return model

    def _predict_seg_tags(self, session, model, words, dataConfig):

        word_data = seg_reader.sentence_to_word_ids(dataConfig.word_to_id, words)
        tag_data = [0] * len(word_data)
        #       state = session.run(model.initial_state)

        predict_id = []
        for step, (x, y) in enumerate(reader.predicted_iterator(word_data, tag_data, model.batch_size, model.num_steps)):
            fetches = [model.cost, model.logits, tf.no_op()]  # eval_op define the m.train_op or m.eval_op
            feed_dict = {}
            feed_dict[model.input_data] = x
            feed_dict[model.targets] = y
            cost, logits, _ = session.run(fetches, feed_dict)
            print(logits)
            x,y=logits.shape
            if x>1:
                for s in logits:
                    predict_id.append(int(np.argmax(s)))
            else:
                predict_id.append(int(np.argmax(logits)))
        predict_tag = seg_reader.word_ids_to_sentence(dataConfig.tag_to_id, predict_id[:len(word_data)])
        #        return zip(words, predict_tag)
        return (words, predict_tag)

def load_model():
    dataConfig = data_config()
    return ModelLoader(dataConfig)
if __name__ == '__main__':
    test = load_model()
    word = ['我', '们', '都', '是', '好', '孩', '子','，','，']
    result=test.predict(word)
    print(result)