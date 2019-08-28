#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
dssm - tf
    data_helper.py

数据处理类
把全部数据读入后，处理成 tf.SparseTensor，再生成 batch iterator

@author fengfei
@date 2019.08.28

"""

import json
import random
import string
import sys
import time
import os

import tensorflow as tf

reload(sys)
sys.setdefaultencoding('utf-8')

class DataHelper():
    """
    用于将字符串逐字转为 one hot
    """
    def __init__(self, wordListFile, trainFile, batchSize, negNum, epochNum):
        """
        @ wordListFile
            常见字表，汉字 + 拼音 + 数字，一行，以空格分割
        @ trainFile
            训练数据，query \t positive_title \t negative_title \001 negative_title
        @ negNum
            负样本个数
        @ batchSize
        @ epochNum

        """
        self.train_file = trainFile  # 训练数据路径
        self.batch_size = batchSize  # batch size
        self.epoch_num = epochNum
        self.neg_num = negNum  # 负样本个数
        self.id_to_word = dict()
        self.word_to_id = dict()

        index = 1  # 对于不在常用词典的字取 0
        with open(wordListFile, "r") as fin:  # 常用汉字
            for line in fin:
                for word in line.strip().split(" "):
                    word = unicode(word)
                    if word in self.word_to_id:
                        print str(word)
                        # raise ValueError('Duplicate')
                        continue
                    self.id_to_word[index] = word
                    self.word_to_id[word] = index
                    index += 1

        self.vocab_len = index + 1  # 词典长度
        print "len of load word list: {}".format(self.vocab_len)


    def sentence_to_id(self, sentence):
        ids = [self.word_to_id[word] if word in self.word_to_id else 0 for word in unicode(sentence)]
        # for index, word in enumerate(unicode(sentence)):
        #     print str(word), ids[index]
        return ids


    def sentence_to_onehot(self, sentence):
        one_hot_dict = dict()
        ids = self.sentence_to_id(sentence)
        for index in ids:
            if index not in one_hot_dict:
                one_hot_dict[index] = 0
            one_hot_dict[index] += 1
        one_hot_tuple = sorted(one_hot_dict.items(), key=lambda x: x[0])
        return one_hot_tuple


    def parse_data(self):
        row = 0
        query_indices, positive_indices, negative_indices = list(), list(), list()
        query_vals, positive_vals, negative_vals = list(), list(), list()

        with open(self.train_file, "r") as fin:
            for line in fin:
                arr = line.strip().split("\t")

                if len(arr) != 3:
                    continue
                query, positive, tmp_negatives = arr[0], arr[1], arr[2]
                negatives = tmp_negatives.split("\001")

                if len(negatives) != self.neg_num:
                    continue

                query_one_hot = self.sentence_to_onehot(query)
                for index, num in query_one_hot:
                    query_vals.append(num)
                    query_indices.append([row, index])

                positive_ont_hot = self.sentence_to_onehot(positive)
                for index, num in positive_ont_hot:
                    positive_vals.append(num)
                    positive_indices.append([row, index])

                for i in range(self.neg_num):
                    negative_ont_hot = self.sentence_to_onehot(negatives[i])
                    for index, num in negative_ont_hot:
                        negative_vals.append(num)
                        negative_indices.append([self.neg_num * row + i, index])
                row += 1

        return tf.SparseTensor(values=query_vals, indices=query_indices, dense_shape=[row + 1, self.vocab_len]),\
            tf.SparseTensor(values=positive_vals, indices=positive_indices, dense_shape=[row + 1, self.vocab_len]), \
            tf.SparseTensor(values=negative_vals, indices=negative_indices, dense_shape=[(row + 1) * self.neg_num, self.vocab_len])


    def get_batch_iterator(self):
        query_tensor, positive_tensor, negative_tensor = self.parse_data()

        query_data =  tf.data.Dataset.from_tensor_slices(query_tensor)
        positive_data = tf.data.Dataset.from_tensor_slices(positive_tensor)
        negative_data = tf.data.Dataset.from_tensor_slices(negative_tensor)

        dataset = tf.data.Dataset.zip((query_data, positive_data, negative_data))
        return dataset.shuffle(10000).repeat(self.epoch_num).batch(self.batch_size)


if __name__ == "__main__":
    data_helper = DataHelper("../data/RELY_commonWords", "../data/trainset", 5, 2, 1)
    print "测试: ", data_helper.sentence_to_id("测试")
    print "测试: ", data_helper.sentence_to_onehot("测试")

    dataset = data_helper.get_batch_iterator()
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    with tf.compat.v1.Session() as sess:
        for i in range(1):
            print(sess.run(next_element))