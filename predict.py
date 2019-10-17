#encoding:utf-8
from data_helper import get_sequence_length, get_training_word2vec_vectors, read_vocab
from rnn_model import *
import  tensorflow as tf
import tensorflow.contrib.keras as kr
import os
import numpy as np
import jieba
import re
import heapq
import codecs

base_dir = 'data/'
train_dir = os.path.join(base_dir, 'cnews.train.txt')
test_dir = os.path.join(base_dir, 'cnews.test.txt')
val_dir = os.path.join(base_dir, 'cnews.val.txt')
vocab_dir = os.path.join(base_dir, 'cnews.vocab.txt')
words_embeding=os.path.join(base_dir, 'cnews.embeding.txt')
vector_word_npz= os.path.join(base_dir, 'cnews.embeding.npz')
vocab_filename=os.path.join(base_dir, 'cnews.vocab.txt')
def predict(sentences):
    config = RNNConfig()
    config.pre_trianing = get_training_word2vec_vectors(vector_word_npz)
    model = TextRNN(config)
    save_dir = './checkpoints/textcnn'
    save_path = os.path.join(save_dir, 'best_validation')
    _,word_to_id=read_vocab(vocab_filename)
    input_x= process_file(sentences,word_to_id,max_length=config.seq_length)
    labels = {0:'体育',
              1:'财经',
              2:'房产',
              3:'家居',
              4:'教育',
              5:'科技',
              6:'时尚',
              7:'时政',
              8:'游戏',
              9:'娱乐'
              }

    feed_dict = {
        model.input_x: input_x,
        model.keep_prob: 1,
        model.sequence_lengths: get_sequence_length(input_x)
    }
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)
    y_prob=session.run(tf.nn.softmax(model.logits), feed_dict=feed_dict)
    y_prob=y_prob.tolist()
    cat=[]
    for prob in y_prob:
        top2= list(map(prob.index, heapq.nlargest(1, prob)))
        cat.append(labels[top2[0]])
    tf.reset_default_graph()
    return  cat

def sentence_cut(sentences):
    """
    Args:
        sentence: a list of text need to segment
    Returns:
        seglist:  a list of sentence cut by jieba

    """
    re_han = re.compile(u"([\u4E00-\u9FD5a-zA-Z0-9+#&\._%]+)")  # the method of cutting text by punctuation
    with codecs.open('./data/stopwords.txt','r',encoding='utf-8') as f:
            stopwords=[line.strip() for line in f.readlines()]
    contents=[]
    for sentence in sentences:
        words=[]
        blocks = re_han.split(sentence)
        for blk in blocks:
            if re_han.match(blk):
                # seglist = jieba.lcut(blk)
                seglist = list(blk)
                words.extend([w for w in seglist if w not in stopwords])
        contents.append(words)
    return  contents


def process_file(sentences,word_to_id,max_length=600):
    """
    Args:
        sentence: a text need to predict
        word_to_id:get from def read_vocab()
        max_length:allow max length of sentence
    Returns:
        x_pad: sequence data from  preprocessing sentence

    """
    data_id=[]
    # seglist=sentence_cut(sentences)
    seglist=list(sentences)
    for i in range(len(seglist)):
        data_id.append([word_to_id[x] for x in seglist[i] if x in word_to_id])
    x_pad=kr.preprocessing.sequence.pad_sequences(data_id,max_length)
    return x_pad





if __name__ == '__main__':
    print('predict random five samples in test data.... ')
    import random
    sentences=[]
    labels=[]
    with codecs.open('./data/cnews.test.txt','r',encoding='utf-8') as f:
        sample=random.sample(f.readlines(),10)
        for line in sample:
            try:
                line=line.rstrip().split('\t')
                assert len(line)==2
                sentences.append(line[1])
                labels.append(line[0])
            except:
                pass
    cat=predict(sentences)
    for i,sentence in enumerate(sentences,0):
        print ('----------------------the text-------------------------')
        print (sentence[:50]+'....')
        print('the orginal label:%s'%labels[i])
        print('the predict label:%s'%cat[i])

