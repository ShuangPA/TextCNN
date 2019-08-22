#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 16:23:24 2018

@author: zhaoshuang
"""
from collections import defaultdict, Counter
import numpy as np
import re
EPSILON = 1e-6

def embedding(data, wordDict):
  dataOutput = []
  for line in data:
    wordSeparate = []
    for idx in range(len(line)):
      wordSeparate.append(line[idx])

    embeddingWord = []
    for char in wordSeparate:
      if char in wordDict.keys():
        embeddingWord.append(wordDict[char])
      else:
        embeddingWord.append(len(wordDict))
    dataOutput.append(np.array(embeddingWord))
  dataOutput = np.array(dataOutput)
  return dataOutput

def word_filter(input):
  temp = re.sub(u"[\u4e00-\u9fa5]+", "", input.strip())
  temp = temp.replace('.', ' .').replace(',', ' ,').replace(':', ' :')\
    .replace(';', ' ;').replace('?', ' ?').replace('!', ' !')
  temp = temp.lower()
  subs = [['haven’t','have not'],['aren’t','are not'],['won’t','will not'],
          ['didn’t','did not'],['wasn’t','was not'],['weren’t','were not'],
          ['couldn’t','could not'],['isn’t','is not'],['doesn’t','does not'],
          ['wouldn’t','would not'],['shouldn’t','should not'],['can’t','can not'],
          ['hasn’t','has not'],['don’t','do not'],
          ['we’ll','we will'],['she’ll','she will'],['he’ll','he will'],
          ['i’ll','i will'],['you’ll','you will'],['they’ll','they will'],
          ['there’s','there is'],['it’s','it is'],['that’s','that is'],
          ['she’s','she is'],['he’s','he is'],['what’s','what is'],
          ['you’re','you are'],['we’re','we are'],['they’re','they are'],
          ['they’ve','they have'],['i’ve','i have'],['you’ve','you have'],
          ['i’m','i am'],['i’d','i would']]
  for item in subs:
    temp = temp.replace(item[0], item[1])
    poss2 = item[0].replace("’","'")
    temp = temp.replace(poss2, item[1])
  return temp

def open_data_and_labels(path, vocab_file, num_words, language):
  vocab = eval(open(vocab_file).readlines()[0])
  raw_data = open(path, 'r').readlines()
  all_data = []
  all_label = []
  for line in raw_data:
    line = eval(line.strip())
    all_label.append(line['class'])
    if language == 'EN':
      en_text = line['text'].lower()
      en_text = word_filter(en_text)
      text_info = en_text.split()[:num_words]
    elif language == 'CH':
      ch_text = line['text'].lower()
      ch_text = ch_text.replace(' ', '')
      text_info = [item for item in ch_text][:num_words]
    else:
      print('language should be set EN or CH')
      assert 1 == 2
    temp = []
    for item in text_info:
      if item in vocab.keys():
        temp.append(int(vocab[item]))
      else:
        temp.append(0)
    temp = temp + [9999]*num_words
    all_data.append(np.array(temp[:num_words]))
  all_data = np.array(all_data)
  # print(all_data)
  all_data = all_data.astype(int)
  return [all_data, one_hot(all_label)]

def open_pred_data(path, vocab_file, num_words, language):
  vocab = eval(open(vocab_file).readlines()[0])
  raw_data = open(path, 'r').readlines()
  all_data = []
  for line in raw_data:
    line = eval(line.strip())['text'].lower()
    if language == 'EN':
      en_text = word_filter(line)
      text_info = en_text.split()[:num_words]
    elif language == 'CH':
      ch_text = line.replace(' ','')
      text_info = [item for item in ch_text][:num_words]
    else:
      print('language should be set EN or CH')
      assert 1 == 2
    temp = []
    for item in text_info:
      if item in vocab.keys():
        temp.append(int(vocab[item]))
      else:
        temp.append(0)
    temp = temp + [9999]*num_words
    all_data.append(np.array(temp[:num_words]))
  all_data = np.array(all_data)
  # print(all_data)
  all_data = all_data.astype(int)
  return all_data

def one_sentence(text, vocab_file, num_words, language):
  vocab = eval(open(vocab_file).readlines()[0])
  line = text.lower()
  if language == 'EN':
    en_text = word_filter(line.strip())
    text_info = en_text.split()[:num_words]
  elif language == 'CH':
    ch_text = line.strip().replace(' ','')
    text_info = [item for item in ch_text][:num_words]
  else:
    print('language should be set EN or CH')
    assert 1 == 2
  temp = []
  for item in text_info:
    if item in vocab.keys():
      temp.append(int(vocab[item]))
    else:
      temp.append(0)
  temp = temp + [9999]*num_words
  return np.array([np.array(temp[:num_words])])

def one_hot(lis):
  m = max(lis)
  out = []
  for i in range(len(lis)):
    temp = np.zeros(m + 1)
    temp[lis[i]] += 1
    out.append(temp)
  out = np.array(out)
  return out

def onehot_2_labels(input):
  temp = []
  for line in input:
    label = np.argmax(np.array(line))
    temp.append(label)
  return temp

def batchIter(data, batchSize, numEpochs, shuffle=True):
  """
  Generates a batch iterator for a dataset.
  """
  data = np.array(data)
  dataSize = len(data)
  numBatchesPerEpoch = int((len(data) - 1) / batchSize) + 1
  for epoch in range(numEpochs):
    # Shuffle the data at each epoch
    if shuffle:
      shuffleIndices = np.random.permutation(np.arange(dataSize))
      shuffledData = data[shuffleIndices]
    else:
      shuffledData = data
    for batchNum in range(numBatchesPerEpoch):
      startIndex = batchNum * batchSize
      endIndex = min((batchNum + 1) * batchSize, dataSize)
      yield shuffledData[startIndex:endIndex]

def calc_precision_recall_fvalue(true_labels, preded_labels):
  '''
  :return (recall, precision, f) for each label, and
  (average_f, weighted_f, precision) for all labels.
  '''
  assert len(true_labels) == len(preded_labels)
  true_label_num = defaultdict(int)
  preded_label_num = defaultdict(int)
  correct_label = defaultdict(int)

  for t_label, p_label in zip(true_labels, preded_labels):
    true_label_num[t_label] += 1
    preded_label_num[p_label] += 1
    if t_label == p_label:
      correct_label[t_label] += 1

  result = dict()
  label_stat = Counter(true_labels)
  for label in label_stat.keys():
    correct = correct_label.get(label, 0)
    recall = correct / (true_label_num.get(label, 0) + EPSILON)
    prec = correct / (preded_label_num.get(label, 0) + EPSILON)
    f_value = 2 * (recall * prec) / (recall + prec + EPSILON)
    result[label] = {
      "recall": round(recall, 4),
      "precision": round(prec, 4),
      "f": round(f_value, 4),
    }

  total_f = sum([result[label]["f"] for label in label_stat.keys()])
  avg_f_value = total_f / len(label_stat)
  result["average_f"] = round(avg_f_value, 4)

  total_f = sum([result[label]["f"] * label_stat.get(label, 0)
                 for label in label_stat.keys()])
  weighted_f_value = total_f / len(true_labels)
  result["weighted_f"] = round(weighted_f_value, 4)

  result["precision"] = round(sum(correct_label.values()) / len(true_labels),
                              4)

  return result