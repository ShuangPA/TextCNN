import optparse
import tensorflow as tf
import numpy as np
import os
import open_data as open_data
from text_CNN import TextCNN

usage = "usage: %prog [options]"
parser = optparse.OptionParser(usage=usage)
parser.add_option("--vocab_file", type=str,
                  default='happy_word_dict.txt',
                  help='english vocab file')
parser.add_option("--l2_reg_lambda", type=float, default=0.0,
                  help='L2 regularization lambda (default: 0.0)')
parser.add_option("--embedding_dim", type=int, default=128,
                  help='Dimensionality of character embedding(default: 128)')
parser.add_option("--kernel_sizes", type=str, default='3,4,5',
                  help='Comma-separated kernel sizes (default: 1,2,3,4,5)')
parser.add_option("--num_kernels", type=int, default=128,
                  help='Number of filters per filter size (default: 128)')
parser.add_option("--num_words", type=int, default=64,
                  help='Number of words kept in each sentence (default: 64)')
parser.add_option("--num_class", type=int, default=196)
parser.add_option("--ckpt", type=str, default='./models/intention-1300')
parser.add_option("--language", default='EN')
parser.add_option("--gpu", default='-1')
(options, args) = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = options.gpu

class Predictor(object):
  def __init__(self):
    self._graph = tf.Graph()
    with self._graph.as_default():
      self._model = TextCNN(
        sequenceLength = options.num_words,
        numClasses=options.num_class,
        vocabSize=10000,
        embeddingSize=options.embedding_dim,
        kernelSizes=list(map(int, options.kernel_sizes.split(","))),
        numKernels=options.num_kernels,
        l2RegLambda=options.l2_reg_lambda
      )
    self._sess = tf.Session(graph=self._graph)

  def load_specific_model(self, model_path):
    print(f"loading model from: '{model_path}")
    with self._graph.as_default():
      tf.train.Saver().restore(
        self._sess,
        model_path
      )

  def predict_dataset(self, data_set):
    model = self._model
    batch_data = open_data.open_pred_data(data_set, options.vocab_file, options.num_words, options.language)
    score, prediction = self._sess.run(
      fetches=[
        model.w_scores,
        model.predictions,
      ],
      feed_dict={
        model.inputX: batch_data,
        model.dropout_keep_prob: 1,
      }
    )
    _classes = prediction
    _score = score
    pred_score = [max(temp) for temp in _score]
    return _classes, pred_score

  def predict_sentence(self, sentence):
    model = self._model
    temp = open_data.one_sentence(sentence.strip(), options.vocab_file, options.num_words, options.language)
    batch_data = np.array(temp)
    score, prediction = self._sess.run(
      fetches=[
        model.w_scores,
        model.predictions,
      ],
      feed_dict={
        model.inputX: batch_data,
        model.dropout_keep_prob: 1,
      }
    )
    _classes = prediction[0]
    _score = score[0]
    return _classes, _score

def main():
  predictor = Predictor()
  predictor.load_specific_model(options.ckpt)
  if len(args) == 0:
    while True:
      line = input('please enter sentence:\n')
      _class, _score = predictor.predict_sentence(line)
      print(_class)
      print(_score[_class])
  data = args[0]
  classes, scores = predictor.predict_dataset(data)
  f = open(data, 'r').readlines()
  true_class = []
  for line in f:
    true_class.append(eval(line)['class'])
  # print(classes)
  # print(true_class)
  # print(scores)
  assert len(true_class) == len(classes)
  _eval = open_data.calc_precision_recall_fvalue(true_class, classes)
  print(_eval)


if __name__ == '__main__':
  main()
