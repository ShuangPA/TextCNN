import optparse
import tensorflow as tf
import numpy as np
import os
import open_data
from text_CNN import TextCNN
import sys
abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(abs_path + '/insight_nlp/')
from measure import Measure

if __name__ == '__main__':
  usage = "usage: %prog [options]"
  parser = optparse.OptionParser(usage=usage)
  parser.add_option("--vocab_file", type=str,
                    default='./happy_word_dict.txt',
                    help='english vocab file')
  parser.add_option("--train_data", type=str,
                    default='../data/train.pydict',
                    help='Data source for training.')
  parser.add_option("--dev_data", type=str,
                    default='../data/train.pydict',
                    help='Data source for testing.')
  parser.add_option("--embedding_dim", type=int, default=128,
                    help='Dimensionality of character embedding(default: 128)')
  parser.add_option("--kernel_sizes", type=str, default='3,4,5',
                    help='Comma-separated kernel sizes (default: 1,2,3,4,5)')
  parser.add_option("--num_kernels", type=int, default=128,
                    help='Number of filters per filter size (default: 128)')
  parser.add_option("--dropout_keep_prob", type=float, default=.7,
                    help='Dropout keep probability (default: 0.5)')
  parser.add_option("--l2_reg_lambda", type=float, default=0.0,
                    help='L2 regularization lambda (default: 0.0)')
  parser.add_option("--num_words", type=int, default=64,
                    help='Number of words kept in each sentence (default: 64)')
  parser.add_option("--batch_size", type=int, default=32,
                    help='Batch Size (default: 64)')
  parser.add_option("--num_epochs", type=int, default=40,
                    help='Number of training epochs (default: 200)')
  parser.add_option("--evaluate_every", type=int, default=100,
                    help='Evaluate model on dev every # steps (default: 100)')
  parser.add_option("--ckpt_dir", type=str, default='./models6')

  parser.add_option("--do_train", default=True)
  parser.add_option("--language", default='EN')
  parser.add_option("--num_class", type=int, default=196)
  parser.add_option("--gpu", default='-1')
  (options, args) = parser.parse_args()
  os.environ["CUDA_VISIBLE_DEVICES"] = options.gpu

  print("Loading data...")

  if options.do_train:
    train_data, train_label = open_data.open_data_and_labels(options.train_data, options.vocab_file, options.num_words, options.language)
    xEVAL, yEVAL = open_data.open_data_and_labels(options.dev_data, options.vocab_file, options.num_words, options.language)

    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(train_label)))
    xTrain = train_data[shuffle_indices]
    yTrain = train_label[shuffle_indices]

    del train_data, train_label

  sess = tf.Session()
  with sess.as_default():
    cnn = TextCNN(
      sequenceLength = options.num_words,
      numClasses=options.num_class,
      vocabSize=10000,
      embeddingSize=options.embedding_dim,
      kernelSizes=list(map(int, options.kernel_sizes.split(","))),
      numKernels=options.num_kernels,
      l2RegLambda=options.l2_reg_lambda
    )

    global_step = tf.Variable(0, name='globalStep', trainable=False)
    optimizer = tf.train.AdamOptimizer(1e-3)
    grads_and_vars = optimizer.compute_gradients(cnn.loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    sess.run(tf.global_variables_initializer())
    _saver = tf.train.Saver(tf.global_variables(), max_to_keep=1000)

    def trainStep(xBatch, yBatch):
      feed_dict = {cnn.inputX: xBatch, cnn.inputY: yBatch,
                  cnn.dropout_keep_prob: options.dropout_keep_prob}
      op, step, loss, accuracy = sess.run(
        [train_op, global_step, cnn.loss, cnn.accuracy], feed_dict)
      print(f'step {step}, loss {loss}, acc {accuracy}')

    def devStep(xBatch, yBatch, step):
      print(f'save model at step: {step}')
      _saver.save(sess, f'{options.ckpt_dir}/intention', global_step=step)

      feed_dict = {cnn.inputX: xBatch, cnn.inputY: yBatch,
                  cnn.dropout_keep_prob: 1}
      step, loss, accuracy, predictions = sess.run(
        [global_step, cnn.loss, cnn.accuracy, cnn.predictions], feed_dict)
      all_true_label = open_data.onehot_2_labels(yBatch)
      all_pred_label = predictions
      eval_output = Measure.calc_precision_recall_fvalue(all_true_label, all_pred_label)
      print(f'step {step}, loss {loss}, acc {accuracy}')
      print(eval_output)

    if options.do_train:
      batches = open_data.batchIter(list(zip(xTrain, yTrain)),
                          options.batch_size, options.num_epochs)
      for batch in batches:
        xBatch, yBatch = zip(*batch)
        trainStep(xBatch, yBatch)
        current_step = tf.train.global_step(sess, global_step)
        if current_step % options.evaluate_every == 0:
          print("\nEvaluation:")
          devStep(xEVAL, yEVAL, current_step)
          print('')
