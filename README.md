# TextCNN

TextCNN classifier model. Can be used to do both english and chinese sentence classification

## For train
	python3 train.py \
			--vocab_file= \
			--train_data= \
			--dev_data= \
			--ckpt_dir= \
			--num_class= \
			--language= \
			--gpu= \

	-vocab_file sample: {word0:1, word1:2, word2:3, ... , wordN:N+1}, save the dictionary by one line.
	-train_data sample: each line for one data sample, like this: {'class': class, 'id': id, 'text':text} (must include 'class' and 'text')
	-dev_data: same structure as train_data
	-ckpt_dir: directory for storing the model
	-num_class: number of total classes
	-language: 'EN' for english, 'CH' for chinese
	-gpu: default is '-1'

	Other default params:
	--embedding_dim = 128
	--kernel_sizes = '3,4,5'
	--num_kernels = 128
	--dropout_keep_prob = 0.7
	--num_words = 64 (number of word kept in each sentence)
	--batch_size = 32
	--num_epochs = 40
	--evaluate_every = 100 (evaluate model on dev every # steps)

## For Predict
	python3 predict.py \
			--vocab_file= \
			--num_class= \
			--ckpt= \
			--language= \
			--gpu= \

	-ckpt: the path of the model to be loaded.

	Other default params:
	--kernel_sizes = '3,4,5'
	
	if args == []: predict one sentence each time
	if args == [dataset]: predict the dataset and print the evaluate result.

## For Use
	from XXX/TextCNN.predict import Predictor
	class TextCnnClassifier():
		def __init__(self, model=XXX):
			self.predictor = Predictor()
			self.predictor.load_specific_model(model_path=model)
		def get_result(self, text):
			_class, _score = self.predictor.predict_sentence(text)
			# _class is the predicted class number, _score include the score for each class.
			# So, _score[_class] represent the score for the predicted class.
