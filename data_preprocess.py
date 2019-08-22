from open_data import word_filter
import optparse

def get_word_dict_from_files(list_of_files, dict_filename, language):
  '''
  Build word dictionary from train and dev pydict file.
  :param list_of_files: [file1, file2, ...]
  :param dict_filename: dictionary file name
  :param language: 'EN' for english, 'CH' for chinese
  :return: write the word dictionary into 'dict_filename' as one line
  '''
  word_dict = {}
  num = 1
  for file in list_of_files:
    f = open(file, 'r').readlines()
    for line in f:
      text = eval(line)['text'].lower()
      if language == 'EN':
        text_info = word_filter(text).split()
      elif language == 'CH':
        text_info = [item for item in text.replace(' ','')]
      else:
        print('language should be set to EN or CH')
        assert 1 == 2
      for word in text_info:
        if word not in word_dict:
          word_dict[word] = num
          num += 1
  f_o = open(dict_filename, 'w')
  f_o.write(str(word_dict))

def main():
  usage = "usage: %prog [options]"
  parser = optparse.OptionParser(usage=usage)
  parser.add_option("--files_list", type=str, default='train.pydict, test.pydict')
  parser.add_option("--dict_filename", type=str, default='./models/intention-1300')
  parser.add_option("--language", default='EN')
  (options, args) = parser.parse_args()
  print(options.files_list)
  files_list = [item.strip() for item in options.files_list.split(',')]
  get_word_dict_from_files(files_list, options.dict_filename, options.language)

if __name__ == '__main__':
  main()