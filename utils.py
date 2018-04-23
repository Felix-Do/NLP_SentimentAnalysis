import gensim
from gensim.models import KeyedVectors
import time, math, random, datetime
import numpy as np
import csv
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize, PunktSentenceTokenizer


w2v_limit_default = 80000
w2v_filepath_default = "GoogleNews-vectors-negative300.bin"
data_filepath_default = "tweets_clean.csv"
# data_filepath_default = "tweets_clean_short.csv"

# the lines below are for first time running, to download the needed library, comment them out after
# nltk.download()
nltk.download("stopwords")  # download stopwords
nltk.download('punkt')      # download tokenize

class Timestamp():
    def __init__(self):
        return None
    
    def get(self):
        dt_now = datetime.datetime.now() # current time in pieces
        stamp = ""
        stamp = stamp + self.format(dt_now.year % 100)
        stamp = stamp + self.format(dt_now.month)
        stamp = stamp + self.format(dt_now.day)
        stamp = stamp + "_"
        stamp = stamp + self.format(dt_now.hour)
        stamp = stamp + self.format(dt_now.minute)
        stamp = stamp + self.format(dt_now.second)
        return stamp

    def format(self, value=0, digit=2):
        return format(value, "0" + str(digit) + "d")


class Data():
    def __init__(self, fixed_word_count_=24, w2v_filepath_=w2v_filepath_default, w2v_limit_=w2v_limit_default):
        self.max_word_count = 0
        self.shuffle_all_data = True
        self.fixed_word_count = fixed_word_count_
        self.stop_words = set(stopwords.words("english"))
        self.tweet_stopwords = ["#", ",", "."]
        self.load_w2v_model(w2v_filepath_, w2v_limit_)
        self.data_index = [0] * 3
        self.prepare_data()
        return None
    
    def text2words(self, text_input=""):
        text_cleansed = text_input
        text_cleansed = text_cleansed.replace('_', ' ')
        text_cleansed = text_cleansed.replace('-', ' ')
        text_cleansed = text_cleansed.replace('.', ' ')
        text = re.sub(r'[^\w\s]', ' ', text_cleansed)
        words = word_tokenize(text)
        words = np.array([w for w in words if not w in self.tweet_stopwords])
        return words

    def text2words_filter(self, text_input=""):
        words = self.text2words(text_input)
        words_filtered = np.array([w for w in words if not w in self.stop_words])
        return words_filtered

    def words2matrix(self, words=[]):
        matrix = np.zeros([self.fixed_word_count, self.w2v_dim])
        word_index = -1
        for w in words:
            if w in self.model.wv:
                word_index += 1
                if self.max_word_count < word_index + 1:
                    self.max_word_count = word_index + 1
                if word_index >= self.fixed_word_count:
                    continue
                word_vec = np.array(self.model.wv[w])
                matrix[word_index] = word_vec
        matrix = np.transpose(matrix)
        return matrix

    def text2matrix(self, text_input="", filtering=True):
        words = np.array([])
        if filtering == True:
            words = self.text2words_filter(text_input)
        else:
            words = self.text2words(text_input)
        matrix = self.words2matrix(words)
        return matrix

    def sentiment_int2vec(self, sent_int=0):
        if sent_int == 1:
            return np.array([0, 1])
        else:
            return np.array([1, 0])

    def load_w2v_model(self, w2v_filepath_=w2v_filepath_default, w2v_limit_=w2v_limit_default, binary_=True):
        self.w2v_filepath = w2v_filepath_
        self.w2v_limit = w2v_limit_
        time_start = time.time()
        self.model = KeyedVectors.load_word2vec_format(self.w2v_filepath, limit=self.w2v_limit, binary=binary_)
        time_finish = time.time()
        time_elapsed = time_finish - time_start
        print("Word2Vec model loaded:", self.w2v_limit, "words in", np.round(time_elapsed*1000)/1000, "seconds")
        common_words = ['the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'I']
        self.w2v_dim = 300
        for cw in common_words:
            if cw in self.model.wv:
                self.w2v_dim = len(self.model.wv[cw])
                break
        return 1
    
    def prepare_data(self, filepath=data_filepath_default):
        dataset_read = []
        with open(filepath, encoding="ISO-8859-1") as f:
            data = csv.reader(f)
            for line in data:
                # print(line)
                line_total = ""
                for line_piece in line:
                    line_total = line_total + str(line_piece)
                if len(line_total) <= 2: continue
                data_output = int(line_total[0])
                data_input = line_total[2 : None]
                dataset_read.append([data_input, data_output])
        
        # shuffle data?
        if self.shuffle_all_data:
            print("shuffling all data before splitting")
            np.random.shuffle(dataset_read)

        # section data
        mult = 1000
        total_len = len(dataset_read)
        print("total length of dataset:", total_len)
        test_len = 2000
        val_len = 2000
        train_len = math.floor((total_len - test_len - val_len) / mult) * mult
        count_break = [None] * 4
        count_break[0] = 0
        count_break[1] = train_len
        count_break[2] = train_len + val_len
        count_break[3] = train_len + val_len + test_len
        self.data_index = [0] * 3

        self.data_train_in, self.data_train_out, self.data_train_len = self.split_grouped_data(dataset_read[count_break[0] : count_break[1]])
        self.data_val_in,   self.data_val_out,   self.data_val_len   = self.split_grouped_data(dataset_read[count_break[1] : count_break[2]])
        self.data_test_in,  self.data_test_out,  self.data_test_len  = self.split_grouped_data(dataset_read[count_break[2] : count_break[3]])
        
        print("data splitted:  train=" + str(self.data_train_len) + "  val=" + str(self.data_val_len) + "  test=" + str(self.data_test_len) )
        print("max word count: ", self.max_word_count)
        return 1
    
    def split_grouped_data(self, data_group=[]):
        np.random.shuffle(data_group)
        data_len = len(data_group)
        data_in  = np.zeros([data_len, self.w2v_dim, self.fixed_word_count])
        data_out = np.zeros([data_len, 2])
        for i in range(data_len):
            data_in[i]  = self.text2matrix(data_group[i][0])
            data_out[i] = self.sentiment_int2vec(data_group[i][1])
        return data_in, data_out, data_len
    
    def next_batch(self, group=0, batch_size=100, specific_start_index=-1):
        if group not in range(3): return [[]] * 2
        data_in = self.data_train_in
        data_out = self.data_train_out
        if group == 1:
            data_in = self.data_val_in
            data_out = self.data_val_out
        if group == 2:
            data_in = self.data_test_in
            data_out = self.data_test_out
        length = len(data_in)
        start_index = self.data_index[group]
        if specific_start_index in range(length):
            start_index = specific_start_index
        if start_index in range(length):
            start_index = 0
        end_index = start_index + batch_size
        if batch_size < 0:
            start_index = 0
            end_index = length
        if end_index > length:
            end_index = length
        batch_size = end_index - start_index
        batch_in = data_in[start_index : end_index]
        batch_out = data_out[start_index : end_index]
        if not (specific_start_index in range(length)):
            self.data_index[group] = end_index
        return batch_in, batch_out


# [TESTING]

# data = Data()
# timestamp = Timestamp()
# print(data.data_train_len, data.data_val_len, data.data_test_len)
# print("max word count: ", data.max_word_count)
# test_batch_in, test_batch_out = data.next_batch(0, 200)
# print("\nbatch test:")
# print(np.shape(test_batch_in))
# print(test_batch_in[0])
# print(test_batch_in[1])
# print(test_batch_in)


