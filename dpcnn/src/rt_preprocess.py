import io
import pickle
import numpy as np
from tqdm import tqdm
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer


UNK, PAD, CLS = '[UNK]', '[PAD]', '[CLS]'


def process(line: str):
    # remove punctuation
    punctuation = ['!', '"', '#', '$', '%', '&', '\'', '(', ')',
                   '*', '+', '-', '.', '/', ':', ';', '<', '=',
                   '>', '?', '@', '[', '\\', ']', '^', '_', '`',
                   '{', '|', '}', '~']
    line = ''.join([char for char in line if char not in punctuation])

    # tokenization
    words = word_tokenize(line)

    # stopwords filtering
    # stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours',
    #               'ourselves', 'you', "you're", "you've", "you'll",
    #               "you'd", 'your', 'yours', 'yourself', 'yourselves',
    #               'he', 'him', 'his', 'himself', 'she', "she's", 'her',
    #               'hers', 'herself', 'it', "it's", 'its', 'itself',
    #               'they', 'them', 'their', 'theirs', 'themselves',
    #               'what', 'which', 'who', 'whom', 'this', 'that',
    #               "that'll", 'these', 'those', 'am', 'is', 'are',
    #               'was', 'were', 'be', 'been', 'being', 'have', 'has',
    #               'had', 'having', 'do', 'does', 'did', 'doing', 'a',
    #               'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as',
    #               'until', 'while', 'of', 'at', 'by', 'for', 'with',
    #               'about', 'against', 'between', 'into', 'through',
    #               'during', 'before', 'after', 'above', 'below', 'to',
    #               'from', 'up', 'down', 'in', 'out', 'on', 'off',
    #               'over', 'under', 'again', 'further', 'then', 'once',
    #               'here', 'there', 'when', 'where', 'why', 'how', 'all',
    #               'any', 'both', 'each', 'few', 'more', 'most', 'other',
    #               'some', 'such', 'no', 'nor', 'not', 'only', 'own',
    #               'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
    #               'will', 'just', 'don', "don't", 'should', "should've",
    #               'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain',
    #               'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't",
    #               'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't",
    #               'haven', "haven't", 'isn', "isn't", 'ma', 'mightn',
    #               "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan',
    #               "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren',
    #               "weren't", 'won', "won't", 'wouldn', "wouldn't"]
    # filtered_words = [word for word in words if word not in stop_words]

    # stemming
    porter = PorterStemmer()
    stemmed = [porter.stem(word) for word in words]

    return stemmed


def build_vocab(pos, neg, encoding='ISO-8859-1', max_size=1000000, min_freq=1):
    vocab_dic = {}
    print('building vocabulary...')
    files = [pos, neg]
    for path in files:
        with open(path, 'r', encoding=encoding) as f:
            for line in tqdm(f):
                words = process(line)
                if not words or len(words) == 0:
                    continue
                for word in words:
                    if word == '':
                        continue
                    vocab_dic[word] = vocab_dic.get(word, 0) + 1
    vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
    vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
    vocab_dic.update({UNK: len(vocab_dic)})
    vocab_dic.update({PAD: len(vocab_dic)})
    vocab_dic.update({CLS: len(vocab_dic)})
    print('finish building')
    return vocab_dic


word_to_id = build_vocab('rt-polarity.pos', 'rt-polarity.neg')
print(len(word_to_id))
vocal_file = 'vocal.pkl'
pickle.dump(word_to_id, open(vocal_file, 'wb'))


# https://fasttext.cc/docs/en/english-vectors.html
def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    data = {}
    # n, d = map(int, fin.readline().split())
    for line in tqdm(fin):
        word, vec = line.split(maxsplit=1)
        data[word] = np.array(list(map(float, vec.strip().split(' '))))
    return data


embedding_dim = 300
# word_vec = load_vectors('../wiki-news-300d-1M.vec')
word_vec = load_vectors('crawl-300d-2M.vec')
embeddings = np.zeros((len(word_to_id), embedding_dim), dtype=float)
for word, id in word_to_id.items():
    if word in word_vec:
        embeddings[id] = word_vec[word]
embeddings_file = 'embeddings_en.npz'
np.savez_compressed(embeddings_file, embeddings=embeddings)
