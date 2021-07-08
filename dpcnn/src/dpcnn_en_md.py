import pickle
import numpy as np
from tqdm import tqdm
from sklearn import metrics
import sys

from mindspore import context
from mindspore import Tensor, Model
from mindspore import dtype as mstype
from mindspore.train.callback import LossMonitor
from mindspore.nn import Accuracy
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.train.callback import Callback
import mindspore
import mindspore.nn as nn
import mindspore.numpy as mdnp
import mindspore.dataset as ds
import mindspore.ops as ops

from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer


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


# 判断是否 Debug 模式
debug = True  # if sys.gettrace() else False

# 获取设备
device = 'CPU'
# context = context.PYNATIVE_MODE if debug else context.GRAPH_MODE
context.set_context(mode=context.PYNATIVE_MODE, device_target=device)

# 加载并处理数据集
UNK, PAD, CLS = '[UNK]', '[PAD]', '[CLS]'
vocal_file = 'vocal.pkl'
embeddings_file = 'embeddings_en.npz'
word_to_id = pickle.load(open(vocal_file, 'rb'))
embeddings = np.load(embeddings_file)['embeddings']
print('embeddings_size: {}'.format(len(embeddings)))


def load_dataset(pos, neg, pad_size=32):
    dataset = []

    with open(pos, 'r', encoding='ISO-8859-1') as f:
        for line in tqdm(f):
            words = process(line)
            tag = 1
            sentence_ids = []
            if not words or len(words) == 0:
                continue
            for word in words:
                if word in word_to_id:
                    sentence_ids.append(word_to_id[word])
                else:
                    sentence_ids.append(word_to_id[UNK])
            while len(sentence_ids) < pad_size:
                sentence_ids.append(word_to_id[PAD])
            sentence_ids = sentence_ids[:pad_size]
            dataset.append((sentence_ids, tag))

    with open(neg, 'r', encoding='ISO-8859-1') as f:
        for line in tqdm(f):
            words = process(line)
            tag = 0
            sentence_ids = []
            if not words or len(words) == 0:
                continue
            for word in words:
                if word in word_to_id:
                    sentence_ids.append(word_to_id[word])
                else:
                    sentence_ids.append(word_to_id[UNK])
            while len(sentence_ids) < pad_size:
                sentence_ids.append(word_to_id[PAD])
            sentence_ids = sentence_ids[:pad_size]
            dataset.append((sentence_ids, tag))

    return dataset


data = load_dataset('rt-polarity.pos', 'rt-polarity.neg')
np.random.shuffle(data)

p0, p1, p2, p3 = 0, int(len(data) * 6 / 10), int(len(data) * 8 / 10), len(data)
train_data = data[p0:p1]
dev_data = data[p1:p2]
test_data = data[p2:p3]
print('{} {} {}'.format(len(train_data), len(dev_data), len(test_data)))


class DatasetGenerator(object):

    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        x = np.array(self.dataset[index][0], dtype=np.int)
        y = np.array(self.dataset[index][1], dtype=np.int)
        return x, y
        # return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


# dataset.split 似乎有 bug, Accuracy 直接起飞, 所以就下面这样了
train_dataset_generator = DatasetGenerator(train_data)
train_dataset = ds.GeneratorDataset(train_dataset_generator, ["data", "label"])
train_dataset = train_dataset.batch(batch_size=128)

dev_dataset_generator = DatasetGenerator(dev_data)
dev_dataset = ds.GeneratorDataset(dev_dataset_generator, ["data", "label"])
dev_dataset = dev_dataset.batch(batch_size=128)

test_dataset_generator = DatasetGenerator(test_data)
test_dataset = ds.GeneratorDataset(test_dataset_generator, ["data", "label"])
test_dataset = test_dataset.batch(batch_size=128)


# -------------------------------------------- model --------------------------------------------
class DPCNN(nn.Cell):

    def __init__(self):
        super(DPCNN, self).__init__()
        self.embedding_dim = 300
        self.kernels = 250
        self.dropout = 0.5
        self.pad_size = 32
        self.num_classes = 2
        self.vocab_size = embeddings.shape[0]

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim,
                                      embedding_table=Tensor(embeddings, mstype.float32))
        self.region_embedding = nn.Conv2d(1, self.kernels, (3, self.embedding_dim), pad_mode='valid')
        self.conv = nn.Conv2d(self.kernels, self.kernels, (3, 1), pad_mode='valid')
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (0, 0)), mode='CONSTANT')
        self.padding2 = nn.Pad(paddings=((0, 0), (0, 0), (0, 1), (0, 0)), mode='CONSTANT')
        self.relu = nn.ReLU()
        self.fc = nn.Dense(self.kernels, self.num_classes)

    def construct(self, x):
        out = self.embedding(x)
        # out = out.unsqueeze(1)
        out = mdnp.expand_dims(out, 1)
        out_res = self.region_embedding(out)
        out = self._conv(out_res)
        out = self._conv(out)
        out = out + out_res

        # while out.size()[2] > 2:
        out = self._block(out)
        out = self._block(out)
        out = self._block(out)
        out = self._block(out)
        # out = self._block(out)

        out = out.squeeze()
        out = self.fc(out)

        return out

    def _block(self, x):
        out = self.padding2(x)
        out_res = self.max_pool(out)
        out = self._conv(out_res)
        out = self._conv(out)
        out = out + out_res
        return out

    def _conv(self, x):
        out = self.padding1(x)
        out = self.relu(out)
        out = self.conv(out)
        return out
# -------------------------------------------- model --------------------------------------------


# configuration
learning_rate = 0.001
weight_decay = 0.0001
epochs = 10
net = DPCNN()
optimizer = nn.Adam(net.trainable_params(), learning_rate=learning_rate)
loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)


# callback
class EvalCheckpoint(Callback):

    def __init__(self, model, net, dev_dataset, steps):
        self.model = model
        self.net = net
        self.dev_dataset = dev_dataset
        self.steps = steps
        self.counter = 0
        self.best_acc = 0

    def step_end(self, run_context):
        self.counter = self.counter + 1
        if self.counter % self.steps == 0:
            self.counter = 0
            acc = self.model.eval(self.dev_dataset, dataset_sink_mode=False)
            if acc['Accuracy'] > self.best_acc:
                self.best_acc = acc['Accuracy']
                mindspore.save_checkpoint(self.net, 'dpcnn.ckpt')
            print("{}".format(acc))


# train
model = Model(net, loss_fn=loss, optimizer=optimizer, metrics={"Accuracy": Accuracy()})
model.train(epoch=epochs, train_dataset=train_dataset, callbacks=[LossMonitor(10), EvalCheckpoint(model, net, dev_dataset, 10)])

param_dict = mindspore.load_checkpoint('dpcnn.ckpt')
mindspore.load_param_into_net(net, param_dict)
model = Model(net, loss_fn=loss, optimizer=None, metrics={"Accuracy": Accuracy()})
acc = model.eval(test_dataset, dataset_sink_mode=False)
print("{}".format(acc))
