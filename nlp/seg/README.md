seg
==============================
分词

Train your model
--------------------
自己训练模型

### seg model
#### Folder Structureseg
```shell
/nlp
./seg
..__init__.py
..bilstm_model.py
..bilstm_train.py
..bilstm_predicted.py
..data_reader.py
..data_Config.py
../data
.../demo
....train.dat
....dev.dat
....test.dat
../model
.../demo
....

```
#### 准备数据
下载复旦微博数据集https://github.com/FudanNLP/NLPCC-WordSeg-Weibo
```shell
/NLPCC-WordSeg-Weibo
./datasets
..nlpcc2016-word-seg-train.dat
..nlpcc2016-wordseg-dev.dat
..nlpcc2016-wordseg-test.dat
..
.
```
将数据集重命名为: 'train.dat', 'dev.dat', 'test.dat'. 
文件中每一行代表一个句子, 句子的格式为: "word1 word2 ...", 中间用空格隔开

```shell
#train.dat
回首 来 时 的 路 ， 坚定 的 信念 载 着 我们 走 了 很 远 。
专栏 ： 中国 汽车 租赁业 将 迎来 发展 良机
```
####从ftp下载已经存在的demo数据
....略
`


#### 指定 data_path
将分好的三个文件放入/nlp/seg/data/demo下
你也可以指定数据文件目录, 创建目录 .../seg/data/'your_folder' and .../seg/model/'your_folder'
更改 data_Config.py中的路径到你自己创建的文件夹

#### Running script
默认训练为双向lstm
```shell

python bilstm_train.py #双向lstm 

```
如果需要训练单向lstm
更改data_Config.py中类LargeConfigChinese中bi_direction = False  # LSTM 
```python
class LargeConfigChinese(object):
    """Large config."""
    init_scale = 0.04
    learning_rate = 0.5
    max_grad_norm = 10#最大梯度
    num_layers = 2
    num_steps = 30
    hidden_size = 128
    max_epoch = 5
    max_max_epoch = 55
    keep_prob = 1.0
    lr_decay = 1 / 1.15 #学习速率
    batch_size = 1  # single sample batch
    vocab_size = 6000
    target_num = 5  # seg tagging tag number for Chinese
    bi_direction = True  # LSTM or BiLSTM
```

#### 训练的模型将保存在 ../nlp/seg/model/demo下

