Modules
========

* NLP 模块
    * seg  分词
    * pos  词性标注
    * dep  依存分析 
    * ner  命名实体识别
    * trunk 组块识别 
    * sentiment 情感分析 
    * element 情感要素抽取 

#### 总的目录结构
```shell
/nlp-tf
./docs      #文档目录
./nlp
../element
../chunk
../dep
../ner
../pos
../seg
../sentiment
../util
..__init__.py
..download.py  #从ftp下载数据
..README.md
```
####每个模块目录
例如：分词模块

```shell
/nlp
./seg
../data                 #训练数据存放目录
.../demo
....train.dat
....dev.dat
....test.dat
../model                #训练模型存放目录
.../demo
..__init__.py
..bilstm_model.py      #模型文件
..bilstm_train.py      #训练
..bilstm_predicted.py  #测试
..data_reader.py
..data_Config.py       #网络配置，数据配置
..README.md             #说明文件
....

```

分词
========
* 下载模型
在nlp-tf/nlp下
```python 
#coding=utf-8
import nlp.download as dl
model="seg"
dl.get_model(data)
暂不提供，需自己训练！！
```
* 分词测试
在nlp-tf/nlp/seg下
直接运行：

```python
python blistm_predicted.py

```
或者：
```python
#coding=utf-8
from __future__ import unicode_literals

from nlp.seg import blistm_predicted as seg

word = ['我', '们', '都', '是', '好', '孩', '子','，','，']
model=seg.load_model()
result=model.predict(word)
print (result)

```