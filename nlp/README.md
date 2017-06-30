Modules
========

* NLP 模块
    * seg  分词
    * pos  词性标注
    * dep  依存分析 
    * ner  命名实体识别
    * trunk 组块识别 
    * Sentiment 情感分析 
    * element 情感要素抽取 

#### 总的目录结构
```shell
/zutnlp-tf
./docs      #文档目录
./data
./zutnlp
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
..seg_predicted.py  #分词测试
..pos_predicted.py
..pos_predicted_blstm.py  #词性标注测试
..README.md
```
####每个模块目录
例如：分词模块

```shell
/zutnlp
./seg
../data                 #训练数据存放目录
.../demo
....train.dat
....dev.dat
....test.dat
../model                #训练模型存放目录
.../demo
..__init__.py
..seg_model_bilstm.py   #模型
..reader.py             #读取数据
..README.md             #说明文件
....

```

分词
========
* 下载模型
在zutnlp-tf/zutnlp下
```python 
#coding=utf-8
import zutnlp.download as dl
model="seg"
dl.get_model(data)

```
* 分词测试
在zutnlp-tf/zutnlp下
直接运行：

```python
python seg_predicted.py

```
或者：
```python
#coding=utf-8
from __future__ import unicode_literals

from zutnlp import seg_predicted as seg

word = ['我', '们', '都', '是', '好', '孩', '子','，','，']
model=seg.load_model()
result=model.predict(word)
print (result)

```