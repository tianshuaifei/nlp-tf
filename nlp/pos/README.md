POS (Part-of-Speech) Tagging
==============================
词性标注

Train your model
--------------------
自己训练模型

###POS model
#### Folder Structure
```shell
/nlp
./pos
..lstm_model.py
..lstm_train.py
..lstm_predicted.py
..reader.py
..config.py
../data
...train.txt
...dev.txt
...test.txt
../model
...
...
```
#### Prepare corpus
First, prepare your corpus and split into 3 files: 'train.txt', 'dev.txt', 'test.txt'.
Each line in the file represents one annotated sentence, in this format: "word1/tag1 word2/tag2 ...", separated by white space.

```python
#train.txt
充满/v  希望/n  的/u  新/a  世纪/n  ——/w  一九九八年/t  新年/t  讲话/n  （/w  附/v  图片/n  １/m  张/q  ）/w  
```


