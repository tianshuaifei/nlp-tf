
import numpy as np

#预测数据转换，对不足一个时间序列的数据进行补全
def predicted_iterator(word_data, tag_data, batch_size, num_steps):
    word_data = np.array(word_data, dtype=np.int32)

    data_len = len(word_data)
    batch_len = data_len // batch_size
    xArray = np.zeros([batch_size, batch_len], dtype=np.int32)
    yArray = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        xArray[i] = word_data[batch_len * i:batch_len * (i + 1)]
        yArray[i] = tag_data[batch_len * i:batch_len * (i + 1)]
    if (batch_len) % num_steps == 0:
        epoch_size = (batch_len) // num_steps
    else:
        epoch_size = (batch_len) // num_steps + 1
    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")
    for i in range(epoch_size):
        x = xArray[:, i * num_steps:(i + 1) * num_steps]
        y = yArray[:, i * num_steps:(i + 1) * num_steps]
        if i == epoch_size - 1:
            yield (np.column_stack((x, np.zeros([batch_size, num_steps - np.shape(x)[1]], dtype=np.int32))),
                   np.column_stack((y, np.zeros([batch_size, num_steps - np.shape(x)[1]], dtype=np.int32))))
        else:
            yield (x, y)

#分词结果转换
def char_to_word(words ,tag):
    sent = ""
    i = 0
    while i < len(tag):
        if tag[i] == 'E' or tag[i] == 'S':
            sent = sent + words[i]
            sent = sent + "\t"
        else:
            sent = sent + words[i]
        i = i + 1
    sents = sent.rstrip().split("\t")
    return (words,sents)

