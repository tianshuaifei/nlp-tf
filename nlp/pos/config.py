



class data_config(object):

    data_path = "data/demo"  # path to find corpus vocab file
    train_dir = "model/demo"  # path to find model saved checkpoint file

    train_path = "data/demo/train.dat"
    dev_path = "data/demo/dev.dat"
    test_path = "data/demo/test.dat"
    word_to_id="data/demo/word_to_id"
    tag_to_id="data/demo/tag_to_id"

    ckpt_path = "model/demo/pos_bilstm.ckpt"


class LargeConfigChinese(object):
    """Large config."""
    init_scale = 0.04
    learning_rate = 0.5
    max_grad_norm = 10
    num_layers = 2
    num_steps = 30
    hidden_size = 128
    max_epoch = 5
    max_max_epoch = 55
    keep_prob = 1.0
    lr_decay = 1 / 1.15
    batch_size = 1  # single sample batch
    vocab_size = 80000
    target_num = 44  # POS tagging tag number for Chinese
    bi_direction = True  # LSTM or BiLSTM