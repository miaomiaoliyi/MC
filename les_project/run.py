import os
import pickle
import argparse
import logging
from collections import OrderedDict

from dataset import LESDataset
from vocab import Vocab
from model import Model
from utils.preprocess import load_data_set, save_data


def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser('Reading Comprehension on military dataset')
    parser.add_argument('--prepare', action='store_true',
                        help='create the directories, prepare the vocabulary and embeddings')
    parser.add_argument('--train', action='store_true',
                        help='train the model')
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate the model on dev set')
    parser.add_argument('--predict', action='store_true',
                        help='predict the answers for test set with trained model')
    parser.add_argument('--gpu', type=str, default='0',
                        help='specify gpu device')

    train_settings = parser.add_argument_group('train settings')
    train_settings.add_argument('--optim', default='adam',
                                help='optimizer type')
    train_settings.add_argument('--learning_rate', type=float, default=0.001,
                                help='learning rate')
    train_settings.add_argument('--weight_decay', type=float, default=0,
                                help='weight decay')
    train_settings.add_argument('--dropout_keep_prob', type=float, default=1,
                                help='dropout keep rate')
    train_settings.add_argument('--batch_size', type=int, default=32,
                                help='train batch size')
    train_settings.add_argument('--epochs', type=int, default=10,
                                help='train epochs')
    train_settings.add_argument('--continue', type=bool, default=False,
                                help='train epochs')

    model_settings = parser.add_argument_group('model settings')
    model_settings.add_argument('--algo', choices=['BIDAF', 'MLSTM'], default='BIDAF',
                                help='choose the algorithm to use')
    model_settings.add_argument('--embed_size', type=int, default=300,
                                help='size of the embeddings')
    model_settings.add_argument('--hidden_size', type=int, default=150,
                                help='size of LSTM hidden units')
    model_settings.add_argument('--max_p_len', type=int, default=500,
                                help='max length of passage')
    model_settings.add_argument('--max_q_len', type=int, default=60,
                                help='max length of question')
    model_settings.add_argument('--max_a_len', type=int, default=200,
                                help='max length of answer')

    path_settings = parser.add_argument_group('path settings')
    path_settings.add_argument('--train_files',
                               default='../data/preprocessed/trainset.json',
                               help='file that contain the preprocessed train data')
    path_settings.add_argument('--dev_files',
                               default='../data/preprocessed/devset.json',
                               help='file that contain the preprocessed dev data')
    path_settings.add_argument('--test_files',
                               default='../data/preprocessed/testset.json',
                               help='file that contain the preprocessed test data')
    path_settings.add_argument('--vocab_dir', default='../data/vocab',
                               help='the dir to save vocabulary')
    path_settings.add_argument('--model_dir', default='../data/models',
                               help='the dir to store models')
    path_settings.add_argument('--result_dir', default='../data/results',
                               help='the dir to output the results')
    path_settings.add_argument('--summary_dir', default='../data/summary',
                               help='the dir to write tensorboard summary')
    path_settings.add_argument('--log_path', default='../data/log/les.log',
                               help='path of the log file. If not set, logs are printed to console')

    restore_settings = parser.add_argument_group('restore settings')
    restore_settings.add_argument('--model', default='model1',
                                  help='path of the model to select')
    restore_settings.add_argument('--model_epoch', default='16',
                                  help='path of the model to select')
    return parser.parse_args()


def prepare(args):
    logger = logging.getLogger("brc")
    logger.info('Checking the data files...')

    logger.info('Preparing the directories...')
    for dir_path in [args.vocab_dir, args.model_dir, args.result_dir, args.summary_dir, args.log_path]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    les_data = LESDataset(max_p_len=args.max_p_len, max_q_len=args.max_q_len, train_file=args.train_files)

    vocab = Vocab(lower=True)
    for word in les_data.word_iter('train'):
        vocab.add(word)
    vocab.randomly_init_embeddings(args.embed_size)
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'wb') as f:
        pickle.dump(vocab, f)

    logger.info('Done with preparing!')


def train(args):
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as f:
        vocab = pickle.load(f)
    # print(vocab.size(), vocab.embed_dim)
    les_data = LESDataset(args.max_p_len, args.max_q_len, args.train_files)

    model = Model(vocab, args)

    # 是否继续以前的模型训练
    if args.continus:
        model.restore(model_dir=os.path.join(args.model_dir, args.model),
                      model_prefix=args.algo + '_' + args.model_epoch)

    model.train(les_data, args.epochs, args.batch_size, save_dir=args.model_dir,
                save_prefix=args.algo,
                dropout_keep_prob=args.dropout_keep_prob,
                evaluate=False)


def evaluate(args):
    """
    evaluate the trained model on dev files
    """
    logger = logging.getLogger("les")
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as f:
        vocab = pickle.load(f)

    les_data = LESDataset(args.max_p_len, args.max_q_len, args.train_files)
    les_data.train_dev_split()

    logger.info('Restoring the model...')
    model = Model(vocab, args)
    model.restore(model_dir=os.path.join(args.model_dir, args.model), model_prefix=args.algo + '_' + args.model_epoch)

    logger.info('Evaluating the model on dev set...')
    dev_batches = les_data.gen_mini_batches('dev', args.batch_size, vocab, shuffle=False)

    dev_loss, dev_bleu_rouge = model.evaluate(
        dev_batches, result_dir=args.result_dir, result_prefix='dev.predicted2')

    logger.info('Loss on dev set: {}'.format(dev_loss))
    logger.info('Result on dev set: {}'.format(dev_bleu_rouge))
    logger.info('Predicted answers are saved to {}'.format(os.path.join(args.result_dir)))


def predict(args):
    logger = logging.getLogger("les")

    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as f:
        vocab = pickle.load(f)

    les_data = LESDataset(args.max_p_len, args.max_q_len, test_file=args.test_files)

    logger.info('Restoring the model...')
    model = Model(vocab, args)
    model.restore(model_dir=os.path.join(args.model_dir, args.model), model_prefix=args.algo + '_' + args.model_epoch)

    logger.info('Predicting answers for test set...')
    test_batches = les_data.gen_mini_batches('test', args.batch_size, vocab, shuffle=False)
    model.evaluate(test_batches, result_dir=args.result_dir, result_prefix='test.predicted')


def gen_final_file(predict_file_path='../data/results/test.predicted.json',
                   raw_data_path='../data/raw_data/question.json',
                   save_file_path='../data/results/predict.json'):
    """
    生成最终的提交文件
    :param predict_file_path: 预测文件路径
    :param raw_data_path: 原始文件路径
    :param save_file_path: 保存文件路径
    :return:
    """
    datas_raw = load_data_set(raw_data_path)
    predict_datas = load_data_set(predict_file_path)

    data_final = []
    for data_raw in datas_raw:
        data_temp = OrderedDict({"article_id": data_raw['article_id'], "questions": []})
        for paris in data_raw['questions']:
            pair_predict = predict_datas.pop(0)
            if paris['questions_id'] == pair_predict['questions_id']:
                data_temp["questions"].append(OrderedDict({"questions_id": pair_predict['questions_id'],
                                                           "answer": pair_predict['answer']}))
        data_final.append(data_temp)

    save_data(data_final, save_file_path)


def run():
    args = parse_args()

    logger = logging.getLogger("les")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if args.log_path:
        file_handler = logging.FileHandler(args.log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info('Running with args : {}'.format(args))

    if args.prepare:
        prepare(args)
    if args.train:
        train(args)
    if args.evaluate:
        evaluate(args)
    if args.predict:
        predict(args)


if __name__ == '__main__':
    run()
    # gen_final_file()
