import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import utils
from utils.utils import get_args
from utils.logger import Logger
from utils.dirs import create_dirs
from utils.config import process_config
from models.mnist_logistic_model import MnistLogisticModel
from trainers.mnist_logistic_trainer import MnistLogisticTrainer


def main():
    global config
    try:
        args = get_args()
        config = process_config(args.config)
        # args.config = 'test_example.json'
        print(args.config)
        print(config.learning_rate)
        print(config.display_step)
        print(config.checkpoint_dir)
        print(config.summary_dir)
        print(config.training_epochs)
    except:
        print("Missing argument")
        exit(0)
    # create_dirs([config.summary_dir, config.checkpoint_dir])  already created dirs

    data = input_data.read_data_sets("/tmp/data/", one_hot=True)

    # sess = tf.Session()
    with tf.Session() as sess:

        model = MnistLogisticModel(config)

        logger = Logger(sess, config)

        trainer = MnistLogisticTrainer(sess, model, data, config, logger)

        trainer.train()

    # sess.close()


if __name__ == "__main__":
    main()

    # CLI: python main_example.py --config test_example.json
