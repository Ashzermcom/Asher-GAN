"""
    Description: 写到这里我已经忘了后面该怎么写了，阅读此段代码将浪费你生命中宝贵的1分钟
    Author: zhouyuzhe@kingsoft.com
    Date: 2020
"""


class BaseTrainer:
    pass


class DefaultTrainer(BaseTrainer):
    def __init__(self):
        super(DefaultTrainer, self).__init__()

    def train(self):
        pass

    @staticmethod
    def _data_process():
        # build a dataset based on config file.
        dataset = build_dataset()
        return dataset

    @staticmethod
    def _init_model():
        model = build_model()
        model.setup(cfg)
        return model

    def _train():
        pass

    def _model_eval(self):
        pass
