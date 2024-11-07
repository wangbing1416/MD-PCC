import os
import torch
import tqdm
import datetime
import copy
from .layers import *
import numpy as np
from sklearn.metrics import *
from transformers import BertModel
from utils.utils import data2gpu, data2gpu_noemo, Averager, metrics, Recorder
from utils.dataloader import get_dataloader, get_dataloader_noemo, word2input
from model import BERTModel, BERTEmoModel, EANNModel, MDFENDModel
from comet import Comet
from transformers import BertTokenizer

class Trainer():
    def __init__(self, config):
        self.args = config
        
        self.save_path = os.path.join(self.args.save_param_dir, self.args.model_name)
        if os.path.exists(self.save_path):
            self.save_param_dir = self.save_path
        else:
            self.save_param_dir = os.makedirs(self.save_path)

        

    def train(self, logger=None):
        nowtime = datetime.datetime.now().strftime("%m%d-%H%M")
        save_path = os.path.join(self.save_path, 'parameter_bert-' + self.args.dataset + '.pkl')
        if(logger):
            logger.info('start training......')
        if self.args.model_name == 'bert':
            self.detector = BERTModel(self.args.pretrain_name, self.args.emb_dim, self.args.inner_dim, self.args.dropout)
        elif self.args.model_name == 'bertemo':
            self.detector = BERTEmoModel(self.args.pretrain_name, self.args.emb_dim, self.args.inner_dim, self.args.dropout)
        elif self.args.model_name == 'eann':
            self.detector = EANNModel(self.args.pretrain_name, self.args.emb_dim, self.args.inner_dim, self.args.dropout)
        elif self.args.model_name == 'mdfend':
            self.detector = MDFENDModel(self.args.pretrain_name, self.args.emb_dim, self.args.inner_dim, self.args.dropout)
        else:
            self.detector = BERTModel(self.args.pretrain_name, self.args.emb_dim, self.args.inner_dim, self.args.dropout)
        recorder = Recorder(self.args.early_stop)

        if self.args.dataset == 'gossip' or self.args.dataset == 'weibo':
            dataloader = get_dataloader
            gpuload = data2gpu
        else:
            dataloader = get_dataloader_noemo
            gpuload = data2gpu_noemo

        train_loader = dataloader(self.args.data_path + 'train' + self.args.dataset_id + '.json', self.args.data_path + 'train_emo.npy',
                                      self.args.max_len, self.args.batch_size, shuffle=True, aug_prob=self.args.aug_prob, pretrain_name=self.args.pretrain_name)
        val_loader = dataloader(self.args.data_path + 'val' + self.args.dataset_id + '.json', self.args.data_path + 'val_emo.npy',
                                    self.args.max_len, self.args.batch_size, shuffle=False, aug_prob=self.args.aug_prob, pretrain_name=self.args.pretrain_name)
        test_loader = dataloader(self.args.data_path + 'test' + self.args.dataset_id + '.json', self.args.data_path + 'test_emo.npy',
                                     self.args.max_len, self.args.batch_size, shuffle=False, aug_prob=self.args.aug_prob, pretrain_name=self.args.pretrain_name)

        self.detector = self.detector.cuda()
        loss_fn = torch.nn.BCELoss()

        diff_part = ["bertModel.embeddings", "bertModel.encoder"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.detector.named_parameters() if any(nd in n for nd in diff_part)],
                "weight_decay": 0.0,
                "lr": self.args.lr
            },
            {
                "params": [p for n, p in self.detector.named_parameters() if not any(nd in n for nd in diff_part)],
                "weight_decay": 0.0,
                "lr": self.args.mlp_lr
            },
        ]
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, eps=self.args.adam_epsilon)

        logger.info("Training the fake news detector based on {}".format(self.args.pretrain_name))
        for epoch in range(self.args.epoch):
            self.detector.train()
            train_data_iter = tqdm.tqdm(train_loader)
            avg_loss = Averager()

            for step_n, batch in enumerate(train_data_iter):
                batch_data = gpuload(batch, use_cuda=True)
                label = batch_data['label']

                pred, _ = self.detector(**batch_data)
                loss = loss_fn(pred, label.float())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss.add(loss.item())
            logger.info('Training Epoch {}; Loss {}; '.format(epoch + 1, avg_loss.item()))

            results = self.test(val_loader)
            mark = recorder.add(results)  # early stop with validation metrics
            if mark == 'save':
                torch.save(self.detector.state_dict(), save_path)
            elif mark == 'esc':
                break
            else:
                continue

        logger.info("Stage: testing...")
        self.detector.load_state_dict(torch.load(save_path))

        future_results = self.test(test_loader)

        logger.info("test score: {}.".format(future_results))
        logger.info("lr: {}, aug_prob: {}, avg test score: {}.\n\n".format(self.args.lr, self.args.aug_prob, future_results['metric']))
        print('test results:', future_results)

        return future_results, save_path

    def test(self, dataloader):
        pred = []
        label = []
        self.detector.eval()
        data_iter = tqdm.tqdm(dataloader)

        if self.args.dataset == 'gossip' or self.args.dataset == 'weibo': gpuload = data2gpu
        else: gpuload = data2gpu_noemo

        with torch.no_grad():
            for step_n, batch in enumerate(data_iter):
                batch_data = gpuload(batch, use_cuda=True)
                batch_label = batch_data['label']

                batch_pred, batch_feature = self.detector(**batch_data)

                label.extend(batch_label.detach().cpu().numpy().tolist())
                pred.extend(batch_pred.detach().cpu().numpy().tolist())

        return metrics(label, pred)