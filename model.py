import os
import torch
import tqdm
import datetime
import copy
from sklearn.metrics import *
from transformers import BertModel
from utils.utils import data2gpu, Averager, metrics, Recorder
from models.layers import *


class BERTModel(torch.nn.Module):
    def __init__(self, pretrain_name, emb_dim, mlp_dims, dropout):
        super(BERTModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrain_name).requires_grad_(False)

        for name, param in self.bert.named_parameters():
            if name.startswith("encoder.layer.11"):
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.mlp = MLP(emb_dim, [mlp_dims], dropout)
        self.attention = MaskAttention(emb_dim)

    def forward(self, **kwargs):
        content, content_masks = kwargs['content'], kwargs['content_masks']

        bert_feature_content = self.bert(content, attention_mask=content_masks)[0]
        bert_feature_content, _ = self.attention(bert_feature_content, content_masks)

        output = self.mlp(bert_feature_content)
        return torch.sigmoid(output.squeeze(1)), bert_feature_content


class BERTEmoModel(torch.nn.Module):
    def __init__(self, pretrain_name, emb_dim, mlp_dims, dropout):
        super(BERTEmoModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrain_name).requires_grad_(False)

        for name, param in self.bert.named_parameters():
            if name.startswith("encoder.layer.11"):
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.fea_size = emb_dim
        if 'chinese' in pretrain_name: self.mlp = MLP(emb_dim * 2 + 47, [mlp_dims], dropout)
        else: self.mlp = MLP(emb_dim * 2 + 38, [mlp_dims], dropout)
        self.rnn = nn.GRU(input_size=emb_dim,
                          hidden_size=self.fea_size,
                          num_layers=1,
                          batch_first=True,
                          bidirectional=True)
        self.attention = MaskAttention(emb_dim * 2)

    def forward(self, **kwargs):
        inputs = kwargs['content']
        masks = kwargs['content_masks']
        emotion = kwargs['emotion']

        bert_feature = self.bert(inputs, attention_mask=masks)[0]
        feature, _ = self.rnn(bert_feature)
        feature, _ = self.attention(feature, masks)
        output = self.mlp(torch.cat([feature, emotion], dim=1))
        return torch.sigmoid(output.squeeze(1)), feature


class EANNModel(torch.nn.Module):
    def __init__(self, pretrain_name, emb_dim, mlp_dims, dropout):
        super(EANNModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrain_name).requires_grad_(False)
        self.embedding = self.bert.embeddings
        self.bertembedding = self.bert.embeddings
        self.bertencoder = self.bert.encoder
        domain_num = 3

        feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64}
        self.convs = cnn_extractor(feature_kernel, emb_dim)
        mlp_input_shape = sum([feature_kernel[kernel] for kernel in feature_kernel])
        self.classifier = MLP(mlp_input_shape, [mlp_dims], dropout)
        self.domain_classifier = nn.Sequential(MLP(mlp_input_shape, [mlp_dims], dropout, False), torch.nn.ReLU(),
                                               torch.nn.Linear(mlp_dims, domain_num))
        self.adapter = nn.ModuleList([self.convs, self.classifier, self.domain_classifier])

    def forward(self, **kwargs):
        inputs = kwargs['content']
        masks = kwargs['content_masks']
        alpha = kwargs['alpha']

        bert_feature = self.bert(inputs, attention_mask=masks)[0]

        feature = self.convs(bert_feature)
        output = self.classifier(feature)
        reverse = ReverseLayerF.apply
        domain_pred = self.domain_classifier(reverse(feature, alpha))
        return torch.sigmoid(output.squeeze(1)), domain_pred


class MDFENDModel(torch.nn.Module):
    def __init__(self, pretrain_name, emb_dim, mlp_dims, dropout):
        super(MDFENDModel, self).__init__()
        self.domain_num = 3
        self.num_expert = 5
        self.emb_dim = emb_dim
        self.bert = BertModel.from_pretrained(pretrain_name).requires_grad_(False)
        self.embedding = self.bert.embeddings
        self.bertencoder = self.bert.encoder

        feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64}
        expert = []
        for i in range(self.num_expert):
            expert.append(cnn_extractor(feature_kernel, emb_dim))
        self.expert = nn.ModuleList(expert)

        self.gate = nn.Sequential(nn.Linear(emb_dim * 2, mlp_dims),
                                  nn.ReLU(),
                                  nn.Linear(mlp_dims, self.num_expert),
                                  nn.Softmax(dim=1))

        self.attention = MaskAttention(emb_dim)

        self.domain_embedder = nn.Embedding(num_embeddings=self.domain_num, embedding_dim=emb_dim)
        self.classifier = MLP(320, [mlp_dims], dropout)

    def forward(self, **kwargs):
        inputs = kwargs['content']
        masks = kwargs['content_masks']
        domain_labels = kwargs['year']

        init_feature = self.bert(inputs, attention_mask=masks)[0]
        gate_input_feature, _ = self.attention(init_feature, masks)
        shared_feature = 0
        if self.training:
            idxs = torch.tensor([index for index in domain_labels]).view(-1, 1).cuda()
            domain_embedding = self.domain_embedder(idxs).squeeze(1)
        else:
            batchsize = inputs.size(0)
            domain_embedding = self.domain_embedder(torch.LongTensor(range(self.domain_num)).cuda()).squeeze(1).mean(dim=0, keepdim=True).expand(batchsize, self.emb_dim)

        gate_input = torch.cat([domain_embedding, gate_input_feature], dim=-1)
        gate_value = self.gate(gate_input)
        for i in range(self.num_expert):
            tmp_feature = self.expert[i](init_feature)
            shared_feature += (tmp_feature * gate_value[:, i].unsqueeze(1))

        label_pred = self.classifier(shared_feature)

        return torch.sigmoid(label_pred.squeeze(1)), shared_feature
