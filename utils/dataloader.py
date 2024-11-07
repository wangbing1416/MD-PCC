import torch
import random
import tqdm
import pandas as pd
import json
import numpy as np
import nltk
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader

label_dict = {
    "real": 0,
    "fake": 1
}

category_dict = {
    "2000": 0, "2001": 0, "2002": 0, "2003": 0, "2005": 0, "2004": 0,
    "2006": 0, "2007": 0, "2008": 0, "2009": 0, "2010": 0, "2011": 0,
    "2012": 0, "2013": 0, "2014": 0, "2015": 0, "2016": 0,
    "2017": 1, "2018": 2
}

def word2input(texts, max_len, tokenizer, path):
    token_ids = []
    print("\nData Processing: tokenizing text from {}".format(path))
    for text in tqdm.tqdm(texts):
        token_ids.append(tokenizer.encode(text, max_length=max_len, add_special_tokens=True, padding='max_length', truncation=True))
    token_ids = torch.tensor(token_ids)
    masks = torch.zeros(token_ids.shape[0], token_ids.shape[1])
    mask_token_id = tokenizer.pad_token_id
    for i, tokens in enumerate(token_ids):
        masks[i, :] = (tokens != mask_token_id)
    return token_ids, masks


def get_dataloader(path, emo_path, max_len, batch_size, shuffle, aug_prob, pretrain_name):
    data_list = json.load(open(path, 'r', encoding='utf-8'))
    df_data = pd.DataFrame(columns=('content', 'label'))
    print("\nData Processing: loading data from {}".format(path))
    for item in tqdm.tqdm(data_list):
        tmp_data = {}
        tmp_data['content'] = item['content']
        if 'chinese' in pretrain_name: tmp_data['label'] = label_dict[item['label']]
        else: tmp_data['label'] = item['label']  # real-0, fake-1
        tmp_data['year'] = item['time'].split(' ')[0].split('-')[0]
        df_data = df_data.append(tmp_data, ignore_index=True)
    emotion = np.load(emo_path).astype('float32')
    emotion = torch.tensor(emotion)
    content = df_data['content'].to_numpy()
    label = torch.tensor(df_data['label'].astype(int).to_numpy())
    year = torch.tensor(df_data['year'].apply(lambda c: category_dict[c]).astype(int).to_numpy())
    tokenizer = BertTokenizer.from_pretrained(pretrain_name)
    content_token_ids, content_masks = word2input(content, max_len, tokenizer, path)

    dataset = TensorDataset(content_token_ids, content_masks, emotion, label, year)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=16,
        pin_memory=True,
        shuffle=shuffle
    )
    return dataloader


def get_dataloader_noemo(path, emo_path, max_len, batch_size, shuffle, aug_prob, pretrain_name):
    data_list = json.load(open(path, 'r', encoding='utf-8'))
    df_data = pd.DataFrame(columns=('content', 'label'))
    print("\nData Processing: loading data from {}".format(path))
    for item in tqdm.tqdm(data_list):
        tmp_data = {}
        tmp_data['content'] = item['content']
        if 'chinese' in pretrain_name: tmp_data['label'] = label_dict[item['label']]
        else: tmp_data['label'] = item['label']  # real-0, fake-1
        tmp_data['year'] = item['time'].split(' ')[0].split('-')[0]
        df_data = df_data.append(tmp_data, ignore_index=True)
    content = df_data['content'].to_numpy()
    label = torch.tensor(df_data['label'].astype(int).to_numpy())
    tokenizer = BertTokenizer.from_pretrained(pretrain_name)
    content_token_ids, content_masks = word2input(content, max_len, tokenizer, path)

    dataset = TensorDataset(content_token_ids, content_masks, label)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=16,
        pin_memory=True,
        shuffle=shuffle
    )
    return dataloader

