import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5ForConditionalGeneration
from comet_cn import Comet
import torch
import argparse
import datetime
import json
import random
import re
from utils.generate_utils import dataprocess, extract_prompts_cn, link_prompts_cn, comet_prompts_cn
from utils.dataloader import get_dataloader

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='ours')
parser.add_argument('--max_len', type=int, default=150)
parser.add_argument('--icl_num', type=int, default=0)
parser.add_argument('--t5_threshold', type=float, default=0.8)
parser.add_argument('--comet_threshold', type=float, default=0.6)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--triplet_model_path', default= '../huggingface/mt5-large')
parser.add_argument('--comet_model_path', default = '../huggingface/comet-t5-zh')
parser.add_argument('--icl_path', default = './data/incontext_prompts_cn.json')
args = parser.parse_args()

data_path = {
    'gossip': './data/gossip/',
    'weibo': './data/weibo/',
    'ours': './data/ours/'
}
args.data_path = data_path[args.dataset]
incontext_prompt = json.load(open(args.icl_path, 'r', encoding='utf-8'))
for rel in list(incontext_prompt.keys()):
    incontext_prompt[rel] = random.sample(incontext_prompt[rel], args.icl_num)

# declare models
print("Loading model from {} and {}".format(args.triplet_model_path, args.comet_model_path))
triplet_model = T5ForConditionalGeneration.from_pretrained(args.triplet_model_path).to(args.device)
triplet_model.zero_grad()
triplet_tokenizer = AutoTokenizer.from_pretrained(args.triplet_model_path)
comet_model = Comet(args.comet_model_path, device=args.device)
comet_model.model.zero_grad()

def generate_answer(question):
    question_ids = triplet_tokenizer(question, max_length=args.max_len, add_special_tokens=True, padding='max_length', truncation=True, return_tensors='pt').to(args.device)
    # todo: two-step? can be improved here
    answer_ids = triplet_model.generate(**question_ids, max_length=10, num_beams=10)
    # why [:, 1:-1]? T5 tokenizer does not have special tokens, e.g. <s> and </s>, but generated ids has these tokens
    loss = triplet_model(**question_ids, labels=answer_ids[:, 1:-1]).loss.item()
    answer = triplet_tokenizer.decode(answer_ids.squeeze(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return answer, loss

nowtime = datetime.datetime.now().strftime("%m%d-%H%M")
event_level = ['xNeed', 'xAttr', 'xReact', 'xEffect', 'xWant', 'xIntent', 'oEffect', 'oReact', 'oWant', 'isAfter', 'HasSubEvent', 'HinderedBy']
entity_level = ['MadeOf', 'AtLocation', 'isA', 'Partof', 'HasA', 'UsedFor']
for split in ['train', 'val', 'test']:
    path = args.data_path + split + '.json'
    data, label, year, entity_list = dataprocess(path=path)
    imp_data = []
    print('Processing data from {}'.format(path))
    for string in tqdm.tqdm(data):
        # S0. extract commonsense triplets
        news_triplets = []
        t5_loss = []
        for rel in list(extract_prompts_cn.keys()):
            # todo: improve extracting entities -> in-context prompts
            icl_text = ''
            for demon in incontext_prompt[rel]:
                icl_text += extract_prompts_cn[rel] + demon
            # linker
            if rel in entity_level: p1, p2 = '. 实体1是 ', '，实体2是 '
            else: p1, p2 = '. 事件1是 ', '，事件2是 '
            # extract entity1
            extract_entity1_prompt = icl_text + extract_prompts_cn[rel] + string[:256] + p1
            entity1, loss1 = generate_answer(extract_entity1_prompt)
            entity1 = re.split('>', entity1)[-1] # entity1: str
            # extract entity2
            extract_entity2_prompt = extract_entity1_prompt + entity1 + p2
            entity2, loss2 = generate_answer(extract_entity2_prompt)
            entity2 = re.split('>', entity2)[-1]  # entity2: str
            t5_loss.append(loss1 + loss2)
            news_triplets.append([entity1, rel, entity2])  # List (rels * 3)
        # S1. commonsense reasoning
        queries = ["{}{}".format(comet_prompts_cn[tri[1]], tri[0]) for tri in news_triplets]
        results, comet_loss = comet_model.generate(queries, news_triplets=news_triplets, decode_method="beam", num_generate=5)
        # todo: select best answers from [num_generate] answers -> select one with the largest conflict score
        gold_triplet = [[news_triplets[index][0], news_triplets[index][1], results[index][0]] for index in range(len(results))]  # List (rels * 3)
        # S2. calculate conflict scores
        # todo: improve scores
        t5_loss = [(num - min(t5_loss)) / (max(t5_loss) - min(t5_loss)) for num in t5_loss] # List (rels)
        comet_loss = [(num - min(comet_loss)) / (max(comet_loss) - min(comet_loss)) for num in comet_loss] # conflict scores: List (rels)
        for index in range(len(t5_loss)):  # filter
            if t5_loss[index] > args.t5_threshold: comet_loss[index] = 0
        score = max(comet_loss)  # select
        gold_index = comet_loss.index(score)
        # S3. construct final explainations
        if score > args.comet_threshold: expl = '但是， ' + news_triplets[gold_index][0] + link_prompts_cn[news_triplets[gold_index][1]] + gold_triplet[gold_index][2] + ' 而不是 ' + news_triplets[gold_index][2]
        else: expl = '并且， ' + news_triplets[gold_index][0] + link_prompts_cn[news_triplets[gold_index][1]] + gold_triplet[gold_index][2]
        imp_data.append(string + ' ' + expl)
    # S4. write file
    json_path = args.data_path + split + nowtime + '.json'
    json_result = [{"content": imp_data[index], "label": label[index], "time": year[index], 'entity_list': entity_list[index]}
                   for index in range(len(imp_data))]
    with open(json_path, 'w') as file:
        json.dump(json_result, file, indent=4, ensure_ascii=False)
    print("{} data has been saved in {}".format(split, json_path))