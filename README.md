# MD-PCC
Source code of the paper titled "Robust Misinformation Detection by Visiting Potential Commonsense
Conflict"

### Requirements

```
torch==1.12.1
cudatoolkit==11.3.1
transformers==4.27.4
```

### Prepare Datasets

You can download _Weibo_ and _GossipCop_ from [ENDEF, SIGIR 2023](https://github.com/ICTMCG/ENDEF-SIGIR2022), and place them to the folder `./data`;
Our constructed dataset is located in `./data/ours`.

### Run

1. Generate augmented samples

- for English datasets, you can run
```shell
python generate.py --dataset gossip --icl_num 5
```
- for Chinese datasets, you can run
```shell
python generate_cn.py --dataset weibo --icl_num 5
```

2. Train misinformation detectors
```shell
python main.py --model_name bert --dataset gossip 
```
where `--dataset` includes gossip, weibo, ours, politifact, snopes; `--model_name` contains bert, bertemo, eann, mdfend.

### Citation
```

```