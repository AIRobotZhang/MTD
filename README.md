## [Exploring Modular Task Decomposition in Cross-domain Named Entity Recognition (SIGIR 2022)](https://github.com/AIRobotZhang/MTD)

## Framework
![image](img/framework.png)

## Requirements
- python==3.7.4
- pytorch==1.8.1
- [huggingface transformers](https://github.com/huggingface/transformers)
- numpy
- tqdm

## Overview
```
├── root
│   └── dataset
│       ├── conll2003_train.json
│       ├── conll2003_tag_to_id.json
│       ├── politics_train.json
│       ├── politics_dev.json
│       ├── politics_test.json
│       ├── politics_tag_to_id.json
│       └── ...
│   └── models
│       ├── __init__.py
│       ├── modeling_span.py
│       └── modeling_type.py
│   └── utils
│       ├── __init__.py
│       ├── config.py
│       ├── data_utils.py
│       ├── eval.py
│       └── ...
│   └── ptms
│       └── ... (trained results, e.g., saved models, log file)
│   └── cached_models
│       └── ... (BERT pretrained model, which will be downloaded automatically)
│   └── run_script.py
│   └── run_script.sh
```

## How to run
```console
sh run_script.sh <GPU ID> <DATASET NAME> <span tau> <type tau> <mu>
```
e.g., in the music domain 
```console
sh run_script.sh 0 music 0.1 0.1 1.0
```

## Citation
```
@inproceedings{zhang:2022,
  title={Exploring Modular Task Decomposition in Cross-domain Named Entity Recognition},
  author={Xinghua Zhang and Bowen Yu and Tingwen Liu and Yubin Wang and Taoyu Su and Hongbo Xu},
  booktitle={Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR’22)},
  year={2022}
}