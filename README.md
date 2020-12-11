# Cross-Modal Generalization: Learning in Low Resource Modalities via Meta-Alignment

> Pytorch implementation for cross-modal generalization across language, visual, and audio modalities.

Correspondence to: 
  - Paul Liang (pliang@cs.cmu.edu)
  - Peter Wu (peterw1@cs.cmu.edu)

## Paper

[**Cross-Modal Generalization: Learning in Low Resource Modalities via Meta-Alignment**](https://arxiv.org/abs/2012.02813)<br>
[Paul Pu Liang](http://www.cs.cmu.edu/~pliang/), [Peter Wu](http://www.cs.cmu.edu/~peterw1/), [Liu Ziyin](http://cat.phys.s.u-tokyo.ac.jp/~zliu/), [Louis-Philippe Morency](https://www.cs.cmu.edu/~morency/), and [Ruslan Salakhutdinov](https://www.cs.cmu.edu/~rsalakhu/)<br>
NeurIPS 2020 Workshop on Meta Learning

If you find this repository useful, please cite our paper:
```
@misc{liang2020crossmodal,
      title={Cross-Modal Generalization: Learning in Low Resource Modalities via Meta-Alignment}, 
      author={Paul Pu Liang and Peter Wu and Liu Ziyin and Louis-Philippe Morency and Ruslan Salakhutdinov},
      year={2020},
      eprint={2012.02813},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Installation

First check that the requirements are satisfied:</br>
Python 3.6</br>
torch 1.2.0</br>
numpy 1.18.1</br>
sklearn 0.20.0</br>
matplotlib 3.1.2</br>
gensim 3.8.0 </br>

The next step is to clone the repository:
```bash
git clone https://github.com/peter-yh-wu/xmodal.git
```

## Background

The natural world is abundant with concepts expressed via visual, acoustic, tactile, and linguistic modalities. Much of the existing progress in multimodal learning, however, focuses primarily on problems where the same set of modalities are present at train and test time, which makes learning in low-resource modalities particularly difficult. In this work, we propose algorithms for cross-modal generalization: a learning paradigm to train a model that can (1) quickly perform new tasks in a target modality (i.e. meta-learning) and (2) doing so while being trained on a different source modality. We study a key research question: how can we ensure generalization across modalities despite using separate encoders for different source and target modalities? Our solution is based on meta-alignment, a novel method to align representation spaces using strongly and weakly paired cross-modal data while ensuring quick generalization to new tasks across different modalities. Our results demonstrate strong performance even when the new target modality has only a few (1-10) labeled samples and in the presence of noisy labels, a scenario particularly prevalent in low-resource modalities.

## Reproducing Results

 - ```cd src```

 - ```./download_recipe.sh```

 - ```python3 preprocess_recipe.py```

 - ```python3 mk_eval_datasets.py -n 3 --seed 0 --train-shots 5 --eval-tasks 8```

 - ```python3 main.py --seed 0 --iseed 0 --cuda 0 --classes 3 --train-shots 5 --batch 8 --meta-lr 1e-4 --lr-clf 1e-4 --eval-tasks 8 --iterations 9 --test-iterations 9 -l tri --margin 0.5```

 - ```classes``` and ```iseed``` in ```main.py``` should match ```n``` and ```seed``` in ```mk_eval_datasets.py```, respectively.

 - To run ablation studies, add the respective tag to ```main.py```, e.g. ```--reptile1``` for Oracle.
