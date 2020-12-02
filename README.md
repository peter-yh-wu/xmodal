# Cross-Modal Generalization: Learning in Low Resource Modalities via Meta-Alignment

Code for the respective [paper](http://www.cs.cmu.edu/~peterw1/website_files/xmodal.pdf).

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
