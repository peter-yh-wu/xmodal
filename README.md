# Learning in Low-resource Modalities via Cross-modal Generalization

 - ```cd src```

 - ```./download_recipe.sh```

 - ```python3 preprocess_recipe.py```

 - ```python3 mk_eval_datasets.py -n 3 --seed 0 --train-shots 5 --eval-tasks 8```

 - ```python3 main.py --seed 0 --iseed 0 --cuda 0 --classes 3 --train-shots 5 --batch 8 --meta-lr 1e-4 --lr-clf 1e-4 --eval-tasks 8 --iterations 9 --test-iterations 9 -l tri --margin 0.5```

 - ```classes``` and ```iseed``` in ```main.py``` should match ```n``` and ```seed``` in ```mk_eval_datasets.py```, respectively.

 - To run ablation studies, add the respective tag to ```main.py```, e.g. ```--reptile1``` for Oracle.
