# Recipe Experiments

 - ```cd src```

 - ```./download_recipe.sh```

 - ```python3 preprocess_recipe.py```

 - ```python3 mk_eval_datasets.py -n 5 --seed 0 --train-shots 5 --eval-tasks 8```

 - ```python3 main.py --reptile1 -n 5 --seed 0 --iseed 0 --cuda 0 --classes 5 --train-shots 5 --batch 8 --lr-clf 1e-4 --eval-tasks 8```