# Environment
python 3.9
torch==1.12.1+cu116
transformers==4.24.0

# Training
```shell
python train.py
```

# Inference
```shell
bash download.sh
bash run.sh <test_file> <path to predicion>
```

# Evaluate
```shell
git clone https://github.com/moooooser999/ADL22-HW3.git
cd ADL22-HW3 && pip install -e tw_rouge
python eval.py -r <reference_file> -s <prediction_file>
```