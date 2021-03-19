# Form data augmentation

## Available augmentations
1. Shadow
!['gif]('output/add_shadow.gif')

2. 

## Steps:

### Install requirements

* `pip install -r requirements.txt`

### Run demo to see the effect of individual augmentations
* `python demo.py`

`demo.py` uses the sample data in `data/` and generates `GIF` outputs in output.

### Run the augmentation pipeline
* `python main.py python main.py --data-root data/ --output-dir output/ --aug-prob 0.1`

    *  `--data-root`: path to data directory
    * `--output-dir`: path to outputs directory
    * `--aug_prob`: probability with which each augmentation is applied, when the value is equal to `1`, all augmentations are applied, and when the value is equal to `0.1` an augmentation is applied with probability equal to `0.1`
