# Form data augmentation

## Available augmentations
1. Shadow
![](https://github.com/gautam-aayush/form-data-augmentation/blob/main/output/shadow.gif)

2. Wrinkles
![](https://github.com/gautam-aayush/form-data-augmentation/blob/main/output/wrinkles.gif)

3. Saturation
![](https://github.com/gautam-aayush/form-data-augmentation/blob/main/output/gamma_saturation.gif)

4. Watermark
![](https://github.com/gautam-aayush/form-data-augmentation/blob/main/output/watermark.gif?raw=true)

5. Binarize
![](https://user-images.githubusercontent.com/70262751/111758313-71413b00-88c4-11eb-846e-4380ee32d606.png)

6. Perspective Distortion
![](https://github.com/gautam-aayush/form-data-augmentation/blob/main/output/perspective.gif?raw=true)

7. Stretch Distortion
![](https://github.com/gautam-aayush/form-data-augmentation/blob/main/output/stretch.gif?raw=true)

8. LCD Texture
![](https://github.com/gautam-aayush/form-data-augmentation/blob/main/output/lcd_overlay.gif?raw=true)

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
