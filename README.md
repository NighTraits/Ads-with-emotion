# Networking Proyect


## Introduction

Detector de expresiones faciales y carteles de publicidad inteligente.


## Libraries

`pip install -r requirements.txt`


## Train model

Train your own models
1. Download dataset [fer2013](https://www.kaggle.com/datasets/deadskull7/fer2013)
2. run `python dataset_prepare.py` to prepare data from dataset.
3. run `python emotions-train.py` to train model (you can change the _batch_size_ and _epoch_ value.

or use the one at /models/model.h5 in which the plot result:

![plot](https://github.com/NighTraits/Ads-with-emotion/blob/master/models/plot.png?raw=true)


## Prepare images to show

In the folder [/images](https://github.com/NighTraits/Ads-with-emotion/tree/master/images) there are 10 images for each category.
* books
* clothing
* cosmetics
* food and beverages
* movies
* technology
* video games

If you want to add more images or categories, run `python generate-image.py` in the console. This will save a _images.pkl_ in the folder [/models](https://github.com/NighTraits/Ads-with-emotion/tree/master/models). Otherwise, use the file [_images.pkl_](https://github.com/NighTraits/Ads-with-emotion/blob/master/models/images.pkl) existent.


## Run programs

```bash
python emotions-display.py
python loadImage.py
```
