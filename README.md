# Networking Proyect


## Introduction

Detector de expresiones faciales y carteles de publicidad inteligente.


## Libraries

```bash
pip install -r requirements.txt
```


## Train model

Train your own models
1. Download dataset https://www.kaggle.com/datasets/deadskull7/fer2013
2. run 
```bash 
python dataset_prepare.py
python emotions-train.py
```

## Prepare images to show

In /images there are 10 images for each category.
```bash
python generate-image.py
```

## Run programs

```bash
python emotions-display.py
python loadImage.py
```
