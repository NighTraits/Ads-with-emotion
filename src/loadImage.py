import cv2
from random import randrange
import pickle
import pandas as pd
import requests

# data log
logCat = {}

# ubidots, add token=*token*
url = "http://things.ubidots.com/api/v1.6/devices/redes/?token=BBFF-SuEt6pAXu6UBMp6QJw4dcByH7yaaNV"

# load images
with open('models/images.pkl', 'rb') as fp:
    image = pickle.load(fp)

# categories and emotions
emotion = ['Disgusted', 'Angry', 'Fearful', 'Sad', 'Neutral', 'Happy', 'Surprised']
cat = ['books', 'clothing', 'cosmetics', 'food_and_beverage', 'movies', 'technology', 'video_games']
good = ['Happy', 'Surprised', 'Neutral']
bad = ['Angry', 'Disgusted', 'Fearful', 'Sad']

# variables
i = randrange(10)
j = randrange(len(cat))
loop = True
emotionPred='Neutral'
while loop:
    cv2.imshow('dst', image[cat[j]][i])
    
    key = cv2.waitKey(2000)
    try:
        with open('models/emotion.pkl', 'rb') as fp:
            emotionPred = pickle.load(fp)
    except:
        emotionPred = emotionPred
    
    print(emotionPred)

    response = requests.post(url, json={cat[j]:{'value': emotion.index(emotionPred), "context":{"emotion":emotionPred}}})
    if response.status_code == 200:
        print("Petición exitosa")
        print(response.json())
    else:
        print("Ocurrio un error")
        print(response.json())

    if(emotionPred in bad):
        cat.remove(cat[j])
        if(len(cat) < 1):
            cat = ['books', 'clothing', 'cosmetics', 'food_and_beverage', 'movies', 'technology', 'video_games']
        j = randrange(len(cat))
        i = randrange(10)
    else:
        if(i+1 > 9):
            i=0
        else:
            i+=1

    if cv2.waitKey(1)  & 0xFF == ord('q'):
        loop = False
              
cv2.destroyAllWindows()
