import cv2
import os
import pickle

# emotions: 1. angry  2. disgusted    3. fearful  4. happy    5. neutral  6. sad  7. suprised
# images folder path
dir_path = os.getcwd() + '\\images'
lib = []
image = {}
for path in os.listdir(dir_path):
    dir = os.path.join(dir_path, path)
    image[path] = 0
    setImg = []
    lib.append(path)
    for img in os.listdir(dir):
        getImage = cv2.imread(os.path.join(dir, img), 1)
        if getImage is not None:
            setImg.append(cv2.resize(getImage, (672, 420)))
    image[path] = setImg

with open('/models/images.pkl', 'wb') as fp:
    pickle.dump(image, fp)
    print('iamges saved successfully to file')

