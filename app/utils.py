import numpy as np
import joblib
import cv2


model_randomforest  = joblib.load('./model/random_forest_models1.dat')

print('Model loaded sucessfully')

# settins
fruit_pre = ['Rotten','Fresh']
font = cv2.FONT_HERSHEY_SIMPLEX

def compute_histogram_img(img_list, bins=(16, 16, 16)):
    img_features = []
    for img in img_list:
        # Convert Color to BGR
        bgr_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Calculate histogram color
        hist = cv2.calcHist([bgr_img], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
        # Normalize Histogram
        hist = cv2.normalize(hist, hist).flatten()
        img_features.append(hist)
    return np.array(img_features)

from PIL import Image
from rembg import remove
import os


def remove_background(path):
    os.makedirs('./static/removebg/', exist_ok=True)
    
    img=Image.open(path)
    output=remove(img)
    name_img=path.split('\\')[-1].replace('jpg','png')
    img_clean_path=f'./static/removebg/{name_img}'
    output.save(img_clean_path)
    return img_clean_path
    
def pipeline_model(path,filename,color='bgr'):
    img_clean_path=remove_background(path)
    img = cv2.imread(img_clean_path)
    img_resize = cv2.resize(img,(224,224))
    hsv=cv2.cvtColor(img_resize,cv2.COLOR_BGR2HSV)
    h,s,v=cv2.split(hsv)
    value=20
    lim=255-value
    v[v>lim]=255
    v[v<=lim]+=value
    final_hsv=cv2.merge((h,s,v))
    img_resize=cv2.cvtColor(final_hsv,cv2.COLOR_HSV2BGR)
    
    img_feature=compute_histogram_img([img_resize])
    predict=model_randomforest.predict(img_feature.reshape(1,-1))[0]
    score=model_randomforest.predict_proba(img_feature)[0][predict]
    return fruit_pre[predict],score