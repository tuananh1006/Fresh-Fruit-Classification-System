import numpy as np
import joblib
import cv2


model_randomforest  = joblib.load('./model/random_forest_models.dat')

print('Model loaded sucessfully')

# settins
fruit_pre = ['Rotten','Fresh']
font = cv2.FONT_HERSHEY_SIMPLEX

def compute_histogram_img(img_list, bins=(8, 8, 8)):
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

def pipeline_model(path,filename,color='bgr'):
    img = cv2.imread(path)
    img_resize = cv2.resize(img,(224,224))
    img_feature=compute_histogram_img([img_resize])
    predict=model_randomforest.predict(img_feature.reshape(1,-1))[0]
    score=model_randomforest.predict_proba(img_feature)[0][predict]
    return fruit_pre[predict],score
    