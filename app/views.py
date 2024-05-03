from flask import render_template, request
from flask import redirect, url_for
import os
from PIL import Image
from app.utils import pipeline_model

UPLOAD_FLODER = 'static/uploads'
def base():
    return render_template('base.html')


def index():
    return render_template('index.html')


def fruit():
    return render_template('fruit.html')

def getwidth(path):
    img = Image.open(path)
    size = img.size # width and height
    aspect = size[0]/size[1] # width / height
    w = 300 * aspect
    return int(w)

def fresh():

    if request.method == "POST":
        f = request.files['image']
        filename=  f.filename
        path = os.path.join(UPLOAD_FLODER,filename)
        f.save(path)
        w = getwidth(path)
        # prediction (pass to pipeline model)
        predict,score=pipeline_model(path,filename,color='bgr')


        return render_template('fresh.html',fileupload=True,img_name=filename, w=w,predict=predict,score=score)


    return render_template('fresh.html',fileupload=False,img_name="freeai.png")