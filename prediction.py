from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2 
import numpy as np
def prediction(saved_image):
    index = ["no","yes"]
    model = load_model("BTD2.h5")
    img = image.load_img("./static/TI/"+saved_image,target_size = (64,64))
    x = image.img_to_array(img)
    x = np.expand_dims(x,axis=0)
    predict_x=model.predict(x) 
    classes_x=np.argmax(predict_x,axis=1) 
    # print(index[pred[0]])
    print(classes_x)
    val = classes_x[0]
    pred = ""
    if val == 1:
        pred='yes'
    else:
        pred = 'no'
    # return index[pred[0]]
    return(pred)