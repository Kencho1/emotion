import numpy as np
from keras.models import load_model
import cv2
from keras.preprocessing import image
from keras.utils import img_to_array
from time import sleep



class Video(object):
    def __init__(self):
        self.video=cv2.VideoCapture(0)
    def __del__(self):
        self.video.release()
    def get_frame(self):
        faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        classifier =load_model(r'model.h5')
        emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']
        
        while True:
            ret,frame=self.video.read()
            labels = []
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        
            faces = faceDetect.detectMultiScale(frame, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
                roi_gray = gray[y:y+h,x:x+w]
                roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

                if np.sum([roi_gray])!=0:
                    roi = roi_gray.astype('float')/255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi,axis=0)

                    prediction = classifier.predict(roi)[0]
                    label=emotion_labels[prediction.argmax()]
                    label_position = (x,y-10)
                    cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        
            ret,jpg=cv2.imencode('.jpg',frame)
            return jpg.tobytes()
            

# faces=faceDetect.detectMultiScale(frame, 1.3, 5)
            # for x,y,w,h in faces:
            #     x1,y1=x+w, y+h
            #     cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,255), 1)
            #     cv2.line(frame, (x,y), (x+30, y),(255,0,255), 6) #Top Left
            #     cv2.line(frame, (x,y), (x, y+30),(255,0,255), 6)

            #     cv2.line(frame, (x1,y), (x1-30, y),(255,0,255), 6) #Top Right
            #     cv2.line(frame, (x1,y), (x1, y+30),(255,0,255), 6)

            #     cv2.line(frame, (x,y1), (x+30, y1),(255,0,255), 6) #Bottom Left
            #     cv2.line(frame, (x,y1), (x, y1-30),(255,0,255), 6)

            #     cv2.line(frame, (x1,y1), (x1-30, y1),(255,0,255), 6) #Bottom right
            #     cv2.line(frame, (x1,y1), (x1, y1-30),(255,0,255), 6)