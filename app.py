from camera import Video
from flask import Flask, render_template, request,Response
import cv2
import numpy as np 
from keras.models import load_model

app=Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    while True:
        frame=camera.get_frame()
        yield(b'--frame\r\n'
       b'Content-Type:  image/jpeg\r\n\r\n' + frame +
         b'\r\n\r\n')

@app.route('/video')

def video():
    return Response(gen(Video()),
    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/image')
def image():
    return render_template('image.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
	image = request.files['select_file']

	image.save('static/file.jpg')

	image = cv2.imread('static/file.jpg')

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	
	faces = cascade.detectMultiScale(gray, 1.1, 3)
	

	for x,y,w,h in faces:
		cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)

		cropped = image[y:y+h, x:x+w]


	cv2.imwrite('static/after.jpg', image)
	try:
		cv2.imwrite('static/cropped.jpg', cropped)

	except:
		pass


	try:
		img = cv2.imread('static/cropped.jpg', 0)

	except:
		img = cv2.imread('static/file.jpg', 0)

	img = cv2.resize(img, (48,48))
	img = img/255

	img = img.reshape(1,48,48,1)

	model = load_model('model.h5')

	pred = model.predict(img)


	label_map = ['Anger','Neutral' , 'Fear', 'Happy', 'Sad', 'Surprise']
	pred = np.argmax(pred)
	final_pred = label_map[pred]


	return render_template('predict.html', data=final_pred)


app.run(debug=True)