"""
creat file: lable.pickle and trainner/trainner.yml
"""
import os
import cv2
import numpy as np
from PIL import Image
import pickle
y_labels = []
x_train = []
#get dir of folder program
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR,"images")

current_id = 0
label_ids ={}

face_cascade = cv2.CascadeClassifier('classifier/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

for root,dirs,files in os.walk(image_dir):
	for file in files:
		if file.endswith("png") or file.endswith("jpg"):
			path = os.path.join(root,file)
			label = os.path.basename(root).replace(" ","-").lower()
			if label in label_ids:
				pass
			else:
				label_ids[label] = current_id
				current_id += 1
			id_ = label_ids[label]
			#print(label_ids)
			#y_labels.append(label)
			#x_train.append(path)
			#convert image to GRAYSCASE use pillow
			pil_image = Image.open(path).convert("L")
			#convert gray to numpy
			image_array = np.array(pil_image,"uint8")
			#print(image_array)
			#dectect face in image_array
			faces = face_cascade.detectMultiScale(image_array, 1.3, 5)
			#if face is detect, append to x_train array, y_labels array
			for(x,y,w,h) in faces:
				roi = image_array[y:y+h,x:x+w]
				x_train.append(roi)
				y_labels.append(id_)

#print(y_labels)
#print(x_train)
print(label_ids)
with open("labels.pickle","wb") as f:
	pickle.dump(label_ids,f)
recognizer.train(x_train,np.array(y_labels))
recognizer.save("trainner/trainner_face.yml")