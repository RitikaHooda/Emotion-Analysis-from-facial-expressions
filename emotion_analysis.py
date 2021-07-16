import cv2 as cv
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
import csv
import os


class FaceDetection(object):
    def __init__(self):
        self.face_cascade = cv.CascadeClassifier('models/haarcascade_frontalface_default.xml')
        self.total_faces = []

    def predict(self, img):
        # Converting frame into grayscale for better efficiency
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Using Haar Cascade for detecting faces in an image
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=3)
        # append values to the array
        self.total_faces.append(len(faces))
        face = None
        # Creating the rectangle around face
        for (x, y, w, h) in faces:
            img = cv.rectangle(img, (x, y), (x + w, y + h), (120, 250, 0), 2)
            face = gray[y + 5:y + h - 5, x + 20:x + w - 20]

        try:
            face = cv.resize(face, (48, 48))
        except:
            return (None, img)

        return (face, img)

    def totalDetections(self):
        total = 0
        for i in self.total_faces:
            total += i
        self.resetTotalFaces()
        return total

    def resetTotalFaces(self):
        self.total_faces = []

class EmotionPrediction(object):
    def __init__(self):
        self.model = tf.keras.models.load_model('models/expression.model')
        self.labels = ["Neutral", "Happy", "Sad", "Surprise", "Angry"]
        self.all_expressions = []
        self.dict_emotions = {}
        self.initEmotionDictionary()

    def initEmotionDictionary(self):
        for i in range(len(self.labels)):
            self.dict_emotions[self.labels[i]] = 0

    def predict(self, face):
        expressions = []
        face = face / 255.0
        predictions = self.model.predict(np.array([face.reshape((48, 48, 1))])).argmax()
        state = self.labels[predictions]
        expressions.append(state)
        print(state)
        self.all_expressions.append(expressions)

    def getEmotions(self):
        for a in self.all_expressions:
            for e in a:
                self.dict_emotions[e] +=1
        dict = self.dict_emotions
        self.resetEmotions()
        return dict

    def resetEmotions(self):
        self.all_expressions = []
        self.dict_emotions = {}
        self.initEmotionDictionary()

    def getEmotionsLabels(self):
        return self.labels
    
    
def saveCsvFile(video_file, output_file='output/out.csv', total_faces=0, emotions_dict={}, emotions=[]):
    data = []
    if not os.path.isfile(output_file):
        data = ['Video path', 'Total of faces', 'Neutral Emotions', 'Happy Emotions', 'Sad Emotions',
                'Surprise Emotions', 'Angry Emotions']
    else:
        data.append(video_file)
        data.append(total_faces)
        for i in emotions:
            data.append(emotions_dict[i])

    with open(output_file, mode='a') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(data)

def main():
    face_detection = FaceDetection()
    emotion_prediction = EmotionPrediction()
    # Capturing Video from primary webcam, you can change number from 0 to any Integer
    ## for other webcams if you have many of them
    cap = cv.VideoCapture(0)
    emotions = emotion_prediction.getEmotionsLabels()
    start_time = datetime.datetime.now()
    output_file = "output/out.csv"
    os.remove(output_file)
    video_file = "camera"  ## TODO choose according to the no of faces detected
    while cap.isOpened():
        # Reading frame from the precorded video.
        ret, frame = cap.read()
        if np.shape(frame) == ():
            break

        face, img = face_detection.predict(frame)

        if np.shape(img) == ():
            print("img is empty")
        else:
            cv.imshow('img', img)
            cv.waitKey(10)

        if face is not None:
            emotion_prediction.predict(face)

        check_time = datetime.datetime.now()

        if (check_time - start_time).total_seconds() > 3:
            emotions_dict = emotion_prediction.getEmotions()
            total_faces = face_detection.totalDetections()
            saveCsvFile(video_file, output_file=output_file, total_faces=total_faces, emotions_dict=emotions_dict,
                        emotions=emotions)
            start_time = datetime.datetime.now()

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
