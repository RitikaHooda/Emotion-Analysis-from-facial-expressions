import cv2 as cv
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


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
