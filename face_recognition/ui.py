import sys
from PyQt5.QtCore import Qt, QSize, QTimer, QThread
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QLabel, QPushButton
from PyQt5.QtGui import QPixmap, QImage
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import uuid
import sqlite3


sqliteConnection = sqlite3.connect('features.db')

show_features = False

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

persons = ["hilel", "lavi", "lidor"]

def get_tensor(landmarks):
    arr = np.array([[],[],[]])
    for l in landmarks:
        arr = np.append(arr, [[l.x] , [l.y], [l.z]], axis=1)        
    return arr


face_mesh =  mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

def add_face_mesh(image):
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.    
    landmarks = None
    image.flags.writeable = False
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_tesselation_style())
#            mp_drawing.draw_landmarks(
#                image=image,
#                landmark_list=face_landmarks,
#                connections=mp_face_mesh.FACEMESH_CONTOURS,
#                landmark_drawing_spec=None,
#                connection_drawing_spec=mp_drawing_styles
#                .get_default_face_mesh_contours_style())
#            mp_drawing.draw_landmarks(
#                image=image,
#                landmark_list=face_landmarks,
#                connections=mp_face_mesh.FACEMESH_IRISES,
#                landmark_drawing_spec=None, 
#                connection_drawing_spec=mp_drawing_styles
#                .get_default_face_mesh_iris_connections_style())
            landmarks = face_landmarks.landmark
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image, landmarks
 
def load_facial_features(file):    
    image = cv2.imread(file)
    im, landmarks = add_face_mesh(image)
    t = get_tensor(landmarks)
    print("Loaded: " + file)
    return t

def main():    
    app = QApplication([])
    
    window = QWidget()
    window.setLayout(QGridLayout(window))    
    window.setMinimumSize(QSize(640, 480))

    label = QLabel()
    label.setFixedSize(640, 640)    
    window.layout().addWidget(label, 0, 0)

    takePictureBtn = QPushButton("Take Picture")        
    # setting geometry of button
    takePictureBtn.setGeometry(200, 150, 100, 40)

    # creating a push button
    button = QPushButton("Get Facial Features")
    # setting geometry of button
    button.setGeometry(200, 150, 100, 40)
    # setting checkable to true
    button.setCheckable(True)
    # setting default color of button to light-grey
    button.setStyleSheet("background-color : lightgrey")

    def changeColor():  
        # if button is checked
        if button.isChecked():
            button.setStyleSheet("background-color : lightblue")
            show_features = True
        # if it is unchecked
        else:  
            # set background color back to light-grey
            button.setStyleSheet("background-color : lightgrey")
            show_features = False

    # setting calling method by button
    button.clicked.connect(changeColor)
    window.layout().addWidget(button, 640, 0)
    window.layout().addWidget(takePictureBtn, 700, 0)

    window.show()

    vc = cv2.VideoCapture(0)
    vc.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def take_picture():
        rval, frame = vc.read()            
        unique_filename = str(uuid.uuid4())
        cv2.imwrite(unique_filename + ".png", frame)

    takePictureBtn.clicked.connect(take_picture)
    
    database = {}
    for p in persons:
        database[p] = load_facial_features(p + ".png")

    timer = QTimer()
    timer.timeout.connect(lambda: nextFrameSlot(vc, label, button, database))
    timer.start(1000. / 12)

    return app.exit(app.exec_())


def nextFrameSlot(vc: cv2.VideoCapture, label: QLabel, button:QPushButton, database):        
    rval, frame = vc.read()    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    x = 200
    y =75
    w = 640 - x*2
    h= 480 - y*2
    
    
    if button and button.isChecked():
        foundPerson = ""
        frame, re = add_face_mesh(frame)    
        if re is not None:
            t = get_tensor(re)
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255,0,0), 2)
            for p in database:
                diff = database[p] - t
                distance = np.sqrt(np.mean(diff**2))                
                if distance <= 0.02:
                    print("Hello " + p)
                    frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (36,255,12), 2)
                    cv2.putText(frame, p, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                    button.click()
                    foundPerson = p
                else:
                    cv2.putText(frame, '%.3f'%distance, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
                    
    else:
        #cv2.putText(frame, foundPerson, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (36,255,12), 2)
    
    

    image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
    pixmap = QPixmap.fromImage(image)
    label.setPixmap(pixmap)

main()