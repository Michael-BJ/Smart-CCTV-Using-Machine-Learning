import face_recognition
import cv2
import numpy as np
import time
import pyttsx3
import telepot

bot = telepot.Bot('insert your token')
# For sound
engine = pyttsx3.init() 

frame = cv2.imread('',cv2.IMREAD_GRAYSCALE)
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

etcodetech_image = face_recognition.load_image_file("etcodetech.jpg")
etcodetech_face_encoding = face_recognition.face_encodings(etcodetech_image)[0]

known_face_encoding = [
    etcodetech_face_encoding
]
known_face_names = [
    "etcodetech"
]

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3,5)
 
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb)
    names = []
    
    for encoding in encodings:
        matches = face_recognition.compare_faces(known_face_encoding,
        encoding)
        name = "Unknown"

        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            for i in matchedIdxs:
                best_match_index = np.argmin(matches)
                name = known_face_names[best_match_index]
                counts[name] = counts.get(name, 0) + 1
            name = max(counts, key=counts.get)
        names.append(name)
        
        for ((x, y, w, h), name) in zip(faces, names):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
             0.75, (0, 255, 0), 2)
            if matches [0] == True:
                engine.say("Wellcome home etcodetech") 
                cv2.imwrite("family.jpg", frame)
                bot.sendPhoto('insert your chat id', photo=open('family.jpg', 'rb'))
            else:
                engine.say("Who are you ? i don't know you")
                engine.say("Please go back later")
                cv2.imwrite("guest.jpg", frame)
                bot.sendPhoto('insert your chat id', photo=open('guest.jpg', 'rb'))
            engine.runAndWait()       
    cv2.imshow("Is that you", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
