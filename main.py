import face_recognition
import cv2
import numpy as np
import playsound
from datetime import datetime

alreadyRan = False
video_capture = cv2.VideoCapture(0)
blocking = False
# This will take a jpg and learn the facial features and store it.
<your_name>_image = face_recognition.load_image_file("<your_image>.jpg")
<your_name>_encoding = face_recognition.face_encodings(<your_name>_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    <your_name>_face_encoding,
]
known_face_names = [
    "<your_name>(OWNER)",
]

# just some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
while True:
    # grab a frame of camera/webcam
    ret, frame = video_capture.read()

    # proscesses every other frame coz facereq is hard to proscess:(
    if process_this_frame:
        # make the frame smaller coz once again facereq eats more ram than chrome
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # turn the brg frame to rgb because cv2 see's things like a tv 1937(black and white) 
        rgb_small_frame = small_frame[:, :, ::-1]

        # check if there are any faces and load the location and encoding
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # check if the face on frame matches any jpg listed above
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, 0.65)
            name = "Unknown"
            
            # Or instead, use the known face with the smallest distance to the new face
            # i started to fall asleep so i have no idea what i was doing
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            face_names.append(name)
            
            # if the person still cant be matched to a photo start playing an alarm and write Unknown Person recorded at the time it happend 
            if name == "Unknown":
                playsound.playsound("alarm.wav", block=blocking)
            with open("unknownEntries.txt", mode='a') as file:
                file.write('Unknown Person recorded at %s.\n' % (datetime.now()))

    process_this_frame = not process_this_frame

    # show the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # we get the video back to its original size before showing it
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # create a visible box around the face to indicate a face was found
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # display the name of the recognised person or display unknown
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # finaly create a frame and desplay it on the screen.
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# kill the process
video_capture.release()
cv2.destroyAllWindows()
