import cv2
import face_recognition
import os
import numpy as np

# load multiple faces.
known_faces = []
known_names = []
for face in os.listdir('faces'):
    known_faces.append(face_recognition.face_encodings(face_recognition.load_image_file('faces/' + face))[0])
    known_names.append(face.split('.')[0])

known_faces = np.array(known_faces)

# capture video stream
cap = cv2.VideoCapture(0)
# set the width and height
cap.set(3, 640)
cap.set(4, 480)

# loop to capture frames from live camera
while True:
    ret, frame = cap.read()
    # convert the frame from BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # get the faces from the frame using mediapipe face detection model
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # loop through the faces and draw the bounding box
    for i, face_encoding in enumerate(face_encodings):
        name = "Unknown"
        if len(known_faces) > 0:
            face_distances = face_recognition.face_distance(known_faces, face_encoding)
            best_match_index = np.argmin(face_distances)              
            name = known_names[best_match_index]

        # Calculate the width and height of the name box
        text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        text_w, text_h = text_size
        
        top, right, bottom, left = face_locations[i]
        # Calculate mid_x for center alignment of the name box
        mid_x = left + (right - left) // 2

        # Calculate top left corner of the name box 1/2 face box height above the face box
        
        box_x1 = mid_x - text_w // 2
        box_y1 = top - ((bottom-top) // 2) - text_h - 10
        box_x2 = mid_x + text_w // 2
        box_y2 = top - ((bottom-top) // 2) - 5

        # Draw the name box
        cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (255, 255, 255), cv2.FILLED)

        # Draw the name text inside the box
        cv2.putText(frame, name, (box_x1, box_y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
    cv2.imshow("frame", frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
cap.release()
cv2.destroyAllWindows() 