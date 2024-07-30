import cv2
import face_recognition
import os
import numpy as np

# Initialize lists to hold the face encodings and their corresponding names
known_faces = []
known_names = []

# Loop through the files in the 'faces' directory
for face in os.listdir('faces'):
    if face == 'README.md':
        continue
    # Load each image file and get the face encoding
    known_faces.append(face_recognition.face_encodings(face_recognition.load_image_file('faces/' + face))[0])
    # Extract the name from the file name (assuming the file name is the person's name)
    known_names.append(face.split('.')[0])

# Convert the list of known faces to a numpy array for faster processing
known_faces = np.array(known_faces)

# Capture video from the default camera (usually the first camera device)
cap = cv2.VideoCapture(0)
# Set the width of the video frame
cap.set(3, 640)
# Set the height of the video frame
cap.set(4, 480)

# Start an infinite loop to continuously capture frames from the camera
while True:
    # Capture a single frame from the camera
    ret, frame = cap.read()
    # Convert the frame from BGR (Blue, Green, Red) to RGB (Red, Green, Blue) color space
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect the face locations in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    # Get the face encodings for the detected faces
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through each face detected in the frame
    for i, face_encoding in enumerate(face_encodings):
        # Default name for the face is "Unknown"
        name = "Unknown"
        if len(known_faces) > 0:
            # Compute the distance between the detected face and all known faces
            face_distances = face_recognition.face_distance(known_faces, face_encoding)
            # Find the index of the known face with the smallest distance (best match)
            best_match_index = np.argmin(face_distances)              
            # Get the name of the best match
            name = known_names[best_match_index]

        # Calculate the size of the text box that will display the name
        text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        text_w, text_h = text_size
        
        # Get the coordinates of the face bounding box
        top, right, bottom, left = face_locations[i]
        # Calculate the horizontal center of the face box
        mid_x = left + (right - left) // 2

        # Calculate the top left corner of the name box (slightly above the face box)
        box_x1 = mid_x - text_w // 2
        box_y1 = top - ((bottom-top) // 2) - text_h - 10
        # Calculate the bottom right corner of the name box
        box_x2 = mid_x + text_w // 2
        box_y2 = top - ((bottom-top) // 2) - 5

        # Draw the name box (filled rectangle) on the frame
        cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (255, 255, 255), cv2.FILLED)

        # Draw the name text inside the box
        cv2.putText(frame, name, (box_x1, box_y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
    # Display the frame with the bounding boxes and names
    cv2.imshow("frame", frame)
    # Wait for a key press; if 'q' is pressed, exit the loop
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
