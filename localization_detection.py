import numpy as np
import cv2
from model import model_and_weight


# XML classifier for face detection
face_cascade = cv2.CascadeClassifier(
    r'C:\Users\91956\miniconda3\envs\env-two\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')

# Font for putting text onto image
font = cv2.FONT_HERSHEY_SIMPLEX

# Video Source to default webcam(put 0), can also be set to local disk file
video_capture = cv2.VideoCapture('facial_expressions.mp4')

# Loads the trained model
trained_model = model_and_weight()

# Expressions
expressions = ["Angry", "Disgust", "Fear",
               "Happy", "Neutral", "Sad", "Surprise"]

# Processing the video frame by frame
while True:

    ret, frame = video_capture.read()  # reads one frame and returns it

    # grayscale conversion of image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Face detection in the current frame
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    # Bounding boxes around the faces
    for (x, y, w, h) in faces:

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]  # Only the face portion but in grayscale
        # input to the CONV net 48x48 grayscale
        network_input = cv2.resize(roi_gray, (48, 48))

        # feeding the input to the network
        predictions = trained_model.predict(
            network_input[np.newaxis, :, :, np.newaxis])

        expression = expressions[np.argmax(predictions)]  # Network Output

        cv2.putText(frame, expression, (x, y), font, 1,
                    (0, 0, 0), 2)  # indicates the expression

    # Display result
    cv2.imshow('Video', frame)

    # quits when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release capture
video_capture.release()
cv2.destroyAllWindows()
