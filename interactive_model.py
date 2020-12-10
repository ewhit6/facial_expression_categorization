import cv2
import sys
from PIL import Image, ImageOps
from model import *
import ntpath

checkpoint_path = ".\\training\\cp.ckpt"
model = create_model()
model.load_weights(checkpoint_path)

class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']




# print(
#     "This facial expression is most likely {} with a {:.2f} percent confidence."
#     .format(class_names[np.argmax(score)], 100 * np.max(score))
# )
########################################################3

# cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi_color = cv2.resize(cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY), dsize=(48, 48), interpolation=cv2.INTER_CUBIC)
        img_array = keras.preprocessing.image.img_to_array(roi_color)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

        predictions = model.predict(img_array)
        prediction = np.argmax(predictions)

        cv2.putText(frame, class_names[prediction], (x, y), cv2.FONT_HERSHEY_SIMPLEX ,  
            1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()