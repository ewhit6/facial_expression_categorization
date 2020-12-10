#This file allows you to test the model's prediction on a given input image.
#Input images should be a close up facial image, 

from model import *
import ntpath

checkpoint_path = ".\\training\\cp.ckpt"
model = create_model()
# model.load_weights(checkpoint_path)

class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

print("---------------------------------------------------------------------------------")
url = input("Enter the URL of the image the facial image you would like to classify:")

face_url = url

face_path = tf.keras.utils.get_file(ntpath.basename(url), origin=face_url)

img = keras.preprocessing.image.load_img(
	face_path, target_size=(48, 48), color_mode="grayscale"
)

# img_array = keras.preprocessing.image.img_to_array(img).reshape(-1,48,48,1)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch
print(img_array.shape)

# predictions = model.predict(img_array)
predictions = model.predict(img_array)
prediction = np.argmax(predictions)

# m=0.000000000000000000001
# a=predictions[0]
# print(len(a))
# for i in range(0,len(a)):
# 	print(a[i])
# 	if a[i]>m:
# 		m=a[i]
# 		ind=i


print(
	"This facial expression is most likely {}"
	.format(class_names[prediction])
)