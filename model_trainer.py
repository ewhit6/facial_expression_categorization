#Running this file trains the model defined by model.py.
#The trained model weights are saved in this directory at the specified checkpoint path

from model import *

epochs = 100

data = parse_csv_to_nparrays()

train_labels, train_images, test_labels, test_images = data[0], data[1], data[2], data[3]

# Verify the data
# class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
class_names = ['Neutral' ,'Happiness' ,'Surprise' ,'Sadness' ,'Anger' ,'Disgust' ,'Fear' ,'Contempt' ,'Unknown' ,'NF']

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap='gray')
    # The CIFAR labels happen to be arrays,
    # which is why you need the extra index
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()

model = create_model()
model.summary()

checkpoint_path = ".\\training2\\cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    verbose=1)

history = model.fit(train_images, train_labels, epochs=epochs,
                validation_data=(test_images, test_labels), callbacks=[cp_callback])

# Evaluate the model
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print(test_acc)