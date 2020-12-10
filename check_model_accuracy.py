from model import *

checkpoint_path = ".\\training\\cp.ckpt"
model = create_model()
model.load_weights(checkpoint_path)

data = parse_csv_to_nparrays()

test_labels, test_images = data[2], data[3]

loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))