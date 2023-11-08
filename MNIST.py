from tensorflow import keras 
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

# load dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# network architecture
model = keras.Sequential([layers.Dense(512, activation="relu"), layers.Dense(10, activation="softmax")])

# compilation step
model.compile(optimizer="rmsprop",loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# pre-processing the image data
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255

# "fitting" the model 
model.fit(train_images, train_labels, epochs=5, batch_size=128)

# make predicitons 
test_digits = test_images[10:20]
predictions = model.predict(test_digits)
for i in range(len(predictions)):
    print(predictions[i].argmax(), test_labels[i])

# evaluation using accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"test accuracy: {test_acc}")
