import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate, Reshape

# Function to create an image with a circle
def create_circle_image(radius, center, size=(64, 64)):
    image = np.zeros(size, dtype=np.uint8)
    cv2.circle(image, center, radius, (255, 255, 255), -1)
    return image

# Function to generate a dataset of images with circles
def generate_dataset(num_samples=1000, image_size=(64, 64), radius=5):
    dataset = []
    labels = []
    for _ in range(num_samples):
        x1, y1 = np.random.randint(radius, image_size[0] - radius, size=2)
        x2, y2 = np.random.randint(radius, image_size[0] - radius, size=2)
        img1 = create_circle_image(radius, (x1, y1), image_size)
        img2 = create_circle_image(radius, (x2, y2), image_size)
        dataset.append((img1, img2))
        labels.append((x2, y2))
    return np.array(dataset), np.array(labels)

# Generate the dataset
dataset, labels = generate_dataset()

# Normalize the images
dataset = dataset / 255.0

# Add channel dimension
dataset = dataset[..., np.newaxis]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, random_state=42)

# Define the model architecture
input_shape = X_train.shape[2:]

input_layer = Input(shape=(2, *input_shape))

# Reshape input for each image
reshaped_input = Reshape((2, *input_shape))(input_layer)

# Process first image
x1 = Conv2D(32, (3, 3), activation='relu')(reshaped_input[:, 0])
x1 = MaxPooling2D((2, 2))(x1)
x1 = Conv2D(64, (3, 3), activation='relu')(x1)
x1 = MaxPooling2D((2, 2))(x1)
x1 = Flatten()(x1)

# Process second image
x2 = Conv2D(32, (3, 3), activation='relu')(reshaped_input[:, 1])
x2 = MaxPooling2D((2, 2))(x2)
x2 = Conv2D(64, (3, 3), activation='relu')(x2)
x2 = MaxPooling2D((2, 2))(x2)
x2 = Flatten()(x2)

# Combine processed outputs
combined = Concatenate()([x1, x2])
output_layer = Dense(128, activation='relu')(combined)
output_layer = Dense(64, activation='relu')(output_layer)
output_layer = Dense(2)(output_layer)  # Predict x, y coordinates

model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Summarize the model
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Predict on test set
predictions = model.predict(X_test)

# Display a sample prediction
index = np.random.randint(0, len(X_test))
sample_input, sample_label, sample_prediction = X_test[index], y_test[index], predictions[index]

plt.figure(figsize=(12, 4))

# Original Image 1
plt.subplot(1, 3, 1)
plt.imshow(sample_input[0].squeeze(), cmap='gray')
plt.title("Original Image 1")

# Original Image 2
plt.subplot(1, 3, 2)
plt.imshow(sample_input[1].squeeze(), cmap='gray')
plt.title("Original Image 2")

# Prediction
plt.subplot(1, 3, 3)
plt.imshow(sample_input[1].squeeze(), cmap='gray')
predicted_center = (int(sample_prediction[0]), int(sample_prediction[1]))
plt.scatter([predicted_center[0]], [predicted_center[1]], color='red')
plt.title("Predicted Next Circle")

plt.show()
