import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Step 4.1 – Load Preprocessed Data

X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

print("Data loaded successfully.")
print("X_train shape:", X_train.shape)

# Step 4.2 – Convert Labels to Categorical (One-Hot Encoding)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print("Labels converted to categorical.")

# Step 4.3 – Build CNN Architecture

# Build CNN Model
model = Sequential()

# First Convolution Layer
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)))
model.add(MaxPooling2D(pool_size=(2,2)))

# Second Convolution Layer
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# Flatten layer
model.add(Flatten())

# Fully Connected Layer
model.add(Dense(128, activation='relu'))

# Dropout (to prevent overfitting)
model.add(Dropout(0.5))

# Output Layer (2 classes)
model.add(Dense(2, activation='softmax'))

print("Model built successfully.")

# Step 4.4 – Compile Model

model.compile(
  optimizer='adam',
  loss = 'categorical_crossentropy',
  metrics=['accuracy']
)

print("Model compiled successfully.")

# Step 4.5 – Display Model Summary

model.summary()

# Step 5.1 – Add Training Code

history = model.fit(X_train,y_train,epochs=15,batch_size=32,validation_data=(X_test,y_test))
print("Training completed.")

# Step 5.2 – Evaluate Model

loss,accuracy = model.evaluate(X_test,y_test)

print("Test Loss : ",loss)
print("Test Accuracy : ",accuracy)

# Step 5.3 – Save the Trained Model

model.save('mask_detector_model.h5')
print("Model saved as mask_detector_model.h5")

# Step 6.1 – Plot Accuracy & Loss Graphs

# Plot Accuracy
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'])
plt.show()

# Plot Loss
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'])
plt.show()

# Step 6.2 – Generate Confusion Matrix

# Predict on test data
y_pred = model.predict(X_test)

# Convert predictions back to class labels
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)
print("Confusion Matrix:")
print(cm)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes))