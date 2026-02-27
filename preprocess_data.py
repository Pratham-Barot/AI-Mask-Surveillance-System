import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Step 3.1 – Load & Resize Images
dataset_path = 'dataset'

categories = ['with_mask', 'without_mask']

data=[]
labels=[]

IMG_SIZE = 128
print("Loading images...")

for category in categories : 
  folder_path = os.path.join(dataset_path,category)
  label = categories.index(category)
  
  for img_name in os.listdir(folder_path):
    img_path = os.path.join(folder_path,img_name)
    
    try:
      img = cv2.imread(img_path)
      img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
      
      data.append(img)
      labels.append(label)
      
    except Exception as e:
      print("Error loading image : ",img_path)
      
print("Images loaded successfully.")
    
# Step 3.2 – Convert To NumPy Array & Normalize

X = np.array(data,dtype='float32')
y = np.array(labels)

X = X/255.0

print("Data shape:", X.shape)
print("Labels shape:", y.shape)

# Step 3.3 – Train-Test Split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
print("Training samples : ",X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# Step 3.4 – Save Preprocessed Data (Important)

# Save processed data for later use
np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

print("Preprocessed data saved successfully.")