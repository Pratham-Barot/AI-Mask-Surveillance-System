import os
import cv2
import matplotlib.pyplot as plt

dataset_path = 'dataset'

with_mask_path = os.path.join(dataset_path,'with_mask')
without_mask_path = os.path.join(dataset_path,'without_mask')

sample_with_mask = os.listdir(with_mask_path)[0]
sample_without_mask = os.listdir(without_mask_path)[0]

img1 = cv2.imread(os.path.join(with_mask_path,sample_with_mask))
img2 = cv2.imread(os.path.join(without_mask_path,sample_without_mask)) 

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.imshow(img1)
plt.title("With Mask")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(img2)
plt.title("Without Mask")
plt.axis("off")

plt.show()