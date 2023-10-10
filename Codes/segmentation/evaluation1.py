#!/usr/bin/env python
# coding: utf-8

# In[12]:


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_curve, auc, precision_recall_curve, roc_auc_score
import matplotlib.pyplot as plt
from data import TerahertzDataset, train_val_test_split
from model import UNet
from torchvision import transforms
import numpy as np


# In[2]:


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
data_transform = transforms.ToTensor()
dataset = TerahertzDataset(root_dir='D:\THz\segmentation_new', transform=data_transform)

# Split the dataset into train, validation, and test sets
train_indices, val_indices, test_indices = train_val_test_split(dataset, train_percent=0.8, val_percent=0.1, test_percent=0.1, shuffle=True)

# Define DataLoader for the test set
test_loader = DataLoader(dataset, batch_size=1, sampler=test_indices)

# Load trained model
model = UNet().to(device)
model.load_state_dict(torch.load('model_seg.pth'))
model.eval()

true_labels = []
predicted_labels = []


# In[3]:


# Evaluate the model
with torch.no_grad():
    for hr_image, hr_mask in test_loader:
        hr_image = hr_image.to(device)
        hr_mask = hr_mask.to(device)

        # Forward pass
        sr_image = model(hr_image)
        sr_image = hr_mask+sr_image
        #sr_image = abs(sr_image)
        # Convert the output to binary (0 or 1) based on a threshold
        threshold = 0.5
        predicted_mask = (sr_image > threshold).float()
        

        true_labels.extend(hr_mask.view(-1).cpu().numpy())
        predicted_labels.extend(predicted_mask.view(-1).cpu().numpy())


# In[4]:


# Convert true_labels and predicted_labels to NumPy arrays
true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)

# Define the threshold (you may need to adjust this threshold)
threshold = 0.5

# Convert true_labels and predicted_labels to binary (0 or 1)
true_labels_binary = (true_labels > threshold).astype(int)
predicted_labels_binary = (predicted_labels > threshold).astype(int)


# In[5]:


# Calculate confusion matrix
confusion = confusion_matrix(true_labels_binary, predicted_labels_binary)


# In[6]:


print(confusion)


# In[7]:


# Calculate True Positives (TP), True Negatives (TN), False Positives (FP), False Negatives (FN)
TP = sum((true == 1) and (pred == 1) for true, pred in zip(true_labels_binary, predicted_labels_binary))
TN = sum((true == 0) and (pred == 0) for true, pred in zip(true_labels_binary, predicted_labels_binary))
FP = sum((true == 0) and (pred == 1) for true, pred in zip(true_labels_binary, predicted_labels_binary))
FN = sum((true == 1) and (pred == 0) for true, pred in zip(true_labels_binary, predicted_labels_binary))

# Calculate Precision, Recall, and F1 Score
accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP) if TP + FP > 0 else 0.0
recall = TP / (TP + FN) if TP + FN > 0 else 0.0
f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0

# Print or use the precision, recall, and f1_score values as needed
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1_score:.4f}')


# In[8]:


import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(confusion_matrix, class_names):
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, square=True,
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

# Assuming you have calculated the confusion matrix
confusion_matrix = [[TN, FP],
                    [FN, TP]]

class_names = ['Negative', 'Positive']  # Replace with your class labels

plot_confusion_matrix(confusion_matrix, class_names)


# In[13]:


fpr, tpr, _ = roc_curve(true_labels_binary, predicted_labels_binary)
roc_auc = roc_auc_score(true_labels_binary, predicted_labels_binary)

precision_values, recall_values, _ = precision_recall_curve(true_labels_binary, predicted_labels_binary)
pr_auc = auc(recall_values, precision_values)


# In[19]:


# Plot ROC curve
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.gca().set_facecolor("white")
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([-0.02, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

# In[42]:

print(roc_auc)

# Plot precision-recall curve
plt.figure()
lw = 2
plt.plot(recall_values, precision_values, color='darkorange', lw=lw, label='PR curve (area = %0.2f)' % pr_auc)
plt.gca().set_facecolor("white")
plt.xlim([0.0, 1.02])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.legend(loc="lower right")
plt.show()


# In[36]:


import matplotlib.patches as patches


# In[ ]:




