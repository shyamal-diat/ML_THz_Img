import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, precision_recall_curve, auc, roc_curve
import matplotlib.pyplot as plt
from data import TerahertzDataset, train_val_test_split
from model_pycharm import UNet
from torchvision.transforms import Compose, ToTensor

def evaluate(model, dataloader):
    model.eval()

    total_correct = 0
    total_samples = 0
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0
    predicted_scores = []
    true_labels = []

    with torch.no_grad():
        for lr_image, hr_image in dataloader:
            lr_image = lr_image.to(device)
            hr_image = hr_image.to(device)

            output = model(lr_image)
            predicted = output > 0

            total_correct += (predicted == hr_image).sum().item()
            total_samples += lr_image.size(0)

            # Threshold hr_image to create a binary mask
            binary_mask = hr_image > 0

            true_positives = ((predicted == 1) & (binary_mask == 1)).sum().item()
            false_positives = ((predicted == 1) & (binary_mask == 0)).sum().item()
            false_negatives = ((predicted == 0) & (binary_mask == 1)).sum().item()

            total_true_positives += true_positives
            total_false_positives += false_positives
            total_false_negatives += false_negatives

            predicted_scores.append(output.cpu().numpy().flatten())
            true_labels.append(binary_mask.cpu().numpy().flatten())

    predicted_scores = np.concatenate(predicted_scores)
    true_labels = np.concatenate(true_labels)

    precision = precision_score(true_labels, predicted_scores > 0, average='binary')
    recall = recall_score(true_labels, predicted_scores > 0, average='binary')
    f1 = f1_score(true_labels, predicted_scores > 0, average='binary')
    accuracy = accuracy_score(true_labels, predicted_scores >0)

    fpr, tpr, _ = roc_curve(true_labels, predicted_scores)
    roc_auc = roc_auc_score(true_labels, predicted_scores)

    precision_values, recall_values, _ = precision_recall_curve(true_labels, predicted_scores)
    pr_auc = auc(recall_values, precision_values)

    print('total_true_positives:', total_true_positives, "total_false_positives: ", total_false_positives,
          'total_correct:', total_correct, 'total_samples:', total_samples)

    print('Accuracy: {:.4f}'.format(accuracy))
    print('Precision: {:.4f}'.format(precision))
    print('Recall: {:.4f}'.format(recall))
    print('F1 Score: {:.4f}'.format(f1))
    print('ROC AUC Score: {:.4f}'.format(roc_auc))
    print('PR AUC Score: {:.4f}'.format(pr_auc))

    # Plot ROC curve
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()

    # Plot precision-recall curve
    plt.figure()
    lw = 2
    plt.plot(recall_values, precision_values, color='darkorange', lw=lw, label='PR curve (area = %0.2f)' % pr_auc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc="lower right")
    plt.show()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Define data transformations
data_transform = Compose([ToTensor()])

#Load the dataset
dataset = TerahertzDataset(root_dir='D:\THz\enhancement_pytorch', transform=data_transform)

#Split the dataset into training, validation, and testing sets
train_indices, val_indices, test_indices = train_val_test_split(dataset, train_percent=.6, val_percent=.2,
                                                                test_percent=.2, shuffle=True)

val_dataset = torch.utils.data.Subset(dataset, val_indices)
print(len(val_dataset), val_indices)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

model = UNet()
# Load the saved model parameters
model.load_state_dict(torch.load('model_2.pth'))
# Send the model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

evaluate(model, val_dataloader)
