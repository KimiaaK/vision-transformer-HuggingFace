import torch
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
import seaborn as sns
import matplotlib.pyplot as plt


def evaluate_model(model, testloader, class_names):
    model.eval()

    all_preds = []
    all_targets = []

    for data, target in testloader:
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()

        output = model(data)
        probabilities = torch.softmax(output.logits, dim=1)
        _, pred = torch.max(probabilities, 1)

        all_preds.extend(pred.cpu().numpy())
        all_targets.extend(target.cpu().numpy())

    cm = confusion_matrix(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average="weighted")
    recall = recall_score(all_targets, all_preds, average="weighted")
    f1 = f1_score(all_targets, all_preds, average="weighted")
    accuracy = accuracy_score(all_targets, all_preds)

    # Print metrics
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Accuracy:", accuracy)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="g",
        cmap="BuPu",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title("Confusion Matrix for test set")
    plt.show()
