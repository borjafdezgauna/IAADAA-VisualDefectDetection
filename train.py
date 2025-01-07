import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from utils.dataloader import get_train_test_loaders, get_cv_train_test_loaders
from utils.model import CustomVGG
from utils.helper import train, evaluate, predict_localize, create_dir
from utils.constants import NEG_CLASS

data_folder = "data/mvtec_anomaly_detection"
subset_name = "leather"
data_folder = os.path.join(data_folder, subset_name)

batch_size = 10
target_train_accuracy = 0.98
lr = 0.0001
epochs = 10
class_weight = [1, 3] if NEG_CLASS == 1 else [3, 1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

heatmap_thres = 0.7
n_cv_folds = 5


train_loader, test_loader = get_train_test_loaders(root=data_folder, batch_size=batch_size, test_size=0.2, random_state=42)


model = CustomVGG()

class_weight = torch.tensor(class_weight).type(torch.FloatTensor).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weight)
optimizer = optim.Adam(model.parameters(), lr=lr)


model = train(train_loader, model, optimizer, criterion, epochs, device, target_train_accuracy)

create_dir("weights")
model_path = f"weights/{subset_name}_model.h5"
torch.save(model, model_path)
# model = torch.load(model_path, map_location=device)

create_dir("output")
output_image_path = f"output/{subset_name}_confusion_matrix.png"
evaluate(model, test_loader, device, output_image_path)

# # Cross Validation

cv_folds = get_cv_train_test_loaders(root=data_folder,batch_size=batch_size,n_folds=n_cv_folds)

class_weight = torch.tensor(class_weight).type(torch.FloatTensor).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weight)

for i, (train_loader, test_loader) in enumerate(cv_folds):
    print(f"Fold {i+1}/{n_cv_folds}")
    model = CustomVGG(2) #2 classes
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model = train(train_loader, model, optimizer, criterion, epochs, device)
    output_image_path = f"output/{subset_name}_{i}_confusion_matrix.png"
    evaluate(model, test_loader, device, output_image_path)


predict_localize(model, test_loader, device, thres=heatmap_thres, n_samples=15, show_heatmap=False)


