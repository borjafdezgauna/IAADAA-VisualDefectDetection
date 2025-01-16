import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from utils.dataloader import get_train_test_loaders, get_cv_train_test_loaders
from utils.model import CustomVGG
from utils.helper import train, evaluate, predict_localize, create_dir, get_bbox_from_heatmap
from utils.constants import (NEG_CLASS,INPUT_IMG_SIZE)

def predict(model_path = "weights/leather_model.h5", image_path = "data/mvtec_anomaly_detection/leather/test/cut/004.png"):
    from PIL import Image
    import torchvision.transforms
    
    
#    image = read_image(image_path)
    image = Image.open(image_path)
    transformation = torchvision.transforms.Compose(
            [torchvision.transforms.Resize(INPUT_IMG_SIZE), torchvision.transforms.ToTensor()]
        )
    image = transformation(image)
    image = torch.unsqueeze(image, 0)
        
    model = CustomVGG()
 
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, weights_only=True))# = torch.load(model_path, map_location=device)
    model.eval()
    probs, heatmap = model(image)
    x0,y0,x1,y1 = get_bbox_from_heatmap(heatmap[0][NEG_CLASS].detach().numpy())

    noDefectPro = probs[0][0].item()
    defectProb = probs[0][1].item()
    if defectProb > noDefectPro:
        print(f"Defect detected at: [{x0},{y0},{x1},{y1}]")
    else:
        print(f"No defect detected")

    

def train_model(data_folder= "data/mvtec_anomaly_detection", subset_name= "leather"):
 
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
    torch.save(model.state_dict(), model_path)
    
    create_dir("output")
    output_image_path = f"output/{subset_name}_confusion_matrix.png"
    evaluate(model, test_loader, device, output_image_path)
    

def cross_validation(data_folder= "data/mvtec_anomaly_detection", subset_name= "leather"):
    
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


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print("Usage: python -m cnn <command> [parameters]")
    elif sys.argv[1] == 'train':
        train_model()
    elif sys.argv[1] == 'cross-validation':
        cross_validation()
    elif sys.argv[1] == 'predict':
        if len(sys.argv) <= 3:
            print("To run prediction, the model and the image need to be passed as arguments")
            print("Example python -m cnn predict <path_to_model> <path_to_iamge>")
            exit()
        
        predict(sys.argv[2], sys.argv[3])
    else:
        print("Unknown command. Valid commands are 'train', 'cross-validation' and 'predict'")
        print("Usage: python -m cnn [command]")