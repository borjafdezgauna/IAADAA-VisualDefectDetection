import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim

from utils.mvtec_dataloader import get_train_test_loaders
from utils.vgg16_defect_classifier import VGGDefectClassifier
from utils.helper import train, evaluate, create_dir, get_bbox_from_heatmap
from utils.constants import INPUT_IMG_SIZE
from PIL import Image
import torchvision.transforms

def predict(model_path, image_path):
    image = Image.open(image_path)
    transformation = torchvision.transforms.Compose(
            [torchvision.transforms.Resize(INPUT_IMG_SIZE), torchvision.transforms.ToTensor()]
        )
    image = transformation(image)
    image = torch.unsqueeze(image, 0)
        
    model = VGGDefectClassifier()
 
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    probs = model(image)

    defectProb = probs[0][0].item()
    noDefectPro = probs[0][1].item()
    
    if defectProb > noDefectPro:
        print(f"Classified as defective")
    else:
        print(f"Classified as non-defective")

    

def train_model(data_folder= "data/mvtec_anomaly_detection", subset_name= "leather"):
 
    data_folder = os.path.join(data_folder, subset_name)
    
    batch_size = 10
    target_train_accuracy = 0.98
    lr = 0.0001
    epochs = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    
    train_loader, test_loader = get_train_test_loaders(root=data_folder, batch_size=batch_size, test_size=0.2, random_state=42)
    
    model = VGGDefectClassifier()

    class_weight = [1, 3]
    class_weight = torch.tensor(class_weight).type(torch.FloatTensor).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    
    model = train(train_loader, model, optimizer, criterion, epochs, device, target_train_accuracy)
    
    create_dir("weights")
    model_path = f"weights/{type(model).__name__}_{subset_name}.h5"
    torch.save(model.state_dict(), model_path)
    


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print("Usage: python -m defect_classification <command> [parameters]")
    elif sys.argv[1] == 'train':
        train_model("data/mvtec_anomaly_detection", "leather")
    elif sys.argv[1] == 'predict':
        if len(sys.argv) <= 3:
            print("To run prediction, the model and the image need to be passed as arguments")
            print("Example python -m defect_classification predict <path_to_model> <path_to_iamge>")
            exit()
        predict(sys.argv[2], sys.argv[3])
    else:
        print("Unknown command. Valid commands are 'train', 'cross-validation' and 'predict'")
        print("Usage: python -m defect_classification [command]")