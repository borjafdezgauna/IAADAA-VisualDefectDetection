import sys
from PIL import Image
import torchvision.transforms

def show(input_image):
    image = Image.open(input_image)
    image.show()

def resize(input_image, output_image, max_size):
    image = Image.open(input_image)
    transformations = torchvision.transforms.Compose(
            [torchvision.transforms.Resize(int(max_size)), torchvision.transforms.ToTensor()]
        )
    tensor = transformations(image)
    torchvision.utils.save_image(tensor, output_image)
    show(output_image)

def rotate(input_image, output_image, angle):
    image = Image.open(input_image)
    transformations = torchvision.transforms.Compose(
            [torchvision.transforms.transforms.RandomRotation([int(angle), int(angle)]), torchvision.transforms.ToTensor()]
        )
    tensor = transformations(image)
    torchvision.utils.save_image(tensor, output_image)
    show(output_image)

def crop(input_image, output_image, crop_size):
    image = Image.open(input_image)
    transformations = torchvision.transforms.Compose(
            [torchvision.transforms.transforms.RandomCrop(int(crop_size)), torchvision.transforms.ToTensor()]
        )
    tensor = transformations(image)
    torchvision.utils.save_image(tensor, output_image)
    show(output_image)

if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print("Usage: python -m process_image <command> [parameters]")
    elif sys.argv[1] == 'resize':
        if len(sys.argv) < 5:
            print("To resize an image, 5 additional parameters are required")
            print("Example python -m process_image resize <input_image> <output_image> <max_size>")
        resize(sys.argv[2], sys.argv[3], sys.argv[4])
    elif sys.argv[1] == 'crop':
        if len(sys.argv) < 5:
            print("To crop an image, 5 additional parameters are required")
            print("Example python -m process_image crop <input_image> <output_image> <crop_size>")
        crop(sys.argv[2], sys.argv[3], sys.argv[4])
    elif sys.argv[1] == 'rotate':
        if len(sys.argv) < 5:
            print("To rotate an image, 5 additional parameters are required")
            print("Example python -m process_image rotate <input_image> <output_image> <angle>")
        rotate(sys.argv[2], sys.argv[3], sys.argv[4])
    
    else:
        print("Unknown command. Valid commands are 'resize', 'rotate' and 'crop'")
        print("Usage: python -m anomaly_detection [command]")