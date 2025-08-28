import torch, os
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

# data_transform = {
#     'train': transforms.Compose([
#         transforms.Resize(256),             # Resize to 256 pixels
#         transforms.RandomResizedCrop(224),  # Resize to 224x224 pixels
#         transforms.RandomHorizontalFlip(),  # Data augmentation to change the size with p = 50%
#         transforms.ToTensor(),              # Convert to PyTorch Tensor
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         # Channels(red, green, blue): output[channel] = (input[channel] - mean[channel]) / std[channel]

#     ]),

#     'val': transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ]),

#     'test': transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ]),
# }

# data_dir = '/home/le-cuong/Downloads/COVID19/COVID-19_Radiography_Dataset'
# image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
#                                           data_transform[x])
#                     for x in ['train', 'val', 'test']}

# dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=4)
#             for x in ['train', 'val', 'test']}

# dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
# class_names = image_datasets['train'].classes

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(f"The available device: {device}")


data_dir = '/home/le-cuong/Downloads/COVID19/COVID-19_Radiography_Dataset'

def load_data(data_dir, batch_size=32, val_split=0.2, test_split=0.1):
    data_transform = transforms.Compose([
        transforms.Resize(256),             # Resize to 256 pixels
        transforms.RandomResizedCrop(224),  # Resize to 224x224 pixels
        transforms.RandomHorizontalFlip(),  # Data augmentation to change the size with p = 50%
        transforms.ToTensor(),              # Convert to PyTorch Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # Channels(red, green, blue): output[channel] = (input[channel] - mean[channel]) / std[channel]
    ])

    dataset = datasets.ImageFolder(data_dir, transform=data_transform)
    class_names = dataset.classes
    dataset_size = len(dataset)
    
    val_size = int(val_split * dataset_size)
    test_size = int(test_split * dataset_size)
    train_size = dataset_size - val_size - test_size
    
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    }
    
    dataset_sizes = {
        'train': train_size,
        'val': val_size,
        'test': test_size
    }
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"The available device: {device}")
    
    return dataloaders, dataset_sizes, class_names, device

dataloaders, dataset_sizes, class_names, device = load_data(data_dir)
print("Data Loaders: ",dataloaders)
print("Dataset sizes: ",dataset_sizes)
print("Name of classes: ",class_names)
print("GPU available: ",device)