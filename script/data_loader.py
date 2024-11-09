import os
import splitfolders
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def prepare_data(data_dir, train_batch_size=4, num_workers=4):
    splitfolders.ratio(data_dir, output="output", seed=1337, ratio=(0.8, 0.2))

    data_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    train_dir = "output/train"
    test_dir = "output/val"

    train_dataset = datasets.ImageFolder(train_dir, data_transforms)
    test_dataset = datasets.ImageFolder(test_dir, data_transforms)

    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    testloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers
    )

    return trainloader, testloader, train_dataset.classes
