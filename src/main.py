import torch
import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader
from torch import nn

# Custom imports
import transformers
import neural_network
from train import DefaultBasicTrainers
from test import DefaultBasicTester

PATH = "../data/raiox/classified"
BATCH_SIZE = 64
EPOCHS = 20

def process():
    raw_dataset = datasets.ImageFolder(root=PATH,
                                       transform=transformers.defaultTransform(),
                                       target_transform=None)
    print("\nMetadata of dataset:")
    print(raw_dataset.classes)
    print(f"Tot imgs: {len(raw_dataset)}")

    # Calculate Indexes
    indexes = list(range(len(raw_dataset)))
    np.random.shuffle(indexes)
    split = int(0.2 * len(raw_dataset))
    train_indexes, test_indexes = indexes[split:], indexes[:split]

    # Create SubsetRandomSampler
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indexes)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_indexes)
    print(f"Tot imgs to train: {len(train_sampler)}")
    print(f"Tot imgs to test: {len(test_sampler)}")

    # Create DataLoader
    print("\n\nDataset loaded:")
    train_dataloader = DataLoader(dataset=raw_dataset,
                                  batch_size=BATCH_SIZE,
                                  sampler=train_sampler,)
    
    test_dataloader = DataLoader(dataset=raw_dataset,
                                 batch_size=BATCH_SIZE,
                                 sampler=test_sampler,)
    
    print(train_dataloader)
    print(test_dataloader)
    print(len(train_dataloader), len(test_dataloader))

    # Get the device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # Create model arch
    model = neural_network.NeuralNetwork().to(device)
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # Train and Test the model
    for t in range(EPOCHS):
        print(f"Epoch {t+1}\n-------------------------------")
        DefaultBasicTrainers.train(train_dataloader, model, loss_fn, optimizer, device)
        DefaultBasicTester.test(test_dataloader, model, loss_fn, device)
    print("Done!")

if __name__ == "__main__":
    process()