# Torch imports
import torch
import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader
from torch import nn

# Custom imports
import transformers
import neural_network
from train import xrayTrainV0
from helper_functions import print_train_time, accuracy_fn

# LOG Import
from tqdm.auto import tqdm
from timeit import default_timer as timer

PATH = "../data/raiox/classified"
BATCH_SIZE = 32
EPOCHS = 3

def prepare_data():
    ### Create dataset from a specific PATH
    raw_dataset = datasets.ImageFolder(root=PATH,
                                       transform=transformers.train_transform(),
                                       target_transform=None)
    print("\nMetadata of dataset:")
    print(f"Dataset classes: {raw_dataset.classes}")
    print(raw_dataset)

    # Calculate Indexes
    indexes = list(range(len(raw_dataset)))
    np.random.shuffle(indexes)
    split = int(0.2 * len(raw_dataset))
    train_indexes, test_indexes = indexes[split:], indexes[:split]

    ### Create SubsetRandomSampler
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indexes)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_indexes)
    print(f"Tot imgs to train: {len(train_sampler)}")
    print(f"Tot imgs to test: {len(test_sampler)}")

    ### Create DataLoader
    print("-----------------------")
    print("\nDataset loaded:")
    train_dataloader = DataLoader(dataset=raw_dataset,
                                  batch_size=BATCH_SIZE,
                                  sampler=train_sampler,)
    
    test_dataloader = DataLoader(dataset=raw_dataset,
                                 batch_size=BATCH_SIZE,
                                 sampler=test_sampler,)
    
    print(f"Tot batchs to train: {len(train_dataloader)} | Tot batchs to test: {len(test_dataloader)}\n")

    return train_dataloader, test_dataloader

def train_model(model=None, train_dataloader=None, device=None, train=None):
    print("\n!!!Start training!!!")

    for epoch in tqdm(range(EPOCHS)):
        print(f"Epoch {epoch}\n------")
        
        train_loss = 0
        # Add a loop to loop through the training batches
        for batch, (X, y) in enumerate(train_dataloader):
            # 0.0 Moving data to desired device
            X = X.to(device)
            y = y.to(device)
            
            # 0. Train
            model.train()

            # 1. Forward pass
            y_pred = model(X)

            # 2. Calculate loss (per batch)
            loss = train.loss(y_pred, y)
            train_loss += loss # accumulate train loss

            # 3. Optimizer zero grad
            train.optimizer.zero_grad()

            # 4. Loss backward
            loss.backward()

            # 5. Optimizer step
            train.optimizer.step()

            # Print out what's happening
            if batch % 400 == 0:
                print(f"Looked at {batch * len(X)}/{len(train_dataloader.dataset)} samples.")

        # Divide total train loss by length of train dataloader
        train_loss /= len(train_dataloader)

    return train_loss

def test_model(model=None, test_dataloader=None, device=None, train=None):
    print("\n!!!Start testing!!!")    
    test_loss, test_acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X_test, y_test in test_dataloader:
            # 0. Move to device
            X_test.to(device)
            y_test.to(device)

            # 1. Forward pass
            test_pred = model(X_test)

            # 2. Calculate loss (accumulatively)
            test_loss += train.loss(test_pred, y_test)

            # 3. Caculate accuracy
            test_acc += accuracy_fn(y_true=y_test, y_pred=test_pred.argmax(dim=1))

        # Calculate the test loss average per batch
        test_loss /= len(test_dataloader)

        # Calculate the test acc average per batch
        test_acc /= len(test_dataloader)

    return test_loss, test_acc

def process():
    start_time = timer()

    # Get the device
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    print("-----------------------")
    print(f"Device which will process the data: {device}")
    print("-----------------------")

    ### Prepare and load our data
    train_dataloader, test_dataloader = prepare_data()

    ### Create model
    model_v0 = neural_network.xrayModelv0(
        input_shape=28*28,
        hidden_units=10,
        output_shape=2,
    ).to(device)

    ### Instance train/loss and optimizer
    train = xrayTrainV0(model=model_v0)

    ### Training
    train_loss = train_model(model=model_v0, train_dataloader=train_dataloader, device=device, train=train)
    
    ### Testing
    test_loss, test_acc = test_model(model=model_v0, train_dataloader=train_dataloader, device=device, train=train)
    
    # Print out what's hapepening
    print(f"\n Train loss: {train_loss:.4f} | Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}")

    ### Calculate training time
    end_time = timer()
    print_train_time(start=start_time, end=end_time, device=device)

if __name__ == "__main__":
    torch.manual_seed(42)
    process()