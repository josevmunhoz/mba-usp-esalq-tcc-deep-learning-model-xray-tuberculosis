# Torch imports
import torch
import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader, random_split

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
EPOCHS = 20
SEED = 42

def prepare_data():
    ### Create dataset from a specific PATH
    raw_dataset = datasets.ImageFolder(root=PATH,
                                       transform=transformers.train_transform(),
                                       target_transform=None)
    print("\nMetadata of dataset:")
    print(f"Dataset classes: {raw_dataset.classes}")

    # Calculate train and test split
    train_size = int(0.8 * len(raw_dataset))
    test_size = len(raw_dataset) - train_size
    print(f"Tot imgs to train: {train_size}")
    print(f"Tot imgs to test: {test_size}")

    ### Create DataLoader
    print("-----------------------")
    print("\nDataset loaded:")
    torch.manual_seed(seed=SEED)
    train_dataset, test_dataset = random_split(raw_dataset, [train_size, test_size])

    print("----------------------------------------------------------------------------------")
    train_dataset = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(train_dataset.dataset)
    print("\n----------------------------------------------------------------------------------")
    test_dataset = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(test_dataset.dataset)
    print("\n----------------------------------------------------------------------------------")
    
    print(f"Tot batchs to train: {len(train_dataset)} | Tot batchs to test: {len(test_dataset)}\n")

    return train_dataset, test_dataset

def train_model(model=None, train_dataset=None, device=None, train=None):
    print("\n!!!Start training!!!")

    for epoch in tqdm(range(EPOCHS)):
        print(f"Epoch {epoch}\n------")
        
        train_loss = 0
        # Add a loop to loop through the training batches
        for batch, (X, y) in enumerate(train_dataset):
            # 0.0 Moving data to desired device
            X, y = X.to(device), y.to(device)
            
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
                print(f"Looked at {batch * len(X)}/{len(train_dataset.dataset)} samples.")

        # Divide total train loss by length of train dataloader
        train_loss /= len(train_dataset)

    return train_loss

def test_model(model=None, test_dataset=None, device=None, train=None):
    print("\n!!!Start testing!!!")    
    test_loss, test_acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X_test, y_test in test_dataset:
            # 0. Move to device
            X_test, y_test = X_test.to(device), y_test.to(device)

            # 1. Forward pass
            test_pred = model(X_test)

            # 2. Calculate loss (accumulatively)
            test_loss += train.loss(test_pred, y_test)

            # 3. Caculate accuracy
            test_acc += accuracy_fn(y_true=y_test, y_pred=test_pred.argmax(dim=1))

        # Calculate the test loss average per batch
        test_loss /= len(test_dataset)

        # Calculate the test acc average per batch
        test_acc /= len(test_dataset)

    return test_loss, test_acc

def eval_model(model: torch.nn.Module,
               data_loader: DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn,
               device):
    """Return a dictionary containing the results of model predicting on data_loader"""
    loss, acc = 0,0
    model.eval()

    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            # Make predictions
            y_pred = model(X)

            # Accumulate the los and acc values per batch
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y,
                               y_pred=y_pred.argmax(dim=1))
            
        # Scale loss and acc to find the average loss/acc per batch
        loss /= len(data_loader)
        acc /= len(data_loader)

    return {"model_name": model.__class__.__name__,
            "model_loss": loss.item(),
            "model_acc": acc}

def process():
    start_time = timer()

    # Get the device
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    print("-----------------------")
    print(f"Device which will process the data: {device}")
    print("-----------------------")

    ### Prepare and load our data
    train_dataset, test_dataset = prepare_data()

    ### Create model
    model_v0 = neural_network.xrayModelv0(
        input_shape=28*28,
        hidden_units=10,
        output_shape=2,
    ).to(device)

    ### Instance train/loss and optimizer
    train = xrayTrainV0(model=model_v0)

    ### Training
    train_loss = train_model(model=model_v0, train_dataset=train_dataset, device=device, train=train)
    
    ### Testing
    # test_loss, test_acc = test_model(model=model_v0, test_dataset=test_dataset, device=device, train=train)
    
    # Print out what's hapepening
    # print(f"\n Train loss: {train_loss:.4f} | Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}")

    ### Calculate training time
    end_time = timer()
    print_train_time(start=start_time, end=end_time, device=device)

    ### Evaluate the model and collect Accuray metric
    model_v0_results = eval_model(model=model_v0,
                                  data_loader=test_dataset, 
                                  loss_fn=train.loss, 
                                  accuracy_fn=accuracy_fn,
                                  device=device)

    print(model_v0_results)

if __name__ == "__main__":
    torch.manual_seed(seed=SEED)
    process()