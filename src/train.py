from torch import nn, optim

class xrayTrainV0():
    def __init__(self, model:None) -> None:
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(params=model.parameters(),
                                   lr=0.1)

class DefaultBasicTrainers:
    def train(dataloader, model, loss_fn, optimizer, device):
        size = len(dataloader.dataset)
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")