from tqdm import tqdm
import torch


def train_cl(cl, train_dataloader, e, optimizer, loss_function):
    all_loss = []
    optimizer = optimizer
    for epoch in range(1, e + 1):
        running_loss = 0
        with tqdm(train_dataloader, unit="batch") as tepoch:
            cl.train()
            for data in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                inputs, labels = data
                inputs = inputs.detach().to('cuda')
                labels = labels.unsqueeze(1).float().detach().to('cuda')
                optimizer.zero_grad()
                outputs = cl(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
        running_loss = running_loss / len(train_dataloader.dataset)
        print(f'Epoch: {epoch}, loss: {running_loss}')
        all_loss.append(running_loss)
    return all_loss


def train_frcnn(frcnn, train_dataloader, optimizer, e):
    all_loss = []
    optimizer = optimizer
    for epoch in range(1, e+1):
        running_loss = 0
        torch.cuda.empty_cache()
        with tqdm(train_dataloader, unit="batch") as tepoch:
            frcnn.train()
            for data in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                inputs, targets = data
                inputs = list(inp.detach().to('cuda') for inp in inputs)
                targets = [{key: value.detach().to('cuda') for key, value in target.items()} for target in targets]
                optimizer.zero_grad()
                losses = frcnn(inputs, targets)
                loss = sum(loss for loss in losses.values())
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
        running_loss = running_loss/len(train_dataloader.dataset)
        print(f'Epoch: {epoch}, loss: {running_loss}')
        all_loss.append(running_loss)
    return all_loss
