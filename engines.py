import tqdm
import torch

from torchmetrics.aggregation import MeanMetric


def train_one_epoch(model, loader, loss_fn, optimizer, scheduler, device):
    # set model to train mode    
    model.train()

    # average meters to trace loss and accuracy
    loss_epoch = MeanMetric()    
    
    # train loop
    for inputs, targets in tqdm.tqdm(loader): 
        # move data to device
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # forward
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # update statistics
        loss_epoch.update(loss.to('cpu'))
    
    summary = {'loss': loss_epoch.compute()}

    return summary


def eval_one_epoch(model, loader, metric_fn, loss_fn, device):
    # set model to evaluatinon mode    
    model.eval()
    
    # average meters to trace loss and accuracy
    loss_epoch = MeanMetric()
    accuracy_epoch = MeanMetric()    
    
    # train loop
    for inputs, targets in tqdm.tqdm(loader):
        # move data to device
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # forward
        with torch.no_grad():
            outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        accuracy = metric_fn(outputs, targets)
        
        # update statistics
        loss_epoch.update(loss.to('cpu'))
        accuracy_epoch.update(accuracy.to('cpu'))
    
    summary = {
        'loss': loss_epoch.compute(),
        'accuracy': accuracy_epoch.compute(),
    }

    return summary