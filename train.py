from utils.datamanager import *
from utils.utils import *
from models.FCN import *
import torch.nn as nn
import torch.optim as optim
import torch
import time

# test_loader = DataLoader(dataset=PASCALVOCDataset('./data/val/'), batch_size=32, shuffle=True,
#                           pin_memory=True, drop_last=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print_freq = 10

def train(model, loader, epoch, criterion, optimizer):
    model.train()
    losses = ExpoAverageMeter()  # loss (per word decoded)
    for i_batch, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()

        y_hat = model(x)

        loss = criterion(y_hat[0], y)
        loss.backward()

        losses.update(loss.item())
        optimizer.step()

        if i_batch % print_freq == 0:
            print(f'Epoch: [{epoch+1:02}/??][{i_batch}/{len(loader)}]\t Loss {losses.val:.7f} ({losses.avg:.7f})\t')


def valid(val_loader, model, criterion):
    model.eval()
    batch_time = ExpoAverageMeter()  # forward prop. + back prop. time
    losses = ExpoAverageMeter()  # loss (per word decoded)
    start = time.time()
    with torch.no_grad():
        # Batches
        for i_batch, (x, y) in enumerate(val_loader):
            # Set device options
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x)
            print(y_hat[0].shape, y.shape)
            loss = criterion(y_hat[0], y)
            #loss = torch.sqrt((y_hat - y).pow(2).mean())
            # Keep track of metrics
            losses.update(loss.item())
            batch_time.update(time.time() - start)
            start = time.time()
            # Print status
            if i_batch % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.7f} ({loss.avg:.7f})\t'.format(i_batch, len(val_loader),
                                                                      batch_time=batch_time,
                                                                      loss=losses))
    return losses.avg

def main():

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [40, xxx] -> [10, ...], [10, ...], [10, ...], [10, ...] on 4 GPUs
        model = nn.DataParallel(model)
    elif torch.cuda.device_count() == 1:
        print("Let's use", torch.cuda.device_count(), "GPU!")
    else:
        print("Let's use CPUs")


    train_loader = DataLoader(dataset=PASCALVOCDataset('./data/train/'), batch_size=32, shuffle=True,
                                  pin_memory=True, drop_last=True)
    val_loader = DataLoader(dataset=PASCALVOCDataset('./data/val'), batch_size=32, shuffle=False,
                            pin_memory=True, drop_last=True)
    model = FCN().to(device)
    # input = torch.randn((10,3,320,480)).to(device)

    best_loss = 100000
    epochs_since_improvement = 0

    epochs = 200
    for epoch in range(epochs):
        criterion = nn.MSELoss().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # One epoch's training
        train(model, train_loader, epoch, criterion, optimizer)
        # One epoch's validation
        val_loss = valid(val_loader, model, criterion)
        print('\n * LOSS - {loss:.8f}\n'.format(loss=val_loss))
        # Check if there was an improvement
        is_best = val_loss < best_loss
        best_loss = min(best_loss, val_loss)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0
        # Save checkpoint
        save_checkpoint(epoch, model, optimizer, val_loss, is_best)

if __name__ == '__main__':
    main()