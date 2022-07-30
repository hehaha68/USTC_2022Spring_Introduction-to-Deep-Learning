import torch
import torch.nn.init as init
import torch.nn as nn
import matplotlib.pyplot as plt

def init_weights(module):
    if isinstance(module, nn.Conv2d):
        init.xavier_uniform_(module.weight)
    if isinstance(module, nn.Linear):
        init.xavier_uniform_(module.weight)

def save_model(filename, model, optimizer, scheduler, epoch, loss_tr_hist, loss_val_hist, accuracy_tr_hist, accuracy_val_hist, early_stop_counter):
    state_dict = {
        'epoch':epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'loss_tr_hist': loss_tr_hist,
        'loss_val_hist': loss_val_hist,
        'accuracy_tr_hist': accuracy_tr_hist,
        'accuracy_val_hist': accuracy_val_hist,
        'early_stop_counter': early_stop_counter
    }
    torch.save(state_dict, filename)

def load_model(filename, model, optimizer = None, scheduler = None, mode = 'test'):
    state_dict = torch.load(filename)

    model.load_state_dict(state_dict['model'])
    if mode == 'test':
        return model

    epoch = state_dict['epoch']
    optimizer.load_state_dict(state_dict['optimizer'])
    loss_tr_hist = state_dict['loss_tr_hist']
    loss_val_hist = state_dict['loss_val_hist']
    accuracy_tr_hist = state_dict['accuracy_tr_hist']
    accuracy_val_hist = state_dict['accuracy_val_hist']
    early_stop_counter = state_dict['early_stop_counter']
    if scheduler is not None:
        scheduler.load_state_dict(state_dict['scheduler'])

    return epoch, model, optimizer, scheduler, early_stop_counter, loss_tr_hist, loss_val_hist, accuracy_tr_hist, accuracy_val_hist

def plot(loss_tr_hist, loss_val_hist, accuracy_tr_hist, accuracy_val_hist, top5_accuracy_tr_hist, top5_accuracy_val_hist):
    fig, ax = plt.subplots()
    fig.set_figwidth(10)
    fig.set_figheight(10)

    plt.subplot(2, 2, 1)
    plt.plot(loss_tr_hist)
    plt.plot(loss_val_hist)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(('Training', 'Validation'))

    plt.subplot(2, 2, 2)
    plt.plot(accuracy_tr_hist)
    plt.plot(accuracy_val_hist)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(('Training', 'Validation'))

    plt.subplot(2, 2, 3)
    plt.plot(top5_accuracy_tr_hist)
    plt.plot(top5_accuracy_val_hist)
    plt.xlabel('Epoch')
    plt.ylabel('Top5 Accuracy')
    plt.legend(('Training', 'Validation'))
    plt.show()