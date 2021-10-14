import numpy as np
from pathlib import Path
import torch
import os
import random
import torch.optim as optim
import pandas as pd
import torch.nn as nn
import torch
from tqdm.autonotebook import tqdm, trange
from datetime import datetime
from dpipe.io import load_numpy
from torch.optim import lr_scheduler
from datetime import datetime
from dpipe.im.metrics import dice_score
import matplotlib.pyplot as plt
from IPython.display import clear_output
from matplotlib import rcParams
from time import time
from data_loaders import find_dice_score, find_dice_score_medic
rcParams['figure.figsize'] = (15,4)
from tqdm import tqdm

def train_best_medic(model, opt, loss_fn, epochs, data_tr, data_val, val):
    X_val1, X_val2, Y_val, subject = next(iter(data_val))
    X_val1 = X_val1.cuda()
    X_val2 = X_val2.cuda()
    Y_val = Y_val.cuda()

#    X_val2 = X_val2.squeeze(1)
    losses = []
    best_model = model.state_dict()
    best_dice_score = 0
    best_epoch = 0

    for epoch in range(epochs):
        avg_loss = 0
        model.train()  # train mode
        print('* Epoch %d/%d' % (epoch + 1, epochs))
        for X1_batch, X2_batch, Y_batch, subject_id in tqdm(data_tr):
            # data to device
            X1_batch = X1_batch.cuda()
            X2_batch = X2_batch.cuda()

            Y_batch = Y_batch.cuda()

            # set parameter gradients to zero
            opt.zero_grad()

            # forward

            # Y_pred = torch.max(model(X_batch), 1).values
            Y_pred = model((X1_batch, X2_batch))
            loss = loss_fn(Y_pred.float(), Y_batch.float())  # forward-pass
            loss.backward()  # backward-pass
            opt.step()  # update weights

            # calculate loss to show the user
            avg_loss += loss / len(data_tr)


        losses.append(avg_loss)

        mean, std = find_dice_score_medic(model, val, 0.5)
        if mean > best_dice_score:
            best_model = model.state_dict()
            best_epoch = epoch
        losses.append(avg_loss)
        model.eval()  # testing mode

        res = np.exp(model((X_val1, X_val2)).to('cpu').detach().numpy()) > 0.5
        clear_output(wait=True)
        # plt.figure(figsize=(18, 6))
        for k in range(6):
            plt.subplot(2, 6, k + 1)
            plt.imshow(Y_val[k, 0].detach().cpu().numpy(), cmap='gray')
            plt.title('Real')
            plt.axis('off')

            plt.subplot(2, 6, k + 7)
            plt.imshow(res[k, 0, ...], cmap='gray')
            plt.title('Output')
            plt.axis('off')
        plt.show()
    model.load_state_dict(best_model)
    return model, best_epoch, losses


def train_best(model, opt, loss_fn, epochs, data_tr, data_val, val, scheduler=None):
    X_val, Y_val, subject = next(iter(data_val))
    X_val = X_val.cuda()
    Y_val = Y_val.cuda()
   # X_val = torch.nn.functional.interpolate(X_val, 512)
   # Y_val = torch.nn.functional.interpolate(Y_val, 512)

    losses = []
    best_model = model.state_dict()
    best_dice_score = 0
    best_epoch = 0
    for epoch in range(epochs):
        print('* Epoch %d/%d' % (epoch + 1, epochs))
        avg_loss = 0
        model.train()  # train mode
        for X_batch, Y_batch, subject_id in tqdm(data_tr):
         #   X_batch = torch.nn.functional.interpolate(X_batch, 512)
         #   Y_batch = torch.nn.functional.interpolate(Y_batch, 512)
            X_batch = X_batch.cuda()
            Y_batch = Y_batch.cuda()
            opt.zero_grad()

            Y_pred = model(X_batch)
            loss = loss_fn(Y_pred.float(), Y_batch.float())  # forward-pass
            loss.backward()  # backward-pass
            opt.step()  # update weights
            avg_loss += loss / len(data_tr)

#        mean, std = find_dice_score(model, val, 0.5)
#        if mean > best_dice_score:
#            best_model = model.state_dict()
#            best_epoch = epoch
#        print(f'mean of dice is {mean}')
#        if scheduler:
#           scheduler.step(mean)

        losses.append(avg_loss)
        model.eval()  # testing mode


        res = np.exp(model(X_val).to('cpu').detach().numpy()) > 0.5
        # res = model(X_val)

        clear_output(wait=True)
        # plt.figure(figsize=(18, 6))
        for k in range(4):
            plt.subplot(2, 4, k + 1)
            plt.imshow(Y_val[k, 0].detach().cpu().numpy(), cmap='gray')
            plt.title('Real')
            plt.axis('off')

            plt.subplot(2, 4, k + 5)
            plt.imshow(res[k, 0, ...], cmap='gray')
            plt.title('Output')
            plt.axis('off')
        plt.show()
 #   model.load_state_dict(best_model)
    return model, best_epoch, losses
