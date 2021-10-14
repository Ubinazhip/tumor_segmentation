import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch
from tqdm.autonotebook import tqdm, trange
from torch.optim import lr_scheduler
from losses import dice_metric, dice_loss_4img, DiceLoss
import copy
from collections import defaultdict
from MaskBinarization import TripletMaskBinarization
#rcParams['figure.figsize'] = (15,4)
from data_loaders import find_dice_score_medic, find_dice_score

class Training():
    def __init__(self,loss_fn, opt, scheduler, epochs, eval_fn, triplets=None, is_medic=False, is_4out = False):

        self.loss_fn = loss_fn
        self.opt = opt
        self.scheduler = scheduler
        self.epochs = epochs
        self.eval_fn = eval_fn
        self.is_medic = is_medic
        self.triplets = triplets
        self.is_4out = is_4out
    def average_predict(self, pred):
        return torch.mean(pred, dim = 1)[:,None,:,:,]
    def train_epoch(self, model, dataset):
        train_tqdm = tqdm(dataset, leave=False)
        average_loss = 0
        cnt = 0
        model.train()
        for idx, (X, y, extra) in enumerate(train_tqdm):
            X = X.cuda()
            y = y.cuda()

            self.opt.zero_grad()
            if self.is_4out:
                predicted = model(X)
                loss_4out = dice_loss_4img(predicted)
                predicted = self.average_predict(predicted)

                loss = self.loss_fn(predicted, y) + loss_4out

            else:
                predicted = model(X)
                loss = self.loss_fn(predicted, y)
            loss.backward()
            self.opt.step()

            cnt += 1
            average_loss += loss.item()#X.shape[0]
        


       # return average_loss / len(self.data_tr)
        return average_loss / cnt

    def validate_epoch(self, model, dataset):
        model.eval()
        if self.triplets:

            mask_binar = TripletMaskBinarization(
                triplets=self.triplets
                )
        else:
            mask_binar = TripletMaskBinarization(
            triplets = [[0.75, 1000, 0.3]]
            )


        used_thresholds = mask_binar.thresholds
        metrics = defaultdict(float)
        tqdm_loader = tqdm(dataset)
        for batch_idx, (X, y, subj) in enumerate(tqdm_loader):
            X, y = X.cuda(), y.cuda()
            with torch.no_grad():
                pred = model(X)
                if self.is_4out:
                    pred = self.average_predict(pred)

            res = torch.exp(pred)
            pred_mask = mask_binar.transform(res)
            for current_thr, current_mask in zip(used_thresholds, pred_mask):
                current_metric = dice_metric(current_mask, y, per_image=True).item()
                current_thr = tuple(current_thr)
                metrics[current_thr] = (metrics[current_thr] * batch_idx + current_metric) / (batch_idx + 1)
            best_threshold = max(metrics, key=metrics.get)
            best_metric = metrics[best_threshold]
            tqdm_loader.set_description('score: {:.5} on {}'.format(best_metric, best_threshold))

        return best_metric, best_threshold

    def train_epoch_medic(self, model, dataset):
        average_loss = 0
        cnt = 0
        model.train()
        for X1,X2, y, extra in tqdm(dataset):
            X1 = X1.cuda()
            X2 = X2.cuda()
            y = y.cuda()
            cnt += 1

            self.opt.zero_grad()
            if self.is_4out:
                predicted = model((X1, X2))
                loss_4out = dice_loss_4img(predicted)
                predicted = self.average_predict(predicted)

                loss = self.loss_fn(predicted, y) + loss_4out
            else:
                predicted = model((X1, X2))
                loss = self.loss_fn(predicted, y)

            loss.backward()
            self.opt.step()
            average_loss += loss.item()

        return average_loss / cnt
    def validate_epoch_medic(self, model, dataset):
        model.eval()
        if self.triplets:

            mask_binar = TripletMaskBinarization(
                triplets=self.triplets
                )
        else:
            mask_binar = TripletMaskBinarization(
            triplets = [[0.75, 1000, 0.3]]
            )
        used_thresholds = mask_binar.thresholds
        metrics = defaultdict(float)
        tqdm_loader = tqdm(dataset)

        for batch_idx, (X1, X2, y, extra) in enumerate(tqdm_loader):
            X1, X2, y = X1.cuda(), X2.cuda(), y.cuda()


            with torch.no_grad():
                pred = model((X1, X2))
                if self.is_4out:
                    pred = self.average_predict(pred)

            res = torch.exp(pred)
            pred_mask = mask_binar.transform(res)
            for current_thr, current_mask in zip(used_thresholds, pred_mask):
                current_metric = dice_metric(current_mask, y, per_image=True).item()
                current_thr = tuple(current_thr)
                metrics[current_thr] = (metrics[current_thr] * batch_idx + current_metric) / (batch_idx + 1)
            best_threshold = max(metrics, key=metrics.get)
            best_metric = metrics[best_threshold]
            tqdm_loader.set_description('score: {:.5} on {}'.format(best_metric, best_threshold))


        return best_metric, best_threshold

    def run_train(self, model, data_tr, data_val, val):
        model.train()
        best_dice = 0
        best_epoch = 0
        best_model = copy.deepcopy(model.state_dict())
        for epoch in range(self.epochs):
            print(f'epoch: {epoch}, training....')
            if self.is_medic:
                train_loss = self.train_epoch_medic(model, data_tr)
                print(f'epoch: {epoch}, validating....')
                dice, threshold = self.validate_epoch_medic(model, data_val)
                #dice, dice_std = find_dice_score_medic(model, val, 0.5)

            else:
                train_loss = self.train_epoch(model, data_tr)
                print(f'epoch: {epoch}, validating....')
                dice, threshold = self.validate_epoch(model, data_val)
                #dice, dice_std = find_dice_score(model, val, 0.5)

            if dice > best_dice:
                best_dice = dice
                best_epoch = epoch
                best_model = copy.deepcopy(model.state_dict())

            if self.scheduler:
                self.scheduler.step(dice)
            
            print(f'train loss is {train_loss:.3f} and validation dice is {dice:.3f}')
            print(f'best_threshold is {threshold}')
            print('\n ----------------------------------------------------------------')

        print(f'best dice score was achieved {best_dice:.3f} in epoch number {best_epoch}')
        model.load_state_dict(best_model)
        return model


