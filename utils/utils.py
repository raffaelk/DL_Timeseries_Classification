import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from sklearn.metrics import accuracy_score, balanced_accuracy_score

"""
some useful stuff to train the models
"""


class UCRDataset(Dataset):
    """
    dataset-object for training
    """

    def __init__(self, path):

        # read in data set as pd.DataFrame
        data = pd.read_csv(path, header=None)

        # gpu or cpu
        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"

        self.dev = torch.device(dev)

        # target-distribution
        self.val_ctn = data.iloc[:, 0].value_counts()

        # points per signal
        self.siglen = data.shape[1]-1

        # count target classes
        self.n_target_classes = self.val_ctn.shape[0]

        # split and reshape, shift targets to start from 0
        targets_raw = data.iloc[:, 0]
        uni = targets_raw.unique()
        uni.sort()
        repl = [i for i in range(len(uni))]
        targets_shift = targets_raw.replace(uni, repl)
        self.targets = targets_shift.to_numpy().reshape(-1, 1)

        # z-norm signal
        signals_np = data.iloc[:, 1:].to_numpy()
        signals_norm = np.apply_along_axis(self._z_norm, 1, signals_np)

        self.signals = signals_norm.reshape(-1, 1, self.siglen)

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):

        inps = torch.Tensor(self.signals[idx]).to(self.dev)
        outs = torch.Tensor(self.targets[idx]).long().to(self.dev)

        return {'inputs': inps, 'outputs': outs}

    def info(self):
        print(f'Signal length: {self.siglen} points')
        print(f'Size of dataset: {self.__len__()} entries')
        print('')
        print(self.val_ctn)
    
    def _z_norm(self, signal):
        mean = signal.mean()
        std = signal.std()
        return (signal - mean) / std

class UCRTorchTrainer:
    """
    object to hide all the intermediate steps during training
    """

    def __init__(self, model, criterion, optimizer):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def train(self,
              traindata_loader,
              n_epochs,
              testdata_loader=None,
              early_stopping=0,
              save_as='',
              save_log_as=''
              ):

        self.model.train()
        self.epochlosses = []
        self.train_accuracies = []
        self.lrs = []

        if testdata_loader is not None:
            self.test_accuracies = []
            self.test_bal_accuracies = []

        for epoch in range(n_epochs):

            batchlosses = []

            print(f'Epoch: {epoch}', end='')

            # used for train accuracy
            correct, total = 0, 0

            for i, batch in enumerate(traindata_loader, 0):

                x_train, y_train = batch['inputs'], batch['outputs']

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + train accuracy
                outputs = self.model(x_train)
                _, predicted = torch.max(outputs.data, 1)
                total += y_train.size(0)
                correct += (predicted == y_train.squeeze()).sum().item()

                # backward + optimize
                loss = self.criterion(outputs, y_train.squeeze())
                loss.backward()
                self.optimizer.step()

                # compute average loss of current batch
                batchlosses.append(float(loss) / traindata_loader.batch_size)

            # compute average loss of current epoch
            epochloss = np.mean(batchlosses)
            print(f' - train loss: {epochloss: .6f}', end='')
            self.epochlosses.append(epochloss)

            # compute average training accuracy of current epoch
            train_acc = correct / total
            print(f' - train accuracy: {train_acc: .6f}', end='')
            self.train_accuracies.append(train_acc)

            # compute test accuracy of current epoch
            if testdata_loader is not None:
                # evaluate test accuracies
                pred_proba_train = self.evaluate(testdata_loader)
                test_acc = accuracy_score(
                    testdata_loader.dataset.targets,
                    np.argmax(pred_proba_train, axis=1)
                    )
 
                test_bal_acc = balanced_accuracy_score(
                    testdata_loader.dataset.targets,
                    np.argmax(pred_proba_train, axis=1)
                    )
                
                print(f' - test accuracy: {test_acc: .6f}', end='')
                self.test_accuracies.append(test_acc)
                self.test_bal_accuracies.append(test_bal_acc)

            # print LR
            try:
                print(f' - LR: {self.optimizer.lr: .6f}', end='')
                self.lrs.append(self.optimizer.lr)
            except AttributeError:
                pass

            # line break
            print()

            # simple form of early stopping
            if early_stopping > 0 and \
                epoch > early_stopping and \
                (self.test_accuracies[-early_stopping:] <=
                 self.test_accuracies[-early_stopping]).all():

                # save model
                if save_as != '':
                    torch.save(self.model, save_as)
                
                # save log
                if save_log_as != '':
                    self.save_log(save_log_as)

                break

            # save model on last epoch
            if epoch == n_epochs-1 and save_as != '':
                torch.save(self.model, save_as)
            
            # save log on last epoch
            if epoch == n_epochs-1 and save_log_as != '':
                self.save_log(save_log_as)

    def evaluate(self, testdata_loader):

        # store prediction after each batch
        predictions = []

        # set model to eval mode
        self.model.eval()

        # predict batch-wise
        with torch.no_grad():

            for i, batch in enumerate(testdata_loader, 0):

                x_test, _ = batch['inputs'], batch['outputs']

                outputs = F.softmax(self.model(x_test), dim=-1)

                predictions.extend(outputs.tolist())

        # set back to train mode
        self.model.train()

        return np.array(predictions)
    
    def save_log(self, name):
        
        log = pd.DataFrame()
        log['train_loss'] = self.epochlosses
        log['train_accuracies'] = self.train_accuracies
        log['test_accuracies'] = self.test_accuracies
        log['test_bal_accuracies'] = self.test_bal_accuracies
        
        if len(self.lrs) != 0:
            log['LR'] = self.lrs
        
        log.to_csv(name)

    def plot_loss(self):

        fig, ax = plt.subplots()
        ax.plot(self.epochlosses, label='train loss')
        ax.set(
            title='loss during training',
            ylabel='loss',
            xlabel='epoch'
            )
        ax.legend(loc='best')
        plt.show()

    def plot_acc(self):

        fig, ax = plt.subplots()
        ax.plot(self.test_accuracies, label='test accuracy')
        ax.plot(self.train_accuracies, label='train accuracy')
        ax.set(
            title='accuracy during training',
            ylabel='accuracy',
            xlabel='epoch'
            )
        ax.legend(loc='best')
        plt.show()

    def show_cam(self, data_loader, idx, show=True):

        # get signal and true label
        item = data_loader.dataset.__getitem__(idx)
        signal = item['inputs']
        label = item['outputs'].numpy()[0]

        # get cam and predicted label
        cam, pred = self.model.get_cam(signal.view(1, 1, -1))

        # get only the map for the predicted class
        cas = cam[pred, :]

        # normalize map
        cam_norm = cas - np.min(cas)
        cam_norm = (cam_norm / np.max(cam_norm)) * 100

        # signal as np array
        signalnp = signal.squeeze().numpy()

        # plot clored cam on signal
        if show:
            t = np.arange(0, len(signalnp))

            fig, ax = plt.subplots()
            sc = ax.scatter(x=t, y=signalnp, c=cam_norm, cmap='jet', vmin=0, vmax=100)
            ax.set(
                title=f'CAM - index: {idx}, true label: {label}, predicted label: {pred}'
                )
            fig.colorbar(sc)
            plt.show()

        return cam_norm, signalnp, label, pred
    
    def show_attention(self, data_loader, idx, block=0, head=0, show=True):
        
        # get signal and true label
        item = data_loader.dataset.__getitem__(idx)
        signal = item['inputs']
        label = item['outputs'].numpy()[0]
        
        # get cam and predicted label
        attmatts, pred = self.model.comp_attention(signal.view(1, 1, -1))
        
        # signal as np array
        signalnp = signal.squeeze().numpy()
        
        # attention matrix of specified block and head
        attmatt = attmatts[block].numpy()[0, :, :, :]
        
        # number of heads in current block
        n_heads = attmatt.shape[0]
        
        # sum of attention values along input
        sum_in = attmatt[head, :, :].sum(axis=0)
        
                # normalize map
        sum_in_norm = sum_in - np.min(sum_in)
        sum_in_norm = (sum_in_norm / np.max(sum_in_norm)) * 100
        
        if show:
        
            # plot distribution of attention on input
            t = np.arange(0, len(signalnp))
        
            fig, axs = plt.subplots(2, 1, sharex=True)
            sc = axs[0].scatter(x=t, y=signalnp, c=sum_in, cmap='jet')
            axs[0].set(title='attention on input signal')
            axs[1].bar(t, sum_in)
            axs[1].set(title='attention weight on input features')
            fig.colorbar(sc, ax=axs, shrink=0.75)
            fig.suptitle(f'Attention - index: {idx}, true label: {label}, predicted label: {pred}')
            plt.show()

    
        
        return sum_in_norm, signalnp, label, pred

    def show_gradients(self, data_loader, idx, show=True):

        # get signal and true label
        item = data_loader.dataset.__getitem__(idx)
        signal = torch.autograd.Variable(item['inputs'], requires_grad=True)
        label = item['outputs'].numpy()[0]

        self.model.eval()
        logprob = self.model.forward(signal.view(1, 1, -1))
        self.model.train()
        
        pred = np.argmax(logprob.detach().squeeze().numpy())
        
        maxprob = logprob.max()
        maxprob.backward()
        dydx = signal.grad[0].numpy()

        # signal as np array
        signalnp = signal.detach().squeeze().numpy()

        # plot clored cam on signal
        if show:
            t = np.arange(0, len(signalnp))

            fig, ax = plt.subplots()
            sc = ax.scatter(x=t, y=signalnp, c=dydx, cmap='jet')
            ax.set(
                title=f'Gradients - index: {idx}, true label: {label}, predicted label: {pred}'
                )
            fig.colorbar(sc)
            plt.show()

        return dydx, signalnp, label, pred


def importance_per_class(model_trainer, data_loader, mode='CAM'):

    # initiate a subplot fore every target class in set
    n_classes = data_loader.dataset.n_target_classes
    fig, axs = plt.subplots(n_classes, 1, sharey=True)

    # create time-axis
    x = np.arange(0, data_loader.dataset.siglen)

    # iterate over dataset
    for idx in range(data_loader.dataset.__len__()):

        if mode == 'attention':
            cam_norm, y, label, pred = model_trainer.show_attention(data_loader, idx, show=False)
        elif mode == 'CAM':
            cam_norm, y, label, pred = model_trainer.show_cam(data_loader, idx, show=False)

        if True: #label == pred:

            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            lc = LineCollection(segments, cmap='jet', norm=plt.Normalize(0, 100))

            lc.set_array(cam_norm)
            lc.set_linewidth(1)

            axs[label].add_collection(lc)

    # create axes limits
    xmin, xmax = x.min(), x.max()
    ymin = np.amin(data_loader.dataset.signals)
    ymax = np.amax(data_loader.dataset.signals)
    xpad = 0.1 * abs(xmax-xmin)
    ypad = 0.1 * abs(ymax-ymin)


    # set design of subplots
    for i in range(n_classes):
        axs[i].set_title(f'{mode} for class {i}')
        axs[i].set_xlim(xmin-xpad, xmax+xpad)
        axs[i].set_ylim(ymin-ypad, ymax+ypad)
        fig.colorbar(lc, ax=axs[i])

    # finalize plot
    plt.show()
