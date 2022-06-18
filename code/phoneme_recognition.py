import h5py
from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.signal
import sklearn
from sklearn.metrics import accuracy_score
import sklearn.preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import wandb

from audiofeatures import encode_lpc, decode_lpc


def notch_filtering(ecog, frequency):
    ecog_filtered = ecog
    for w0 in [48, 49, 50, 100, 150]:
        norch_b, norch_a = scipy.signal.iirnotch(w0, Q=10, fs=frequency)
        ecog_filtered = scipy.signal.filtfilt(norch_b, norch_a, ecog_filtered, axis=0)
    return ecog_filtered


def remove_eyes_artifacts(ecog, frequency):
    HIGH_PASS_FREQUENCY = 20
    bgamma, agamma = scipy.signal.butter(5, HIGH_PASS_FREQUENCY / (frequency / 2), btype='high')
    return scipy.signal.filtfilt(bgamma, agamma, ecog, axis=0)


def remove_target_leakage(ecog, frequency):
    LOW_PASS_FREQUENCY = 250
    bgamma, agamma = scipy.signal.butter(5, LOW_PASS_FREQUENCY / (frequency / 2), btype='low')
    return scipy.signal.filtfilt(bgamma, agamma, ecog, axis=0)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, 
                 files_list, is_train,  # common parameters
                 frequency, downsampling_coef, selected_channels, lag_backward, lag_forward,  # ecog parameters
                 window_size, step_size, lpc_order, audio_feature, clustering_model  # audio parameters
                ):
        self.frequency = frequency
        self.downsampling_coef = downsampling_coef
        self.lag_backward = lag_backward
        self.lag_forward = lag_forward
        self.ecog_len_window = (self.lag_backward + self.lag_forward) // self.downsampling_coef
        self.window_size = window_size
        self.step_size = step_size

        ecog_merged = []
        sound_merged = []
        
        for file in tqdm(files_list):
            with h5py.File(file, 'r+') as input_file:
                data = input_file['RawData']['Samples'][()]
            ecog = data[:, selected_channels].astype("double")
            ecog = scipy.signal.decimate(ecog, downsampling_coef, axis=0)
            ecog = remove_eyes_artifacts(ecog, frequency // downsampling_coef)
            ecog = notch_filtering(ecog, frequency // downsampling_coef)
            ecog = remove_target_leakage(ecog, frequency // downsampling_coef)
            ecog = sklearn.preprocessing.scale(ecog, copy=False)
            ecog_merged.append(ecog)
            
            sound = data[:, 31].astype("double")
            sound = encode_lpc(sound, window_size, step_size, lpc_order, audio_feature).T
            sound_merged.append(sound)
            
        
        ecog_merged = np.concatenate(ecog_merged, axis=0)
        sound_merged = np.concatenate(sound_merged, axis=0)
        
        if is_train:
            clustering_model.fit(sound_merged)
        
        self.ecog = ecog_merged.astype("float32").T
        self.phoneme = clustering_model.predict(sound_merged).astype("int64")
        
        i_min = max(0, (lag_backward -  window_size + step_size - 1) // step_size)
        len_sound = (len(self.phoneme) - 1) * step_size + window_size
        i_max = min(len(self.phoneme), (len_sound - lag_forward - window_size) // step_size)
        self.fitting_indexes = np.arange(i_min, i_max + 1)


    def remove_objects(self, indexes):
        self.fitting_indexes = np.delete(self.fitting_indexes, indexes)        


    def return_removed(self):
        i_min = max(0, (self.lag_backward -  self.window_size + self.step_size - 1) // self.step_size)
        len_sound = (len(self.phoneme) - 1) * self.step_size + self.window_size
        i_max = min(len(self.phoneme), (len_sound - self.lag_forward - self.window_size) // self.step_size)
        self.fitting_indexes = np.arange(i_min, i_max + 1)


    def get_all_phonemes(self):
        return self.phoneme[self.fitting_indexes]


    def __len__(self):
        return len(self.fitting_indexes)
    
    def __getitem__(self, index):
        i = self.fitting_indexes[index]
        ecog_start_window = (i * self.step_size + self.window_size - self.lag_backward) // self.downsampling_coef
        X = self.ecog[:,ecog_start_window:ecog_start_window+self.ecog_len_window]
        y = self.phoneme[i]

        return (X, y)


def train_one_epoch(model, train_dataloader, criterion, optimizer, epoch=0, n_epochs=1, device="cpu"):
    model.train()
    running_loss, running_accuracy = 0.0, 0.0
    for X_batch, y_batch in tqdm(train_dataloader, desc=f"Training {epoch}/{n_epochs}"):
        optimizer.zero_grad()
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * X_batch.shape[0]
        running_accuracy += (output.argmax(dim=-1) == y_batch).sum().item()
    running_loss /= len(train_dataloader.dataset)
    running_accuracy /= len(train_dataloader.dataset)
    return running_loss, running_accuracy


def predict(model, val_dataloder, criterion, epoch=0, n_epochs=1, device="cpu"):
    model.eval()
    
    running_loss = 0.0
    predicted_classes = np.array([])
    true_classes = np.array([])

    for X_batch, y_batch in tqdm(val_dataloder, desc=f"Testing {epoch}/{n_epochs}"):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        with torch.no_grad():
            output = model(X_batch)
            loss = criterion(output, y_batch)
        
        running_loss += loss.item() * X_batch.shape[0]
        predicted_classes = np.append(predicted_classes, output.argmax(dim=-1).cpu().detach().numpy())
        true_classes = np.append(true_classes, y_batch.cpu().detach().numpy()) 
    
    running_loss /= len(val_dataloder.dataset)
    return running_loss, predicted_classes, true_classes


def plot_losses(train_losses, test_losses, train_accuracies, test_accuracies):
    clear_output()
    fig, axs = plt.subplots(1, 2, figsize=(13, 4))
    axs[0].plot(range(1, len(train_losses) + 1), train_losses, label='train')
    axs[0].plot(range(1, len(test_losses) + 1), test_losses, label='test')
    axs[0].set_ylabel('loss')

    axs[1].plot(range(1, len(train_accuracies) + 1), train_accuracies, label='train')
    axs[1].plot(range(1, len(test_accuracies) + 1), test_accuracies, label='test')
    axs[1].set_ylabel('accuracy')

    for ax in axs:
        ax.grid()
        ax.set_xlabel('epoch')
        ax.legend()

    plt.show()


def train(model, train_dataloader, val_dataloader, criterion, optimizer, device="cpu", n_epochs=10, save_path=None, use_plt=False, use_wandb=False):
    model.to(device)
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    for epoch in range(n_epochs):
        cur_loss, cur_acc = train_one_epoch(model, train_dataloader, criterion, optimizer, epoch, n_epochs, device)
        train_losses.append(cur_loss)
        train_accuracies.append(cur_acc)
        
        cur_loss, predicted_classes, true_classes = predict(model, val_dataloader, criterion, epoch, n_epochs, device)
        test_losses.append(cur_loss)
        test_accuracies.append(accuracy_score(true_classes, predicted_classes))
        if use_plt:
            plot_losses(train_losses, test_losses, train_accuracies, test_accuracies)
        if use_wandb:
            wandb.log({
                "train_loss": train_losses[-1],
                "test_loss": test_losses[-1],
                "train_accuracy": train_accuracies[-1],
                "test_accuracy": test_accuracies[-1]
            })
        if save_path:
            torch.save(model.state_dict(), save_path)


def predict_proba(model, dataloader, device="cpu"):
    model.eval()
    
    probas = []

    for X_batch, y_batch in tqdm(dataloader):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        with torch.no_grad():
            output = model(X_batch)
            batch_probas = F.softmax(output, dim=-1).cpu().detach().numpy()
            probas.append(batch_probas)

    probas = np.concatenate(probas, axis=0)

    return probas


# updated version of https://medium.com/@zhe.feng0018/coding-viterbi-algorithm-for-hmm-from-scratch-ca59c9203964
def viterbi(A, B, pi):
    """
        viterbi algorithm
        :param A: the transition matrix
        :param B: the emission matrix
        :param pi: the initial probability distribution
    """
    N = B.shape[0]
    x_seq = np.zeros([N, 0], dtype=int)
    V = np.log(B[:, 0]) + np.log(pi)

    # forward to compute the optimal value function V
    for y_ in range(1, B.shape[1]):
        _V = np.log(np.tile(B[:, y_], reps=[N, 1]).T) + np.log(A.T) + np.tile(V, reps=[N, 1])
        x_ind = np.argmax(_V, axis=1)
        x_seq = np.hstack([x_seq, np.c_[x_ind]])
        V = _V[np.arange(N), x_ind]
    x_T = np.argmax(V)

    # backward to fetch optimal sequence
    x_seq_opt, i = np.ones(x_seq.shape[1]+1, dtype=int), x_seq.shape[1]
    prev_ind = x_T
    while i >= 0:
        x_seq_opt[i] = prev_ind
        i -= 1
        prev_ind = x_seq[prev_ind, i]
    return x_seq_opt


def apply_viterbi_to_windows(A, B, pi, window_size=50):
    """
        viterbi algorithm
        :param A: the transition matrix
        :param B: the emission matrix
        :param pi: the initial probability distribution
        :param window_size: the number of samples in one window
    """
    blocks_number = B.shape[1] // window_size
    result = np.array([], dtype=int)
    for i in range(blocks_number):
        result = np.append(result, viterbi(A, B[:,i * window_size:(i + 1) * window_size], pi))
    if blocks_number * window_size != blocks_number:
        result = np.append(result, viterbi(A, B[:,blocks_number * window_size:], pi))
    return result

