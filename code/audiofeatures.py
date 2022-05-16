import librosa
import matplotlib.pyplot as plt
import numpy as np
import scipy
import spectrum


def encode_phoneme_lpc(x, order, feature="lpc"):
    poly = librosa.lpc(x, order=order)
    if feature == "lpc":
        res = -poly[1:]
    elif feature == "rc":
        res = spectrum.poly2rc(poly, 0)
    elif feature == "lsf":
        res = spectrum.poly2lsf(poly)
    elif feature == "lar":
        rc = spectrum.poly2rc(poly, 0)
        res = spectrum.rc2lar(rc)
    return np.array(res)


def decode_phoneme_lpc(y, size, feature="lpc"):
    if feature == "lpc":
        poly = np.concatenate(([1], -y))
    elif feature == "rc":
        poly = spectrum.rc2poly(y)[0]
    elif feature == "lsf":
        poly = spectrum.lsf2poly(y)
    elif feature == "lar":
        rc = spectrum.lar2rc(y)
        poly = spectrum.rc2poly(rc)[0]
    src = np.random.randn(size)
    res = scipy.signal.lfilter([1], poly, src)
    return res 


def split_into_frames(x, window_size, step_size):
    if window_size == step_size:
        window_function = np.ones(window_size)
    else:
        window_function = scipy.signal.windows.hann(window_size, False)

    blocks_number = int((len(x) - window_size) / step_size) + 1
    frames = np.zeros((window_size, blocks_number))
    for i in range(blocks_number):
        frames[:,i] = window_function * x[i * step_size: i * step_size + window_size]

    return frames


def merge_frames(frames, step_size):
    n = (frames.shape[1] - 1) * step_size + frames.shape[0]
    x = np.zeros(n)
    for i in range(frames.shape[1]):
        x[i * step_size : i * step_size + frames.shape[0]] += frames[:,i]
    return x


def encode_lpc(x, window_size, step_size, order, feature="lpc"):
    frames = split_into_frames(x, window_size, step_size)
    res = []
    for i in range(frames.shape[1]):
        res.append(encode_phoneme_lpc(frames[:,i], order, feature))
    return np.column_stack(res)


def decode_lpc(feature_frames, window_size, step_size, feature="lpc"):
    source_frames = np.zeros((window_size, feature_frames.shape[1]))
    for i in range(feature_frames.shape[1]):
        source_frames[:,i] = decode_phoneme_lpc(feature_frames[:,i], window_size, feature=feature)
    return merge_frames(source_frames, step_size)
