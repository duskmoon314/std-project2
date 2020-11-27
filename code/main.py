#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Author       : BobAnkh
# @Github       : https://github.com/BobAnkh
# @Date         : 2020-11-26 21:03:58
# @LastEditTime : 2020-11-27 16:51:06
# @Description  :

import librosa
import librosa.display
from scipy import signal
import noisereduce
import numpy as np
import math
import os
import matplotlib.pyplot as plt

# 定义参数
SR_ORIGIN = 20000  # 原始数据的采样率，已测试过所有数据相同
BAND_LOW = 0.02  # 带通滤波的下限 200Hz
BAND_HIGH = 0.3  # 带通滤波的上限 3000Hz
WINDOW_LEN = 1024  # 带通滤波的窗长
BANDPASS_FILTER = signal.firwin(
    WINDOW_LEN, [BAND_LOW, BAND_HIGH], pass_zero=False)  # 带通FIR滤波器
C0 = 343  # 声速
D = 0.2
T_NEIGHBOR = 2 * D / C0


def read_wav(path, n):
    '''
    从给定路径中读取指定序号的四个声音信号
    返回一个len=4 的list，为四个声音信号
    '''
    paths = [os.path.join(path, str(n) + '_mic' + str(i+1) + '.wav')
             for i in range(4)]
    wavs = []
    for p in paths:
        wavs.append(librosa.load(p, sr=None)[0])
    [_, sr] = librosa.load(p, sr=None)
    return wavs, sr


def reduce_noise(input):
    '''
    进行噪声去除，总共有两步
    - 进行带通滤波
    - 根据噪声频谱去除噪声
    '''
    output = []
    for y in input:
        # 带通滤波
        y_band = signal.lfilter(BANDPASS_FILTER, [1.0], y)

        # 去除通带中的噪声
        y_r = noisereduce.reduce_noise(
            audio_clip=y_band, noise_clip=y_band[0:4000], verbose=False)

        output.append(y_r)

    return output


def resample(ch1, ch2, orig_sr, target_sr):
    ''' 变换采样率 '''
    ch1_new = librosa.resample(ch1, orig_sr, target_sr)
    ch2_new = librosa.resample(ch2, orig_sr, target_sr)
    return ch1_new, ch2_new


def calc_relevance(ch1, ch2):
    ''' 计算相关函数 '''
    n_sample = len(ch1)
    n_fft = 2 ** math.ceil(math.log2(2 * n_sample - 1))
    CH1 = np.fft.fft(ch1, n_fft)
    CH2 = np.fft.fft(ch2, n_fft)
    G = np.multiply(CH1, np.conj(CH2))
    r = np.fft.fftshift(np.real(np.fft.ifft(G, n_fft)))
    return r


def calc_angle(delta_n, sr, c0, d):
    ''' 估计声源角度 '''
    delta_t = delta_n / sr
    cos_theta = c0 * delta_t / d

    # 非线性映射，减小两端的估计误差
    if cos_theta > 0.995:
        theta = math.acos(1 - 10.92 * math.exp(24.17 - 32.16 * cos_theta))
    elif cos_theta < -0.995:
        theta = math.acos(-1 + 10.92 * math.exp(24.17 + 32.16 * cos_theta))
    else:
        theta = math.acos(cos_theta)

    theta_degree = theta / math.pi * 180
    return theta_degree


def bandpass_filter(ch1, ch2):
    ''' 去除信号的低频、高频分量 '''
    ch1_ft = signal.lfilter(BANDPASS_FILTER, [1.0], ch1)
    ch2_ft = signal.lfilter(BANDPASS_FILTER, [1.0], ch2)
    return ch1_ft, ch2_ft


def estimate_angle(c1, c2, c3, c4, sample_rate):
    '''
    估计声源角度，入口函数
    c1, c2, c3, c4: 4 channel sound
    sample_rate
    '''

    sr_up = sample_rate * 16
    ch1_up, ch3_up = resample(c1, c3, sample_rate, sr_up)

    r = calc_relevance(ch1_up, ch3_up)
    n_mid = int(len(r) / 2)
    n_neighbor = int(sr_up * T_NEIGHBOR)
    delta_n = np.argmax(
        r[n_mid - n_neighbor: n_mid + n_neighbor]) - n_neighbor
    angle_13 = calc_angle(delta_n, sr_up, C0, D)
    return angle_13


def main():
    PATH = os.path.join('data', 'train')
    n = int(os.listdir(PATH)[-2].split('_')[0])
    for i in range(n):
        # 读取数据
        wav, sr = read_wav(PATH, i+1)
        # 进行降噪
        wav_rn = reduce_noise(wav)
        # 计算角度
        ang = estimate_angle(wav_rn[0], wav_rn[1], wav_rn[2], wav_rn[3], sr)


if __name__ == '__main__':
    main()
