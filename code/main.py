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
import os

# 定义参数
SR_ORIGIN = 20000  # 原始数据的采样率，已测试过所有数据相同
BAND_LOW = 0.02  # 带通滤波的下限 200Hz
BAND_HIGH = 0.3  # 带通滤波的上限 3000Hz
WINDOW_LEN = 1024  # 带通滤波的窗长
BANDPASS_FILTER = signal.firwin(
    WINDOW_LEN, [BAND_LOW, BAND_HIGH], pass_zero=False)  # 带通FIR滤波器


def read_wav(path, n):
    '''
    从给定路径中读取指定序号的四个声音信号
    返回一个len=4 的list，为四个声音信号
    '''
    paths = [os.path.join(path, str(n) + '_mic' + str(i+1) + '.wav') for i in range(4)]
    wavs = []
    for p in paths:
        wavs.append(librosa.load(p, sr=None)[0])
    return wavs


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


def main():
    PATH = os.path.join('data', 'train')
    n = int(os.listdir(PATH)[-2].split('_')[0])
    for i in range(n):
        # 读取数据
        wav = read_wav(PATH, i+1)
        # 进行降噪
        wav_rn = reduce_noise(wav)
        # 计算角度
        # TODO


if __name__ == '__main__':
    main()
