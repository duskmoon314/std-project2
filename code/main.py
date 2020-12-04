import argparse
import glob
import os
from multiprocessing import Pool, cpu_count

import librosa
import librosa.display
import noisereduce
import numpy as np
from scipy import signal

# 定义参数
SR_ORIGIN = 20000  # 原始数据的采样率，已测试过所有数据相同
BAND_LOW = 0.02  # 带通滤波的下限 200Hz
BAND_HIGH = 0.3  # 带通滤波的上限 3000Hz
WINDOW_LEN = 1024  # 带通滤波的窗长
BANDPASS_FILTER = signal.firwin(WINDOW_LEN, [BAND_LOW, BAND_HIGH],
                                pass_zero=False)  # 带通FIR滤波器
C0 = 343  # 声速
D1 = 0.2
D2 = 0.2 / np.sqrt(2)


def read_wav(path, n):
    '''
    从给定路径中读取指定序号的四个声音信号
    返回一个len=4 的list，为四个声音信号

    Args:
        path (str): 读取音频文件的路径
        n (int): 指定序号

    Returns:
        list: 4个mic的声音信号
    '''
    paths = [
        os.path.join(path,
                     str(n) + '_mic' + str(i + 1) + '.wav') for i in range(4)
    ]
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

    Args:
        input (list): 音频信号

    Returns:
        list: 去除噪声之后的音频信号
    '''
    output = []
    for y in input:
        # 带通滤波
        y_band = signal.lfilter(BANDPASS_FILTER, [1.0], y)

        # 去除通带中的噪声
        y_r = noisereduce.reduce_noise(audio_clip=y_band,
                                       noise_clip=y_band[0:4000],
                                       verbose=False)

        output.append(y_r)

    return output


def resample(ch, orig_sr, target_sr):
    '''
    变换采样率

    Args:
        ch (array): 4个mic的音频信号
        orig_sr (int): 原始采样率
        target_sr (int): 新采样率

    Returns:
        list: 四个mic变换采样之后的结果
    '''
    ch_new = []
    for n in range(len(ch)):
        ch_new.append(librosa.resample(ch[n], orig_sr, target_sr))

    return ch_new


def calc_relevance(ch1, ch2):
    '''
    计算相关函数

    Args:
        ch1 (array): 一个mic的音频信号
        ch2 (array): 一个mic的音频信号

    Returns:
        array: 相关函数
    '''
    n_sample = len(ch1)
    n_fft = 2**np.ceil(np.log2(2 * n_sample - 1)).astype(np.int16)
    CH1 = np.fft.fft(ch1, n_fft)
    CH2 = np.fft.fft(ch2, n_fft)
    G = np.multiply(CH1, np.conj(CH2))
    r = np.fft.fftshift(np.real(np.fft.ifft(G, n_fft)))
    return r


def calc_angle(delta_n, sr, c0, d):
    '''
    计算一对mic线上声源的偏角

    Args:
        delta_n (int): 偏差值（离散值）
        sr (int): 采样率
        c0 (int): 声速
        d (int): mic之间的距离

    Returns:
        float64: 声源偏角（单位：°）
    '''
    delta_t = delta_n / sr
    cos_theta = c0 * delta_t / d

    theta_degree = np.arccos(cos_theta) / np.pi * 180
    return theta_degree


def estimate_angle(wav_rn, sample_rate):
    '''
    估计各组mic得到的声源角度

    Args:
        wav_rn (list): 降噪处理过后的音频信号
        sample_rate (int): 采样率

    Returns:
        list: 返回6组mic估计得到的声源角度
    '''
    # 升采样
    sr_up = sample_rate * 16
    ch_up = resample(wav_rn, sample_rate, sr_up)
    angle_list = []
    for i in range(3):
        for j in range(i + 1, 4):
            r = calc_relevance(ch_up[i], ch_up[j])
            n_mid = int(len(r) / 2)
            if j - i == 2:
                n_neighbor = int(sr_up * 2 * D1 / C0)
                delta_n = np.argmax(
                    r[n_mid - n_neighbor:n_mid + n_neighbor]) - n_neighbor
                angle = calc_angle(delta_n, sr_up, C0, D1)
                angle_list.append(angle)
            else:
                n_neighbor = int(sr_up * 2 * D2 / C0)
                delta_n = np.argmax(
                    r[n_mid - n_neighbor:n_mid + n_neighbor]) - n_neighbor
                angle = calc_angle(delta_n, sr_up, C0, D2)
                angle_list.append(angle)
    return angle_list


def wav_process(PATH, i):
    '''
    音频处理，在路径下读取指定序号的文件进行处理

    Args:
        PATH (str): 音频文件路径
        i (int): 指定序号

    Returns:
        float: 计算得到的音源角度（单位：°）
    '''
    # 读取数据
    wav, sr = read_wav(PATH, i + 1)
    # 进行降噪
    wav_rn = reduce_noise(wav)
    # 计算角度
    angle_list = estimate_angle(wav_rn, sr)
    # 确定基准方向
    angle_13, angle_24 = angle_list[1], angle_list[4]
    theta13p, theta13n = (180 + angle_13) % 360, 180 - angle_13
    theta24p, theta24n = (270 + angle_24) % 360, 270 - angle_24
    if angle_13 > 15 and angle_13 < 165:
        if (theta24p > theta13p - 10 and theta24p < theta13p + 10) or (
                theta24n > theta13p - 10 and theta24n < theta13p + 10):
            scope_mid = theta13p
        else:
            scope_mid = theta13n
    else:
        if (theta13p > theta24p - 10 and theta13p < theta24p + 10) or (
                theta13n > theta24p - 10 and theta13n < theta24p + 10):
            scope_mid = theta24p
        else:
            scope_mid = theta24n
    angle_base = [135, 180, 225, 225, 270, 315]
    processed_angle = []
    sum = 0
    weights = 0
    for i, elem in enumerate(angle_list):
        if elem > 15 and elem < 165:
            # 加权计算最终角度
            if elem > 65 and elem < 115:
                weight = 100
            else:
                weight = 1
            ap = (angle_base[i] + elem + 360) % 360
            an = (angle_base[i] - elem + 360) % 360
            if ap > scope_mid - 10 and ap < scope_mid + 10:
                processed_angle.append(ap)
                sum = sum + ap * weight
                weights = weights + weight
            else:
                processed_angle.append(an)
                sum = sum + an * weight
                weights = weights + weight
    return sum / weights


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=
        'This tool can estimate the Angle of Arrival (AoA) of the sound source from audio files.'
    )
    parser.add_argument('-d',
                        '--directory',
                        dest='directory',
                        default='test',
                        help='the parent directory of audio files')

    args = parser.parse_args()
    PATH = args.directory
    # PATH = 'test'
    n = int(len(glob.glob(os.path.join(PATH, '*.wav'))) / 4)
    # n = int(os.listdir(PATH)[-2].split('_')[0])
    angles = []
    print("Start Processing!")
    # for i in range(n):
    #     # 对数据进行处理
    #     ang = wav_process(PATH, i)
    #     angles.append(ang)
    args = []
    for i in range(n):
        args.append((PATH, i))
    # 采用进程池来对数据进行并行处理
    pool = Pool(cpu_count())
    angles = pool.starmap(wav_process, args)
    pool.close()
    pool.join()
    # 对角度进行四舍五入处理
    angles = np.round(np.array(angles))
    # 将结果输出到文件中
    with open(os.path.join(PATH, 'result.txt'), 'w') as f:
        for elem in angles:
            f.write("{0:.7e}\n".format(elem))
    print('Results are saved in', os.path.join(PATH, 'result.txt'))
