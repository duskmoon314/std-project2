import librosa
import librosa.display
from scipy import signal
import noisereduce

# 定义参数
BAND_LOW = 0.02  # 带通滤波的下限 200Hz
BAND_HIGH = 0.3  # 带通滤波的上限 3000Hz
WINDOW_LEN = 1024  # 带通滤波的窗长
BANDPASS_FILTER = signal.firwin(
    WINDOW_LEN, [BAND_LOW, BAND_HIGH], pass_zero=False)  # 带通FIR滤波器


def read_wav(path, n):
    '''
    从给定路径中读取指定序号的四个声音信号
    返回一个列表，列表中是四个(y, sample_rate)的元组
    '''
    paths = [path+'/'+str(n)+'_mic'+str(i+1)+'.wav' for i in range(4)]
    wavs = []
    for p in paths:
        wavs.append(librosa.load(p, sr=None))
    return wavs


def reduce_noise(y):
    '''
    进行噪声去除，总共有两步
    - 进行带通滤波
    - 根据噪声频谱去除噪声
    '''
    # 带通滤波
    y_band = signal.lfilter(BANDPASS_FILTER, [1.0], y)
    # 去除通带中的噪声
    y_r = noisereduce.reduce_noise(
        audio_clip=y_band, noise_clip=y_band[0:4000], verbose=False)

    return y_r


def main():
    pass


if __name__ == '__main__':
    main()
