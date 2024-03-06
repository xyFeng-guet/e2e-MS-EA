import wave as we
from matplotlib import pyplot as plt
import numpy as np
from scipy.io import wavfile
import librosa
from scipy.fft import fft


filename = "..\\video\\wMbj6ajWbic_30.wav"
WAVE = we.open(filename)

plt.figure(dpi=600) # 将显示的所有图分辨率调高
plt.rc("font", family='SimHei')  # 显示中文
plt.rcParams['axes.unicode_minus'] = False  # 显示符号


# a = WAVE.getparams().nframes
# f = WAVE.getparams().framerate
#
# sample_time = 1/f
# time = a/f
#
# sample_frequency, audio_sequence = wavfile.read(filename)
# x_seq = np.array(0, time, sample_time)

# plt.plot(x_seq, audio_sequence, 'blue')
# plt.xlabel("time (s)")
# plt.savefig("..\\video\\audio_fig.png")
# plt.show()

def displayWaveform(path):  # 显示语音时域波形
    """
    display waveform of a given speech sample
    :param sample_name: speech sample name
    :param fs: sample frequency
    :return:
    """
    samples, sr = librosa.load(path, sr=16000)
    # samples = samples[6000:16000]

    print(len(samples), sr)
    time = np.arange(0, len(samples)) * (1.0 / sr)

    plt.plot(time, samples, color='blue')
    plt.title("Time domain waveform of speech signal")
    plt.xlabel("Duration (seconds)")
    plt.ylabel("Amplitude")
    plt.savefig(".\\..\\wMbj6ajWbic_30_a.png", dpi=600)
    # plt.show()


displayWaveform(filename)
