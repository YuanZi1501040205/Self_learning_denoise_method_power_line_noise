# %%
import numpy as np
import random
import matplotlib.pyplot as plt
# configure the dataset size to generate
path_save = "/home/yzi/research/Self_learning_denoise_method_power_line_noise/output/figures/"
num_receiver = 1
num_shot = 1
time_step = 0.002  # sample 2 ms
fs = 1 / time_step
time_range = 5  # signal length 5s at the time domain
time_vec = np.arange(0, time_range, time_step)
# configure the number of model's layer and maximum value
num_traces = num_receiver * num_shot
num_layers = 5
max_reflection = 5
traces = []

for i in range(num_traces):
    # random choose the ten locations of the reflection layers
    loc_layers = []
    trace = np.zeros(2500)
    for j in range(num_layers):
        loc_layers.append(random.Random(1997 + j * (i + 1)).randint(j * time_vec.shape[0] / num_layers,
                                                                    ((j + 1) * time_vec.shape[0] / num_layers) - 1))
        trace[loc_layers[-1]] = random.Random(1997 + j * (i + 1)).randint(-max_reflection, max_reflection)
    traces.append(trace)
traces = np.array(traces)
traces = traces / 10

title = "Synthetic seismic reflection Time"
plt.plot(time_vec, traces[0])
plt.ylabel('Amplitude')
plt.xlabel('Time [s]')
plt.title(title)
plt.savefig(path_save + title + '.png')
plt.show()
plt.cla()
# %% synthetic seismic trace test
import numpy as np
import matplotlib.pyplot as plt

t = time_vec
# 5 hz
fm = 5 # set the main frequency component of seismic wavelet is 15hz
ricker = (1 - 2 * np.pi ** 2 * fm ** 2 * t  ** 2) * np.exp(-np.pi ** 2 * fm ** 2 * t ** 2)
time_shift = np.where(ricker == 0)[0][0]
ricker_5 = (1 - 2 * np.pi ** 2 * fm ** 2 * (t - time_vec[time_shift])  ** 2) * np.exp(-np.pi ** 2 * fm ** 2 * (t - time_vec[time_shift]) ** 2)
plt.plot(time_vec, ricker_5, label = "5hz")

# 10hz
fm = 10 # set the main frequency component of seismic wavelet is 15hz
ricker = (1 - 2 * np.pi ** 2 * fm ** 2 * t  ** 2) * np.exp(-np.pi ** 2 * fm ** 2 * t ** 2)
time_shift = np.where(ricker == 0)[0][0]
ricker_10 = (1 - 2 * np.pi ** 2 * fm ** 2 * (t - time_vec[time_shift])  ** 2) * np.exp(-np.pi ** 2 * fm ** 2 * (t - time_vec[time_shift]) ** 2)
plt.plot(time_vec, ricker_10, label = "10hz")

# 15 hz
fm = 15 # set the main frequency component of seismic wavelet is 15hz
time_step = 0.002  # sample 2 ms
time_range = 5  # signal length 5s at the time domin
time_vec = np.arange(0, time_range, time_step)
t = time_vec
ricker = (1 - 2 * np.pi ** 2 * fm ** 2 * t  ** 2) * np.exp(-np.pi ** 2 * fm ** 2 * t ** 2)
time_shift = np.where(ricker == 0)[0][0]
ricker_15 = (1 - 2 * np.pi ** 2 * fm ** 2 * (t - time_vec[time_shift])  ** 2) * np.exp(-np.pi ** 2 * fm ** 2 * (t - time_vec[time_shift]) ** 2)

plt.plot(time_vec, ricker_15, label = "15hz")

title = "RickerWavelet_Time_Domain"
plt.ylabel('Amplitude')
plt.xlabel('Time [s]')
plt.title(title)
plt.legend()
plt.savefig(path_save + title + '.png')
plt.show()
plt.cla()
plt.close()
#%%plot signal fft power
from scipy import fftpack
import numpy as np
trace = traces[0]
trace_fft = fftpack.fft(trace)

# sample freq
sample_freq = fftpack.fftfreq(trace_fft.size, d=time_step)
# power
trace_fft_power = np.abs(trace_fft)

plt.plot(sample_freq[0:401], trace_fft_power[0:401])

title = "Synthetic Seismic Reflection FFT Power"
plt.ylabel('Power')
plt.xlabel('Frequency [Hz]')
plt.title(title)
plt.savefig(path_save + title + '.png')
plt.show()
plt.cla()
plt.close()
#%%plot wavelet fft power
trace = traces[0]
ricker_5_fft = fftpack.fft(ricker_5)
ricker_10_fft = fftpack.fft(ricker_10)
ricker_15_fft = fftpack.fft(ricker_15)

# power
ricker_5_fft_power = abs(ricker_5_fft)
ricker_10_fft_power = abs(ricker_10_fft)
ricker_15_fft_power = abs(ricker_15_fft)

plt.plot(sample_freq[0:401], ricker_5_fft_power[0:401], label='5hz')
plt.plot(sample_freq[0:401], ricker_10_fft_power[0:401], label='10hz')
plt.plot(sample_freq[0:401], ricker_15_fft_power[0:401], label='15hz')


title = "Synthetic wavelet fft Power"
plt.ylabel('Power')
plt.xlabel('Frequency [Hz]')
plt.title(title)
plt.legend()
plt.savefig(path_save + title + '.png')
plt.show()
plt.cla()
plt.close()
# %% convolution wavelet with signal plot power
trace_sig_5_fft = ricker_5_fft * trace_fft
trace_sig_10_fft = ricker_10_fft * trace_fft
trace_sig_15_fft = ricker_15_fft * trace_fft

trace_sig_5_fft_power = abs(trace_sig_5_fft)
trace_sig_10_fft_power = abs(trace_sig_10_fft)
trace_sig_15_fft_power = abs(trace_sig_15_fft)

plt.plot(sample_freq[0:401], trace_sig_5_fft_power[0:401], label='5hz')
plt.plot(sample_freq[0:401], trace_sig_10_fft_power[0:401], label='10hz')
plt.plot(sample_freq[0:401], trace_sig_15_fft_power[0:401], label='15hz')

title = "Synthetic seismic signal fft Power"
plt.ylabel('Power')
plt.xlabel('Frequency [Hz]')
plt.title(title)
plt.legend()
plt.savefig(path_save + title + '.png')
plt.show()
plt.cla()
plt.close()
# %% plot time domain
trace_sig_5 = fftpack.ifft(trace_sig_5_fft).real
trace_sig_10 = fftpack.ifft(trace_sig_10_fft).real
trace_sig_15 = fftpack.ifft(trace_sig_15_fft).real

plt.plot(time_vec, trace_sig_5, label='5hz')
plt.plot(time_vec, trace_sig_10, label='10hz')
plt.plot(time_vec, trace_sig_15, label='15hz')

title = "Synthetic seismic signal time domain"
plt.ylabel('Amplitude')
plt.xlabel('Time [s]')
plt.title(title)
plt.legend()
plt.savefig(path_save + title + '.png')
plt.show()
plt.cla()
plt.close()
# %% PLot the cos harmonic noise
f0_10 = 10
f0_20 = 20
f0_50 = 50

noise_amplitude = 0.2
noise_phase = np.pi/6

noise_10hz = noise_amplitude * np.cos(2*f0_10*np.pi*time_vec + noise_phase)
noise_20hz = noise_amplitude * np.cos(2*f0_20*np.pi*time_vec + noise_phase)
noise_50hz = noise_amplitude * np.cos(2*f0_50*np.pi*time_vec + noise_phase)


plt.plot(time_vec[:250], noise_10hz[:250], label='10hz')
plt.plot(time_vec[:250], noise_20hz[:250], label='20hz')
plt.plot(time_vec[:250], noise_50hz[:250], label='50hz')

title = "Synthetic harmonic noise time domain"
plt.ylabel('Amplitude')
plt.xlabel('Time [s]')
plt.title(title)
plt.legend()
plt.savefig(path_save + title + '.png')
plt.show()
plt.cla()
plt.close()
# %% PLot the cos harmonic noise fft power

noise_10hz_fft_power = abs(fftpack.fft(noise_10hz))
noise_20hz_fft_power = abs(fftpack.fft(noise_20hz))
noise_50hz_fft_power = abs(fftpack.fft(noise_50hz))


plt.plot(sample_freq[:401], noise_10hz_fft_power[:401], label='10hz')
plt.plot(sample_freq[:401], noise_20hz_fft_power[:401], label='20hz')
plt.plot(sample_freq[:401], noise_50hz_fft_power[:401], label='50hz')

title = "Synthetic harmonic noise frequency domain power"
plt.ylabel('Amplitude')
plt.xlabel('Frequency [hz]')
plt.title(title)
plt.legend()
plt.savefig(path_save + title + '.png')
plt.show()
plt.cla()
plt.close()
# %% add (10, 20, 50 Hz)noise to the (5, 10, 15)signal. Totally 9 signal with noise
trace_noise_10_sig_5 = noise_10hz + trace_sig_5
trace_noise_20_sig_5 = noise_20hz + trace_sig_5
trace_noise_50_sig_5 = noise_50hz + trace_sig_5

trace_noise_10_sig_10 = noise_10hz + trace_sig_10
trace_noise_20_sig_10 = noise_20hz + trace_sig_10
trace_noise_50_sig_10 = noise_50hz + trace_sig_10

trace_noise_10_sig_15 = noise_10hz + trace_sig_15
trace_noise_20_sig_15 = noise_20hz + trace_sig_15
trace_noise_50_sig_15 = noise_50hz + trace_sig_15
# %% 10hz noise + 5hz signal
plt.plot(time_vec, trace_noise_10_sig_5, label='trace_noise_10_sig_5')

title = "Synthetic 10hz noise + 5hz signal time domain"
plt.ylabel('Amplitude')
plt.xlabel('Time [s]')
plt.title(title)
plt.legend()
plt.savefig(path_save + title + '.png')
plt.show()
plt.cla()
plt.close()
# %% 20hz noise + 5hz signal
plt.plot(time_vec, trace_noise_20_sig_5, label='trace_noise_20_sig_5')

title = "Synthetic 20hz noise + 5hz signal time domain"
plt.ylabel('Amplitude')
plt.xlabel('Time [s]')
plt.title(title)
plt.legend()
plt.savefig(path_save + title + '.png')
plt.show()
plt.cla()
plt.close()
# %% 50hz noise + 5hz signal
plt.plot(time_vec, trace_noise_50_sig_5, label='trace_noise_50_sig_5')

title = "Synthetic 50hz noise + 5hz signal time domain"
plt.ylabel('Amplitude')
plt.xlabel('Time [s]')
plt.title(title)
plt.legend()
plt.savefig(path_save + title + '.png')
plt.show()
plt.cla()
plt.close()
# %% 10hz noise + 10hz signal
plt.plot(time_vec, trace_noise_10_sig_10, label='trace_noise_10_sig_10')

title = "Synthetic 10hz noise + 10hz signal time domain"
plt.ylabel('Amplitude')
plt.xlabel('Time [s]')
plt.title(title)
plt.legend()
plt.savefig(path_save + title + '.png')
plt.show()
plt.cla()
plt.close()
# %% 20hz noise + 5hz signal
plt.plot(time_vec, trace_noise_20_sig_10, label='trace_noise_20_sig_10')

title = "Synthetic 20hz noise + 10hz signal time domain"
plt.ylabel('Amplitude')
plt.xlabel('Time [s]')
plt.title(title)
plt.legend()
plt.savefig(path_save + title + '.png')
plt.show()
plt.cla()
plt.close()
# %% 50hz noise + 10hz signal
plt.plot(time_vec, trace_noise_50_sig_10, label='trace_noise_50_sig_10')

title = "Synthetic 50hz noise + 10hz signal time domain"
plt.ylabel('Amplitude')
plt.xlabel('Time [s]')
plt.title(title)
plt.legend()
plt.savefig(path_save + title + '.png')
plt.show()
plt.cla()
plt.close()
# %% 10hz noise + 15hz signal
plt.plot(time_vec, trace_noise_10_sig_15, label='trace_noise_10_sig_15')

title = "Synthetic 10hz noise + 15hz signal time domain"
plt.ylabel('Amplitude')
plt.xlabel('Time [s]')
plt.title(title)
plt.legend()
plt.savefig(path_save + title + '.png')
plt.show()
plt.cla()
plt.close()
# %% 20hz noise + 15hz signal
plt.plot(time_vec, trace_noise_20_sig_15, label='trace_noise_20_sig_15')

title = "Synthetic 20hz noise + 15hz signal time domain"
plt.ylabel('Amplitude')
plt.xlabel('Time [s]')
plt.title(title)
plt.legend()
plt.savefig(path_save + title + '.png')
plt.show()
plt.cla()
plt.close()
# %% 50hz noise + 15hz signal
plt.plot(time_vec, trace_noise_50_sig_15, label='trace_noise_50_sig_15')

title = "Synthetic 50hz noise + 15hz signal time domain"
plt.ylabel('Amplitude')
plt.xlabel('Time [s]')
plt.title(title)
plt.legend()
plt.savefig(path_save + title + '.png')
plt.show()
plt.cla()
plt.close()
# %% plot the signal with noise fft power
trace_noise_10_sig_5_fft = fftpack.fft(trace_noise_10_sig_5)
trace_noise_20_sig_5_fft = fftpack.fft(trace_noise_20_sig_5)
trace_noise_50_sig_5_fft = fftpack.fft(trace_noise_50_sig_5)

trace_noise_10_sig_10_fft = fftpack.fft(trace_noise_10_sig_10)
trace_noise_20_sig_10_fft = fftpack.fft(trace_noise_20_sig_10)
trace_noise_50_sig_10_fft = fftpack.fft(trace_noise_50_sig_10)

trace_noise_10_sig_15_fft = fftpack.fft(trace_noise_10_sig_15)
trace_noise_20_sig_15_fft = fftpack.fft(trace_noise_20_sig_15)
trace_noise_50_sig_15_fft = fftpack.fft(trace_noise_50_sig_15)
# %%
plt.plot(sample_freq[:401], abs(trace_noise_10_sig_5_fft)[:401], label='trace_noise_10_sig_5')

title = "Synthetic 10hz noise + 5hz signal frequency domain"
plt.ylabel('Power')
plt.xlabel('Frequency [hz]')
plt.title(title)
plt.legend()
plt.savefig(path_save + title + '.png')
plt.show()
plt.cla()
plt.close()
# %% plot the signal with noise fft power

plt.plot(sample_freq[:401], abs(trace_noise_10_sig_10_fft)[:401], label='trace_noise_10_sig_10_fft')

title = "Synthetic 10hz noise + 10hz signal frequency domain"
plt.ylabel('Power')
plt.xlabel('Frequency [hz]')
plt.title(title)
plt.legend()
plt.savefig(path_save + title + '.png')
plt.show()
plt.cla()
plt.close()
# %% plot the signal with noise fft power

plt.plot(sample_freq[:401], abs(trace_noise_10_sig_15_fft)[:401], label='trace_noise_10_sig_15_fft')

title = "Synthetic 10hz noise + 15hz signal frequency domain"
plt.ylabel('Power')
plt.xlabel('Frequency [hz]')
plt.title(title)
plt.legend()
plt.savefig(path_save + title + '.png')
plt.show()
plt.cla()
plt.close()
# %%
plt.plot(sample_freq[:401], abs(trace_noise_20_sig_5_fft)[:401], label='trace_noise_20_sig_5')

title = "Synthetic 20hz noise + 5hz signal frequency domain"
plt.ylabel('Power')
plt.xlabel('Frequency [hz]')
plt.title(title)
plt.legend()
plt.savefig(path_save + title + '.png')
plt.show()
plt.cla()
plt.close()
# %% plot the signal with noise fft power

plt.plot(sample_freq[:401], abs(trace_noise_20_sig_10_fft)[:401], label='trace_noise_20_sig_10_fft')

title = "Synthetic 20hz noise + 10hz signal frequency domain"
plt.ylabel('Power')
plt.xlabel('Frequency [hz]')
plt.title(title)
plt.legend()
plt.savefig(path_save + title + '.png')
plt.show()
plt.cla()
plt.close()
# %% plot the signal with noise fft power

plt.plot(sample_freq[:401], abs(trace_noise_20_sig_15_fft)[:401], label='trace_noise_20_sig_15_fft')

title = "Synthetic 20hz noise + 15hz signal frequency domain"
plt.ylabel('Power')
plt.xlabel('Frequency [hz]')
plt.title(title)
plt.legend()
plt.savefig(path_save + title + '.png')
plt.show()
plt.cla()
plt.close()
# %%
plt.plot(sample_freq[:401], abs(trace_noise_50_sig_5_fft)[:401], label='trace_noise_50_sig_5')

title = "Synthetic 50hz noise + 5hz signal frequency domain"
plt.ylabel('Power')
plt.xlabel('Frequency [hz]')
plt.title(title)
plt.legend()
plt.savefig(path_save + title + '.png')
plt.show()
plt.cla()
plt.close()
# %% plot the signal with noise fft power

plt.plot(sample_freq[:401], abs(trace_noise_50_sig_10_fft)[:401], label='trace_noise_50_sig_10_fft')

title = "Synthetic 50hz noise + 10hz signal frequency domain"
plt.ylabel('Power')
plt.xlabel('Frequency [hz]')
plt.title(title)
plt.legend()
plt.savefig(path_save + title + '.png')
plt.show()
plt.cla()
plt.close()
# %% plot the signal with noise fft power

plt.plot(sample_freq[:401], abs(trace_noise_50_sig_15_fft)[:401], label='trace_noise_50_sig_15_fft')

title = "Synthetic 20hz noise + 15hz signal frequency domain"
plt.ylabel('Power')
plt.xlabel('Frequency [hz]')
plt.title(title)
plt.legend()
plt.savefig(path_save + title + '.png')
plt.show()
plt.cla()
plt.close()
# %% prepare 9 signal and corresponding ground truth
import h5py
len_trace = 9
data_traces = [trace_noise_10_sig_5,
               trace_noise_20_sig_5,
               trace_noise_50_sig_5,
               trace_noise_10_sig_10,
               trace_noise_20_sig_10,
               trace_noise_50_sig_10,
               trace_noise_10_sig_15,
               trace_noise_20_sig_15,
               trace_noise_50_sig_15]
gt = [trace_sig_5,
      trace_sig_5,
      trace_sig_5,
      trace_sig_10,
      trace_sig_10,
      trace_sig_10,
      trace_sig_15,
      trace_sig_15,
      trace_sig_15]
# %% notch filter
from scipy import signal
from scipy.signal import filtfilt

fs = 1/time_step
Q = 30.0
sig_filter = []
for i in range(len_trace):
    f0_notch = sample_freq[abs(fftpack.fft(data_traces[i])).argmax()]
    b, a = signal.iirnotch(f0_notch, Q, fs)
    sig_filter.append(filtfilt(b, a, data_traces[i]))

# %% notch filter time domain trace 1
plt.plot(time_vec, gt[0], label='5hz ground truth signal')
plt.plot(time_vec, sig_filter[0], label='notch filter remove 10hz harmonic noise')

title = "5hz signal notch filter remove 10hz noise"
plt.ylabel('Amplitude')
plt.xlabel('Time [s]')
plt.title(title)
plt.legend()
plt.savefig(path_save + title + '.png')
plt.show()
plt.cla()
plt.close()
# %% notch filter time domain trace 2
plt.plot(time_vec, gt[1], label='5hz ground truth signal')
plt.plot(time_vec, sig_filter[1], label='notch filter remove 20hz harmonic noise')

title = "5hz signal notch filter remove 20hz noise"
plt.ylabel('Amplitude')
plt.xlabel('Time [s]')
plt.title(title)
plt.legend()
plt.savefig(path_save + title + '.png')
plt.show()
plt.cla()
plt.close()
# %% notch filter time domain trace 3
plt.plot(time_vec, gt[2], label='5hz ground truth signal')
plt.plot(time_vec, sig_filter[2], label='notch filter remove 50hz harmonic noise')

title = "5hz signal notch filter remove 50hz noise"
plt.ylabel('Amplitude')
plt.xlabel('Time [s]')
plt.title(title)
plt.legend()
plt.savefig(path_save + title + '.png')
plt.show()
plt.cla()
plt.close()
# %% notch filter time domain trace 4
plt.plot(time_vec, gt[3], label='10hz ground truth signal')
plt.plot(time_vec, sig_filter[3], label='notch filter remove 10hz harmonic noise')

title = "10hz signal notch filter remove 10hz noise"
plt.ylabel('Amplitude')
plt.xlabel('Time [s]')
plt.title(title)
plt.legend()
plt.savefig(path_save + title + '.png')
plt.show()
plt.cla()
plt.close()
# %% notch filter time domain trace 5
plt.plot(time_vec, gt[4], label='10hz ground truth signal')
plt.plot(time_vec, sig_filter[4], label='notch filter remove 20hz harmonic noise')

title = "10hz signal notch filter remove 20hz noise"
plt.ylabel('Amplitude')
plt.xlabel('Time [s]')
plt.title(title)
plt.legend()
plt.savefig(path_save + title + '.png')
plt.show()
plt.cla()
plt.close()
# %% notch filter time domain trace 6
plt.plot(time_vec, gt[5], label='10hz ground truth signal')
plt.plot(time_vec, sig_filter[5], label='notch filter remove 50hz harmonic noise')

title = "10hz signal notch filter remove 50hz noise"
plt.ylabel('Amplitude')
plt.xlabel('Time [s]')
plt.title(title)
plt.legend()
plt.savefig(path_save + title + '.png')
plt.show()
plt.cla()
plt.close()
# %% notch filter time domain trace 7
plt.plot(time_vec, gt[6], label='15hz ground truth signal')
plt.plot(time_vec, sig_filter[6], label='notch filter remove 10hz harmonic noise')

title = "15hz signal notch filter remove 10hz noise"
plt.ylabel('Amplitude')
plt.xlabel('Time [s]')
plt.title(title)
plt.legend()
plt.savefig(path_save + title + '.png')
plt.show()
plt.cla()
plt.close()
# %% notch filter time domain trace 8
plt.plot(time_vec, gt[7], label='15hz ground truth signal')
plt.plot(time_vec, sig_filter[7], label='notch filter remove 20hz harmonic noise')

title = "15hz signal notch filter remove 20hz noise"
plt.ylabel('Amplitude')
plt.xlabel('Time [s]')
plt.title(title)
plt.legend()
plt.savefig(path_save + title + '.png')
plt.show()
plt.cla()
plt.close()
# %% notch filter time domain trace 9
plt.plot(time_vec, gt[8], label='15hz ground truth signal')
plt.plot(time_vec, sig_filter[8], label='notch filter remove 50hz harmonic noise')

title = "15hz signal notch filter remove 50hz noise"
plt.ylabel('Amplitude')
plt.xlabel('Time [s]')
plt.title(title)
plt.legend()
plt.savefig(path_save + title + '.png')
plt.show()
plt.cla()
plt.close()
# %% notch filter fft

sig_filter_fft = []
gt_fft = []
for i in range(len_trace):
    sig_filter_fft.append(fftpack.fft(sig_filter[i]))
    gt_fft.append(fftpack.fft(gt[i]))

# %% notch filter frequency domain trace 1
plt.plot(sample_freq[:401], abs(trace_noise_10_sig_5_fft)[:401], label='5hz ground truth signal + 10hz harmonic noise')
plt.plot(sample_freq[:401], abs(sig_filter_fft[0])[:401], label='notch filter')
plt.plot(sample_freq[:401], abs(gt_fft[0])[:401], label='ground truth')


title = "5hz signal notch filter remove 10hz noise power"
plt.ylabel('Amplitude')
plt.xlabel('Frequency [hz]')
plt.title(title)
plt.legend()
plt.savefig(path_save + title + '.png')
plt.show()
plt.cla()
plt.close()
# %% notch filter frequency domain trace 2
plt.plot(sample_freq[:401], abs(trace_noise_20_sig_5_fft)[:401], label='5hz ground truth signal + 20hz harmonic noise')
plt.plot(sample_freq[:401], abs(sig_filter_fft[1])[:401], label='notch filter')
plt.plot(sample_freq[:401], abs(gt_fft[1])[:401], label='ground truth')

title = "5hz signal notch filter remove 20hz noise power"
plt.ylabel('Amplitude')
plt.xlabel('Frequency [hz]')
plt.title(title)
plt.legend()
plt.savefig(path_save + title + '.png')
plt.show()
plt.cla()
plt.close()
# %% notch filter frequency domain trace 3
plt.plot(sample_freq[:401], abs(trace_noise_50_sig_5_fft)[:401], label='5hz ground truth signal + 50hz harmonic noise')
plt.plot(sample_freq[:401], abs(sig_filter_fft[2])[:401], label='notch filter')
plt.plot(sample_freq[:401], abs(gt_fft[2])[:401], label='ground truth')

title = "5hz signal notch filter remove 50hz noise power"
plt.ylabel('Amplitude')
plt.xlabel('Frequency [hz]]')
plt.title(title)
plt.legend()
plt.savefig(path_save + title + '.png')
plt.show()
plt.cla()
plt.close()
# %% notch filter Frequency domain trace 4
plt.plot(sample_freq[:401], abs(trace_noise_10_sig_10_fft)[:401], label='10hz ground truth signal + 10hz harmonic noise')
plt.plot(sample_freq[:401], abs(sig_filter_fft[3])[:401], label='notch filter')
plt.plot(sample_freq[:401], abs(gt_fft[3])[:401], label='ground truth')


title = "10hz signal notch filter remove 10hz noise power"
plt.ylabel('Amplitude')
plt.xlabel('Frequency [hz]')
plt.title(title)
plt.legend()
plt.savefig(path_save + title + '.png')
plt.show()
plt.cla()
plt.close()
# %% notch filter Frequency domain trace 5
plt.plot(sample_freq[:401], abs(trace_noise_20_sig_10_fft)[:401], label='10hz ground truth signal + 20hz harmonic noise')
plt.plot(sample_freq[:401], abs(sig_filter_fft[4])[:401], label='notch filter')
plt.plot(sample_freq[:401], abs(gt_fft[4])[:401], label='ground truth')


title = "10hz signal notch filter remove 20hz noise power"
plt.ylabel('Amplitude')
plt.xlabel('Frequency [hz]')
plt.title(title)
plt.legend()
plt.savefig(path_save + title + '.png')
plt.show()
plt.cla()
plt.close()
# %% notch filter Frequency domain trace 6
plt.plot(sample_freq[:401], abs(trace_noise_50_sig_10_fft)[:401], label='10hz ground truth signal + 50hz harmonic noise')
plt.plot(sample_freq[:401], abs(sig_filter_fft[5])[:401], label='notch filter')
plt.plot(sample_freq[:401], abs(gt_fft[5])[:401], label='ground truth')


title = "10hz signal notch filter remove 50hz noise power"
plt.ylabel('Amplitude')
plt.xlabel('Frequency [hz]')
plt.title(title)
plt.legend()
plt.savefig(path_save + title + '.png')
plt.show()
plt.cla()
plt.close()
# %% notch filter Frequency domain trace 7
plt.plot(sample_freq[:401], abs(trace_noise_10_sig_15_fft)[:401], label='15hz ground truth signal + 10hz harmonic noise')
plt.plot(sample_freq[:401], abs(sig_filter_fft[6])[:401], label='notch filter')
plt.plot(sample_freq[:401], abs(gt_fft[6])[:401], label='ground truth')


title = "15hz signal notch filter remove 10hz noise power"
plt.ylabel('Amplitude')
plt.xlabel('Frequency [hz]')
plt.title(title)
plt.legend()
plt.savefig(path_save + title + '.png')
plt.show()
plt.cla()
plt.close()
# %% notch filter Frequency domain trace 8
plt.plot(sample_freq[:401], abs(trace_noise_20_sig_15_fft)[:401], label='15hz ground truth signal + 20hz harmonic noise')
plt.plot(sample_freq[:401], abs(sig_filter_fft[7])[:401], label='notch filter')
plt.plot(sample_freq[:401], abs(gt_fft[7])[:401], label='ground truth')


title = "15hz signal notch filter remove 20hz noise power"
plt.ylabel('Amplitude')
plt.xlabel('Frequency [hz]')
plt.title(title)
plt.legend()
plt.savefig(path_save + title + '.png')
plt.show()
plt.cla()
plt.close()
# %% notch filter Frequency domain trace 9
plt.plot(sample_freq[:401], abs(trace_noise_50_sig_15_fft)[:401], label='15hz ground truth signal + 50hz harmonic noise')
plt.plot(sample_freq[:401], abs(sig_filter_fft[8])[:401], label='notch filter')
plt.plot(sample_freq[:401], abs(gt_fft[8])[:401], label='ground truth')


title = "15hz signal notch filter remove 50hz noise power"
plt.ylabel('Amplitude')
plt.xlabel('Frequency [hz]')
plt.title(title)
plt.legend()
plt.savefig(path_save + title + '.png')
plt.show()
plt.cla()
plt.close()
# %% down sample signal after notch filter
import cv2
sig_filter_ds = []
for i in range(len_trace):
    sig_img = sig_filter[i].reshape(1, sig_filter[i].shape[0])
    sig_img = cv2.resize(sig_img, (int(sig_filter[i].shape[0] / 5), 1), cv2.INTER_LINEAR)
    sig_img = sig_img.reshape(int(sig_filter[i].shape[0] / 5))
    sig_filter_ds.append(sig_img)
# %%plot down sample extracted signal after notch filter
plt.plot(time_vec[:500], sig_filter_ds[0], label='5hz signal remove 10hz harmonic noise with notch filter')

title = "down sample 5hz signal notch filter remove 10hz noise"
plt.ylabel('Amplitude')
plt.xlabel('Time [s]')
plt.title(title)
plt.legend()
plt.savefig(path_save + title + '.png')
plt.show()
plt.cla()
plt.close()
# %%plot down sample extracted signal after notch filter
plt.plot(time_vec[:500], sig_filter_ds[1], label='5hz signal remove 20hz harmonic noise with notch filter')

title = "down sample 5hz signal notch filter remove 20hz noise"
plt.ylabel('Amplitude')
plt.xlabel('Time [s]')
plt.title(title)
plt.legend()
plt.savefig(path_save + title + '.png')
plt.show()
plt.cla()
plt.close()
# %%plot down sample extracted signal after notch filter
plt.plot(time_vec[:500], sig_filter_ds[2], label='5hz signal remove 50hz harmonic noise with notch filter')

title = "down sample 5hz signal notch filter remove 50hz noise"
plt.ylabel('Amplitude')
plt.xlabel('Time [s]')
plt.title(title)
plt.legend()
plt.savefig(path_save + title + '.png')
plt.show()
plt.cla()
plt.close()
# %%plot down sample extracted signal after notch filter
plt.plot(time_vec[:500], sig_filter_ds[3], label='10hz signal remove 10hz harmonic noise with notch filter')

title = "down sample 10hz signal notch filter remove 10hz noise"
plt.ylabel('Amplitude')
plt.xlabel('Time [s]')
plt.title(title)
plt.legend()
plt.savefig(path_save + title + '.png')
plt.show()
plt.cla()
plt.close()
# %%plot down sample extracted signal after notch filter
plt.plot(time_vec[:500], sig_filter_ds[4], label='10hz signal remove 20hz harmonic noise with notch filter')

title = "down sample 10hz signal notch filter remove 20hz noise"
plt.ylabel('Amplitude')
plt.xlabel('Time [s]')
plt.title(title)
plt.legend()
plt.savefig(path_save + title + '.png')
plt.show()
plt.cla()
plt.close()
# %%plot down sample extracted signal after notch filter
plt.plot(time_vec[:500], sig_filter_ds[5], label='10hz signal remove 50hz harmonic noise with notch filter')

title = "down sample 10hz signal notch filter remove 50hz noise"
plt.ylabel('Amplitude')
plt.xlabel('Time [s]')
plt.title(title)
plt.legend()
plt.savefig(path_save + title + '.png')
plt.show()
plt.cla()
plt.close()
# %%plot down sample extracted signal after notch filter
plt.plot(time_vec[:500], sig_filter_ds[6], label='15hz signal remove 10hz harmonic noise with notch filter')

title = "down sample 15hz signal notch filter remove 10hz noise"
plt.ylabel('Amplitude')
plt.xlabel('Time [s]')
plt.title(title)
plt.legend()
plt.savefig(path_save + title + '.png')
plt.show()
plt.cla()
plt.close()
# %%plot down sample extracted signal after notch filter
plt.plot(time_vec[:500], sig_filter_ds[7], label='15hz signal remove 20hz harmonic noise with notch filter')

title = "down sample 15hz signal notch filter remove 20hz noise"
plt.ylabel('Amplitude')
plt.xlabel('Time [s]')
plt.title(title)
plt.legend()
plt.savefig(path_save + title + '.png')
plt.show()
plt.cla()
# %%plot down sample extracted signal after notch filter
plt.plot(time_vec[:500], sig_filter_ds[8], label='15hz signal remove 50hz harmonic noise with notch filter')

title = "down sample 15hz signal notch filter remove 50hz noise"
plt.ylabel('Amplitude')
plt.xlabel('Time [s]')
plt.title(title)
plt.legend()
plt.savefig(path_save + title + '.png')
plt.show()
plt.cla()
plt.close()
# %% blur down sample signal after notch filter
import cv2
sig_filter_ds_blur = []
kernel_size = (35, 1)
sigma = 5
for i in range(len_trace):
    img = cv2.GaussianBlur(sig_filter_ds[i].reshape(1, sig_filter_ds[i].shape[0]), kernel_size, sigma);
    img = img.reshape(sig_filter_ds[i].shape[0])
    sig_filter_ds_blur.append(img)
# %%plot blur down sample extracted signal after notch filter
plt.plot(time_vec[:500], sig_filter_ds_blur[0], label='5hz signal remove 10hz harmonic noise with notch filter')

title = "blur down sample 5hz signal notch filter remove 10hz noise"
plt.ylabel('Amplitude')
plt.xlabel('Time [s]')
plt.title(title)
plt.legend()
plt.savefig(path_save + title + '.png')
plt.show()
plt.cla()
plt.close()
# %%plot blur down sample extracted signal after notch filter
plt.plot(time_vec[:500], sig_filter_ds_blur[1], label='5hz signal remove 20hz harmonic noise with notch filter')

title = "blur down sample 5hz signal notch filter remove 20hz noise"
plt.ylabel('Amplitude')
plt.xlabel('Time [s]')
plt.title(title)
plt.legend()
plt.savefig(path_save + title + '.png')
plt.show()
plt.cla()
plt.close()
# %%plot blur down sample extracted signal after notch filter
plt.plot(time_vec[:500], sig_filter_ds_blur[2], label='5hz signal remove 50hz harmonic noise with notch filter')

title = "blur down sample 5hz signal notch filter remove 50hz noise"
plt.ylabel('Amplitude')
plt.xlabel('Time [s]')
plt.title(title)
plt.legend()
plt.savefig(path_save + title + '.png')
plt.show()
plt.cla()
plt.close()
# %%plot blur down sample extracted signal after notch filter
plt.plot(time_vec[:500], sig_filter_ds_blur[3], label='10hz signal remove 10hz harmonic noise with notch filter')

title = "blur down sample 10hz signal notch filter remove 10hz noise"
plt.ylabel('Amplitude')
plt.xlabel('Time [s]')
plt.title(title)
plt.legend()
plt.savefig(path_save + title + '.png')
plt.show()
plt.cla()
plt.close()
# %%plot blur down sample extracted signal after notch filter
plt.plot(time_vec[:500], sig_filter_ds_blur[4], label='10hz signal remove 20hz harmonic noise with notch filter')

title = "blur down sample 10hz signal notch filter remove 20hz noise"
plt.ylabel('Amplitude')
plt.xlabel('Time [s]')
plt.title(title)
plt.legend()
plt.savefig(path_save + title + '.png')
plt.show()
plt.cla()
plt.close()
# %%plot blur down sample extracted signal after notch filter
plt.plot(time_vec[:500], sig_filter_ds_blur[5], label='10hz signal remove 50hz harmonic noise with notch filter')

title = "blur down sample 10hz signal notch filter remove 50hz noise"
plt.ylabel('Amplitude')
plt.xlabel('Time [s]')
plt.title(title)
plt.legend()
plt.savefig(path_save + title + '.png')
plt.show()
plt.cla()
plt.close()
# %%plot blur down sample extracted signal after notch filter
plt.plot(time_vec[:500], sig_filter_ds_blur[6], label='15hz signal remove 10hz harmonic noise with notch filter')

title = "blur down sample 15hz signal notch filter remove 10hz noise"
plt.ylabel('Amplitude')
plt.xlabel('Time [s]')
plt.title(title)
plt.legend()
plt.savefig(path_save + title + '.png')
plt.show()
plt.cla()
# %%plot blur down sample extracted signal after notch filter
plt.plot(time_vec[:500], sig_filter_ds_blur[7], label='15hz signal remove 20hz harmonic noise with notch filter')

title = "blur down sample 15hz signal notch filter remove 20hz noise"
plt.ylabel('Amplitude')
plt.xlabel('Time [s]')
plt.title(title)
plt.legend()
plt.savefig(path_save + title + '.png')
plt.show()
plt.cla()
plt.close()
# %%plot blur down sample extracted signal after notch filter
plt.plot(time_vec[:500], sig_filter_ds_blur[8], label='15hz signal remove 50hz harmonic noise with notch filter')

title = "blur down sample 15hz signal notch filter remove 50hz noise"
plt.ylabel('Amplitude')
plt.xlabel('Time [s]')
plt.title(title)
plt.legend()
plt.savefig(path_save + title + '.png')
plt.show()
plt.cla()
plt.close()
# %% dataset for training
# Calculate the X (all data)
X = []
# Convert dataset to format of (len_trace)
len_trace = 9
for i in range(len_trace):
    # extract trace
    sig_true = gt[i]# extract synthetic seismic signal as the ground truth of NN
    sig_input = data_traces[i]  # extract signal with harmonic noise as the input of NN
    sig_ds_blur = sig_filter_ds_blur[i]
    data_trace = np.concatenate((sig_true, sig_input, sig_ds_blur), axis=0)
    X.append(data_trace)

X = np.array(X)
# Write the dataset to HDF5 file

name_data = 'Self_Syn_harmonic_dataset_9'

f = h5py.File(path_save + name_data + '.h5', 'w')
f.create_dataset('X', data=X)
f.close()
# %% dataset for evaluation
# Calculate the X (all data)
X = []
# Convert dataset to format of (len_trace)
len_trace = 9
for i in range(len_trace):
    # extract trace
    sig_true = gt[i]# extract synthetic seismic signal as the ground truth of NN
    sig_input = data_traces[i]  # extract signal with harmonic noise as the input of NN
    sig_notch = sig_filter[i]
    data_trace = np.concatenate((sig_true, sig_input, sig_notch), axis=0)
    X.append(data_trace)

X = np.array(X)
# Write the dataset to HDF5 file

name_data = 'Self_Syn_harmonic_dataset_9_test'

f = h5py.File(path_save + name_data + '.h5', 'w')
f.create_dataset('X', data=X)
f.close()




