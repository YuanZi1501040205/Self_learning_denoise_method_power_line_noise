# %%
import numpy as np
import random
from Functions import load_dataset
from Functions import extract
# configure the dataset size to generate
path_save = "/home/yzi/research/Self_learning_denoise_method_power_line_noise/output/figures/"
path_train_dataset = "/home/yzi/research/AGT_FWI_2020/datasets_AGT_FWI/Time_hdf5/Time_MarineS_BP2004.h5"
x, x_freq, length = load_dataset(path_train_dataset)
data_traces = extract(x, length)
# %%
import h5py
f = h5py.File('/home/yzi/research/Self_learning_denoise_method_power_line_noise/output/experiments/20210211/bp_3_adaptive.h5')
sig1 = f['X'][:, 0][:2500]
sig2 = f['X'][:, 1][:2500]
sig3 = f['X'][:, 2][:2500]

sig1_adaptive_filter = f['X'][:, 0][2500:5000]
sig2_adaptive_filter = f['X'][:, 1][2500:5000]
sig3_adaptive_filter = f['X'][:, 2][2500:5000]

sig1_noise = f['X'][:, 0][-2500:]
sig2_noise = f['X'][:, 1][-2500:]
sig3_noise = f['X'][:, 2][-2500:]
# %%
time_step = 0.002  # sample 2 ms
fs = 1 / time_step
time_range = 5  # signal length 5s at the time domain
time_vec = np.arange(0, time_range, time_step)
# %%
num_trace = 3 * x.shape[1] + 93
signal_gt = data_traces[num_trace]
sig1 = signal_gt # BP shot 4 receiver 93
# %%
num_trace = 25 * x.shape[1] + 26
signal_gt = data_traces[num_trace]
sig2 = signal_gt #
# %%
num_trace = 48 * x.shape[1] + 50
signal_gt = data_traces[num_trace]
sig3 = signal_gt

# %% Plot the signal 1 ground truth
import matplotlib.pyplot as plt

plt.plot(time_vec, sig1[2500:], label = "signal")

title = "Time_Domain_Marine_BP_shot_" + str(4) + '_receiver_' + str(93)
plt.ylabel('Amplitude')
plt.xlabel('Time [s]')
plt.title(title)
plt.legend()
plt.savefig(path_save + title + '.png')
plt.show()
plt.cla()
plt.close()
# %% Plot the signal ground truth
import matplotlib.pyplot as plt

plt.plot(time_vec, sig2[2500:], label = "signal")

title = "Time_Domain_Marine_BP_shot_" + str(26) + '_receiver_' + str(20)
plt.ylabel('Amplitude')
plt.xlabel('Time [s]')
plt.title(title)
plt.legend()
plt.savefig(path_save + title + '.png')
plt.show()
plt.cla()
plt.close()
# %% Plot the signal ground truth
import numpy as np
import matplotlib.pyplot as plt

plt.plot(time_vec, sig3[2500:], label = "signal")

title = "Time_Domain_Marine_Marmousi_shot_" + str(49) + '_receiver_' + str(50)
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
import matplotlib.pyplot as plt
import numpy as np
name_traces = ['BP_4_93',
               'BP_24_46',
               'BP_52_10']

sigs = [sig1,
        sig2,
        sig3]

for i in range(3):
    trace_fft = fftpack.fft(sigs[i])
    # sample freq
    sample_freq = fftpack.fftfreq(trace_fft.size, d=time_step)
    # power
    trace_fft_power = np.abs(trace_fft)

    plt.plot(sample_freq[0:401], trace_fft_power[0:401])

    title = "Signal GT FFT Power " + name_traces[i]
    plt.ylabel('Power')
    plt.xlabel('Frequency [Hz]')
    plt.title(title)
    plt.savefig(path_save + title + '.png')
    plt.show()
    plt.cla()
    plt.close()

# %% PLot the cos harmonic noise
f20_low = 20.8
f20_high = 21.1

f50_low = 50.8
f50_high = 51

noise_amplitude = 2
noise_phase1 = np.pi/6
noise_phase = 0
drift_noise_20hz = noise_amplitude * np.cos(2*(f20_low + (f20_high - f20_low)*time_vec/5) * np.pi * time_vec + noise_phase)
drift_noise_50hz_low = (noise_amplitude/4) * np.cos(2*f50_low * np.pi * time_vec + noise_phase)
drift_noise_50hz_high = (noise_amplitude/4) * np.cos(2*f50_high * np.pi * time_vec + noise_phase)
drift_noise_50hz = drift_noise_50hz_low + drift_noise_50hz_high

# %%
plt.plot(time_vec, drift_noise_20hz, label='noise 21 hz')
# plt.plot(time_vec, drift_noise_20hz_low, label='20.5hz')

title = "Synthetic drift 20hz noise time domain"
plt.ylabel('Amplitude')
plt.xlabel('Time [s]')
plt.title(title)
plt.legend()
plt.savefig(path_save + title + '.png')
plt.show()
plt.cla()
plt.close()
# %%
plt.plot(time_vec[:201], drift_noise_50hz[:201], label='50hz noise')
# plt.plot(time_vec, drift_noise_20hz_low, label='20.5hz')

title = "Synthetic 50hz noise time domain"
plt.ylabel('Amplitude')
plt.xlabel('Time [s]')
plt.title(title)
plt.legend()
plt.savefig(path_save + title + '.png')
plt.show()
plt.cla()
plt.close()
# %% PLot the cos harmonic noise fft power

noise_20hz_fft_power = abs(fftpack.fft(drift_noise_20hz))
noise_50hz_fft_power = abs(fftpack.fft(drift_noise_50hz))



plt.plot(sample_freq[:401], noise_20hz_fft_power[:401], label='drift 20.7-21.1')
plt.plot(sample_freq[:401], noise_50hz_fft_power[:401], label='50.8/51hz')

title = "Synthetic power-line noise frequency domain power"
plt.ylabel('Amplitude')
plt.xlabel('Frequency [hz]')
plt.title(title)
plt.legend()
plt.savefig(path_save + title + '.png')
plt.show()
plt.cla()
plt.close()
# %% noise 20hz + 50hz
noise = drift_noise_50hz + drift_noise_20hz
# sig + noise time
sig_noise_trace = []
for i in range(3):
    sig_noise_trace.append(noise + sigs[i])
    plt.plot(time_vec, sig_noise_trace[i], label='sig + noise')
    title = "Synthetic noise + signal time domain " + name_traces[i]
    plt.ylabel('Amplitude')
    plt.xlabel('Time [s]')
    plt.title(title)
    plt.legend()
    plt.savefig(path_save + title + '.png')
    plt.show()
    plt.cla()
    plt.close()

# %% sig + noise fft
sig_noise_trace_fft = []
for i in range(3):
    sig_noise_trace_fft.append(fftpack.fft(sig_noise_trace[i]))
    plt.plot(sample_freq[:401], abs(sig_noise_trace_fft[i])[:401], label='sig + noise')
    title = "Synthetic noise + signal frequency domain " + name_traces[i]
    plt.ylabel('Amplitude')
    plt.xlabel('Frequency [hz]')
    plt.title(title)
    plt.legend()
    plt.savefig(path_save + title + '.png')
    plt.show()
    plt.cla()
    plt.close()


# %% plot notch filter
from scipy import signal
from scipy.signal import filtfilt
import matplotlib.pyplot as plt
import numpy as np
import copy
Q = 12 # 9
time_step = 0.002  # sample 2 ms
fs = 1 / time_step
f20_notch = 25 # 20.8 - 21.1 hz

title = "Notch Filter Frequency Response"
b, a = signal.iirnotch(f20_notch, Q, fs)

# Frequency response
w, h = signal.freqz(b, a)

# Generate frequency axis
freq = w*fs/(2*np.pi)
# Plot
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(freq, 20*np.log10(abs(h)), color='blue')

ax.set_title(title)
ax.set_ylabel("Amplitude (dB)", color='blue')
ax.set_xlim([0, 100])
ax.set_ylim([-20, 10])
ax.grid()
plt.xticks(np.arange(1,100,5))
ax.set_xlabel("Frequency (Hz)")
# plt.savefig(path_save + title + '.png')
plt.show()






# %%
f_est = [24.9985, 25.0028, 25.0023]
sigs = [sig1, sig2, sig3]
sig_noise_trace = [sig1_noise, sig2_noise, sig3_noise]
# %% notch filter
from scipy import signal
from scipy.signal import filtfilt

fs = 1/time_step
Q = 12
sig_notch_filter = []
sig_notch_filter_fft = []
for i in range(3):
    f20_notch = f_est[i] # 20.8 - 21.1 hz
    b, a = signal.iirnotch(f20_notch, Q, fs)
    sig_filter20 = filtfilt(b, a, sig_noise_trace[i])
    sig_notch_filter.append(sig_filter20)

    # plot time domain
    plt.plot(time_vec, sigs[i], label='ground truth')
    plt.plot(time_vec, sig_notch_filter[i], label='notch filter')


    title = "Notch filter result time domain " + name_traces[i]
    plt.ylabel('Amplitude')
    plt.xlabel('Time [s]')
    plt.title(title)
    plt.legend()
    plt.savefig(path_save + title + '.png')
    plt.show()
    plt.cla()
    plt.close()

    # plot fft domain
    sig_notch_filter_fft.append(fftpack.fft(sig_notch_filter[i]))
    plt.plot(sample_freq[:401], abs(sig_notch_filter_fft[i])[:401], label='notch filter')
    plt.plot(sample_freq[:401], abs(fftpack.fft(sigs[i]))[:401], label='ground truth')

    title = "Notch filter result frequency domain " + name_traces[i]
    plt.ylabel('Amplitude')
    plt.xlabel('Time [s]')
    plt.title(title)
    plt.legend()
    plt.savefig(path_save + title + '.png')
    plt.show()
    plt.cla()
    plt.close()
    print('mse: ', sum((sigs[i] - sig_notch_filter[i])**2))

# %% down sample signal after notch filter
import cv2
sig_filter_ds = []
for i in range(3):
    sig_img = sig_notch_filter[i].real.reshape(1, sig_notch_filter[i].shape[0])
    sig_img = cv2.resize(sig_img, (int(sig_notch_filter[i].shape[0] / 5), 1), cv2.INTER_LINEAR)
    sig_img = sig_img.reshape(int(sig_notch_filter[i].shape[0] / 5))
    sig_filter_ds.append(sig_img)
    # plot down sample extracted signal after notch filter
    plt.plot(time_vec[:500], sig_filter_ds[i], label='notch sig down sample')

    title = "down sample notched signal " + name_traces[i]
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
kernel_size = (5, 1)
sigma = 1
for i in range(3):
    img = cv2.GaussianBlur(sig_filter_ds[i].reshape(1, sig_filter_ds[i].shape[0]), kernel_size, sigma)
    img = img.reshape(sig_filter_ds[i].shape[0])
    sig_filter_ds_blur.append(img)
    # plot blur down sample extracted signal after notch filter
    plt.plot(time_vec[:500], sig_filter_ds_blur[i], label='down sample blur')
    title = "blur down sample notch filter signal " + name_traces[i]
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
len_trace = 3
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
import h5py
X = []
# Convert dataset to format of (len_trace)
len_trace = 3
for i in range(len_trace):
    # extract trace
    sig_true = sigs[i]# extract synthetic seismic signal as the ground truth of NN
    sig_input = sig_noise_trace[i]  # extract signal with harmonic noise as the input of NN
    sig_notch = sig_notch_filter[i]
    data_trace = np.concatenate((sig_true.real, sig_input.real, sig_notch.real), axis=0)
    X.append(data_trace)

X = np.array(X)
# Write the dataset to HDF5 file

name_data = 'drift_power_line_dataset_BP3_niu_year'

f = h5py.File(path_save + name_data + '.h5', 'w')
f.create_dataset('X', data=X)
f.close()
