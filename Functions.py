"""Functions.py: File to stores all self-defined functions for the prepossessing, training and testing"""

__author__ = "Yuan Zi"
__email__ = "yzi2@central.uh.edu"
__version__ = "3.0.0"


# Functions
import numpy as np


# Create dataset
def create_dataset(dataset, look_back=2):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)


# Normalization
def feature_normalize(data):
    if len(data.shape) == 1:
        normalized_data = np.zeros((data.shape[0]), dtype=float)
        for i in range(data.shape[0]):
            if (i + 1) % 2 == 1:
                amplitude = np.sqrt(data[i] ** 2 + data[i + 1] ** 2)
            normalized_data[i] = data[i] / amplitude
    elif len(data.shape) == 2:
        normalized_data = np.zeros((data.shape[0], data.shape[1]), dtype=float)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if (j + 1) % 2 == 1:
                    amplitude = np.sqrt(data[i][j] ** 2 + data[i][j + 1] ** 2)
                normalized_data[i][j] = data[i][j] / amplitude
    return normalized_data


def time_normalize(data):
    # data size is (shot*receiver, length of trace)
    normalized_data = np.zeros((data.shape[0], data.shape[1]), dtype=float)
    normalized_data_buffer = np.zeros((data.shape[0]), dtype=float)
    for i in range(data.shape[0]):
        trace_max = abs(data[i][:]).max()  # abs normalization
        normalized_data[i][:] = data[i][:] / trace_max
        # store the max used to normalize the trace by trace number.
        normalized_data_buffer[i] = trace_max
    return normalized_data, normalized_data_buffer


def mean_phase_abs_error(predictions, targets):
    """ !!!phase wrapping: Mean Absolute Error calculation for 2 1d numpy arrays
        inputs : two same size 1d numpy array, predicted results and the labels
        output: one float value indicate the mean square error"""
    error_buffer = [abs(predictions - targets), abs(predictions - targets + 2 * np.pi),
                    abs(predictions - targets - 2 * np.pi)]
    error_buffer = np.array(error_buffer)
    mean_abs_error = error_buffer.min(axis=0).mean()
    return mean_abs_error


def load_dataset(path):
    """ read the hdf5 dataset
            input:
            path of the hdf5 dataset like 'dataSets/BP2004-new (trset1).h5'
        Return:
            x: All the shots' and receivers' data the shape is (num_shot, num_receiver,
            length of trace)
            x_freq: frequency components in this dataset like (1-40) Hz
            len: For each shot, how many receiver they have; not every shot have the same number of receivers

            Because of the fourier transform, data is in the frequency domain, so there
            are two types of value at the trace: real and imaginary part just like (1Hz_Real,
            1Hz_Imaginary, 2Hz_Real, 2Hz_Imaginary, ..., 40Hz_Real, 40Hz_Imaginary). So the size of this trace is 2
            times than number of frequency points"""
    import h5py
    # Read the train dataset
    f = h5py.File(path, 'r')
    # list(f.keys())
    # print([key for key in f.keys()], "\n")
    x = f['X']
    x_freq = f['X_freq']
    length = f['len']
    return x, x_freq, length


def extract(x, length):
    # Extract shot and receiver pairs as traces shape = (num of traces, number of frequency * 2)
    data_traces = []
    for i in range(x.shape[0]):  # iterate all shots
        for j in range(length[i]):  # iterate all receivers related to the shot i
            data_traces.append(np.array(x[i][j][:]))
    return data_traces


# convert Re and Im to Phase
def tophase(data_re_im):
    """Feed in all traces, it would convert each trace and return all traces"""
    # If the input is a vector
    if len(data_re_im.shape) == 1:
        num_points = data_re_im.shape[0]
        data_phase = np.zeros((int(num_points / 2)), dtype=np.float32)  # half of the data_Re_Im
        if type(data_re_im) == torch.Tensor:
            data_re_im_array = data_re_im.detach().numpy()
        else:
            data_re_im_array = data_re_im
        for j in range(num_points):
            if (j + 1) % 2 == 1:
                data_phase[int(j / 2)] = np.arctan2(data_re_im_array[j + 1], data_re_im_array[j])
    else:
        num_trace = data_re_im.shape[0]
        num_points = data_re_im.shape[1]
        data_phase = np.zeros((num_trace, int(num_points / 2)), dtype=np.float32)  # half of the data_Re_Im
        for i in range(num_trace):
            # iteration all data point (Re Im) for this trace
            for j in range(num_points):
                if (j + 1) % 2 == 1:
                    data_phase[i][int(j / 2)] = np.arctan2(data_re_im[i][j + 1], data_re_im[i][j])
    return data_phase


def guassian_filter(signal, low_cut, high_cut):
    from scipy import fftpack
    sig_fft = fftpack.fft(signal)
    # gaussian filter
    time_step = 0.002  # sample 2 ms
    sample_freq = fftpack.fftfreq(sig_fft.size, d=time_step)
    mu = low_cut + (high_cut - low_cut) / 2
    sigma = (high_cut - low_cut) / 6  # set the cut off frequency is 3 sigma of the filter
    filter_fft_positive = np.exp(-pow(sample_freq[:int(sample_freq.shape[0] / 2)] - mu, 2) / (2 * pow(sigma, 2))) / (
            sigma * np.sqrt(2 * np.pi))
    filter_fft_negative = np.exp(-pow(sample_freq[int(sample_freq.shape[0] / 2):] + mu, 2) / (2 * pow(sigma, 2))) / (
            sigma * np.sqrt(2 * np.pi))
    filter_fft = np.concatenate((filter_fft_positive, filter_fft_negative), axis=0)
    sig_filtered_fft = sig_fft * filter_fft
    return fftpack.ifft(sig_filtered_fft)

from scipy.signal import butter, filtfilt

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_lowpass(highcut, fs, order=5):
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = butter(order, high, btype='low')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def butter_lowpass_filter(data, highcut, fs, order=5):
    b, a = butter_lowpass(highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def gain_AGC(shot_gather, rms, gate_height, gate_width):
    #mask = np.ones((shot_gather.shape[0], shot_gather.shape[1]))
    zeropool_height = np.zeros((shot_gather.shape[0],int((gate_height-1)/2)))
    zeropool_width = np.zeros((int((gate_width - 1) / 2), shot_gather.shape[1]))
    # sig_pooled = np.concatenate((zeropool_height, shot_gather), axis=1)
    # sig_pooled = np.concatenate((shot_gather, zeropool_height), axis=1)
    sig_pooled = np.concatenate((zeropool_width, shot_gather), axis=0)
    sig_pooled = np.concatenate((sig_pooled, zeropool_width), axis=0)
    mask = np.zeros((shot_gather.shape[0], shot_gather.shape[1] - int(gate_height-1)))
    for i in range(shot_gather.shape[0]):
        for j in range(shot_gather.shape[1] - int((gate_height-1))):
            mask[i][j] = sig_pooled[i: gate_width + i, j : gate_height+j].__abs__().mean()
    sig_after_gain = shot_gather[:, int((gate_height-1)/2):shot_gather.shape[1]-int((gate_height-1)/2)] * rms / (mask + 0.000001)
    sig_after_gain = np.concatenate((zeropool_height, sig_after_gain), axis=1)
    sig_after_gain = np.concatenate((sig_after_gain, zeropool_height), axis=1)
    return sig_after_gain
# --------------------------------------utils for Model---------------------------------------------
""" utils for the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(mid_channels),
            nn.PReLU(),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        return self.double_conv(x)


class ResDoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        if in_channels != out_channels:
            self.same_shape = False
        else:
            self.same_shape = True

        self.res_double_conv = nn.Sequential(
            my_residual_block(in_channels, out_channels, same_shape=self.same_shape),
            my_residual_block(out_channels, out_channels)

        )

    def forward(self, x):
        return self.res_double_conv(x)

class ResDown(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            ResDoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = x1.unsqueeze(0)
        x2 = x2.unsqueeze(0)
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x1 = x1.squeeze(0)
        x2 = x2.squeeze(0)
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


""" utils for the ResNet model """


def conv1x3(in_channel, out_channel, stride=1):
    return nn.Conv1d(in_channel, out_channel, 3, stride=stride, padding=1, bias=False)


class residual_block(nn.Module):
    def __init__(self, in_channel, out_channel, same_shape=True):
        super(residual_block, self).__init__()
        self.same_shape = same_shape
        stride = 1 if self.same_shape else 2

        self.conv1 = conv1x3(in_channel, out_channel, stride=stride)
        self.bn1 = nn.BatchNorm1d(out_channel)

        self.conv2 = conv1x3(out_channel, out_channel)
        self.bn2 = nn.BatchNorm1d(out_channel)
        if not self.same_shape:
            self.conv3 = nn.Conv1d(in_channel, out_channel, 1, stride=stride)

    def forward(self, x):
        out = self.conv1(x)
        out = F.leaky_relu_(self.bn1(out))
        out = self.conv2(out)
        out = F.leaky_relu_(self.bn2(out))

        if not self.same_shape:
            x = self.conv3(x)
        return F.leaky_relu_(x + out)


class my_residual_block(nn.Module):
    def __init__(self, in_channel, out_channel, same_shape=True):
        super(my_residual_block, self).__init__()
        self.same_shape = same_shape
        stride = 1

        self.conv1 = conv1x3(in_channel, out_channel, stride=stride)
        self.bn1 = nn.BatchNorm1d(out_channel)

        self.conv2 = conv1x3(out_channel, out_channel)
        self.bn2 = nn.BatchNorm1d(out_channel)
        if not self.same_shape:
            self.conv3 = nn.Conv1d(in_channel, out_channel, 1, stride=stride)

    def forward(self, x):
        out = self.conv1(x)
        out = F.leaky_relu_(self.bn1(out), True)
        out = self.conv2(out)
        out = F.leaky_relu_(self.bn2(out), True)

        if not self.same_shape:
            x = self.conv3(x)
        return F.leaky_relu_(x + out)


""" utils for the vgg model """


def vgg_block(num_convs, in_channels, out_channels):
    net = [nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1), nn.ReLU(True)]  #

    for i in range(num_convs - 1):  #
        net.append(nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1))
        net.append(nn.Dropout(0.5))  # !!! add dropout
        net.append(nn.ReLU(True))

    net.append(nn.MaxPool1d(2))
    return nn.Sequential(*net)


def vgg_stack(num_convs, channels):
    net = []
    for n, c in zip(num_convs, channels):
        in_c = c[0]
        out_c = c[1]
        net.append(vgg_block(n, in_c, out_c))
    return nn.Sequential(*net)


""" utils for the CBAM Attention model """


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x) * residual


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Conv1d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv1d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out) * residual


class AttentionVisualization:
    """    Example use:
    # %%
    from Functions import AttentionVisualization
    name_model = 'CNN12_AttenResNet'
    model_path = '/homelocal/AGT_FWI_2020/output/models/CNN12_AttenResNet_MarineS_Marmousi_Train_state_dict.pt'
    shot = 4
    receiver = 135
    input_path = '/homelocal/AGT_FWI_2020/MLFWI_data-20200528T194322Z-001/shift_F_hdf5/MarineS_Marmousi_Test.h5'
    output_path = '/homelocal/AGT_FWI_2020/output/'
    myClass = AttentionVisualization(name_model, model_path, shot, receiver, input_path, output_path)
    myClass.save_attentionmap_to_img()
    # %%
    """

    def __init__(self, name_model, model_path, shot, receiver, input_path, output_path):
        import torch
        from models_zoo import models
        self.input_path = input_path
        self.output_path = output_path
        self.shot = shot
        self.receiver = receiver
        self.name_model = name_model
        self.pretrained_model, self.loss_func, self.optimizer = models(name_model)
        self.pretrained_model.load_state_dict(torch.load(model_path))
        self.pretrained_model.eval()

    def process_input(self):
        x, x_freq, length = load_dataset(self.input_path)
        data_traces = extract(x, length)
        num_trace = self.shot * x.shape[1] + self.receiver
        data_test = data_traces[num_trace]
        data_test = feature_normalize(np.array(data_test))
        data_test = torch.tensor(data_test).type('torch.FloatTensor')
        # set the input and the output size manually for every model choosed from models.py file
        if 'ResNet' in self.name_model:
            num_input = 300
            num_output = 92
        elif 'UNet' in self.name_model:
            num_input = 2500
            num_output = 2500
        test_input = data_test[num_output: num_output + num_input]

        test_input = test_input.unsqueeze(0).unsqueeze(0)
        return test_input

    def get_attentionmap(self):
        import torch.nn.functional as F
        # input = torch.randn(1, 1, 300)
        test_input = self.process_input()
        x = test_input
        if 'UNet' in self.name_model:
            # forward UNet
            x1 = self.pretrained_model.inc(x)
            x2 = self.pretrained_model.down1(x1)
            x3 = self.pretrained_model.down2(x2)
            x4 = self.pretrained_model.down3(x3)
            x5 = self.pretrained_model.down4(x4)
            x6 = self.pretrained_model.down5(x5)
            x7 = self.pretrained_model.down6(x6)
            x = self.pretrained_model.up1(x7, x6)
            x = self.pretrained_model.up2(x, x5)
            x = self.pretrained_model.up3(x, x4)
            x = self.pretrained_model.up4(x, x3)
            x = self.pretrained_model.up5(x, x2)
            x = self.pretrained_model.up6(x, x1)
            x = self.pretrained_model.outc(x)
        elif 'ResNet' in self.name_model:
            # forward ResNet
            x = self.pretrained_model.block1(x)
            x = self.pretrained_model.block2(x)
            x = self.pretrained_model.block3(x)
            x = self.pretrained_model.block4(x)
            x = self.pretrained_model.block5(x)
            x = self.pretrained_model.block6(x)
            x = self.pretrained_model.block7(x)
            x = self.pretrained_model.block8(x)
            x = self.pretrained_model.block9(x)
        # featuremap: torch.size([1, 1024, 300]) for ResNetF [1,2500,2500] for UNetT
        featuremap = x
        # featuremap: torch.size([1024, 300])
        featuremap = featuremap.squeeze(0)
        # output.weight size is([92,1024])
        # attentionmap size is ([92,300])
        attention_map = torch.mm(self.pretrained_model.output.weight, featuremap)
        # relu eliminated negative value
        attention_map = F.relu(attention_map)
        # Global Average Pooling: torch.size([1, 1024, 1])
        # x = self.pretrained_model.GAP(x)
        # # Reshape: torch.size([1, 1024])
        # x = x.view(x.shape[0], -1)
        # x = self.pretrained_model.output(x)
        return attention_map

    def save_attentionmap_to_img(self):
        import cv2
        # to numpy [92,300]
        attention_map = self.get_attentionmap()
        attention_map = attention_map.detach().numpy()

        # row normalization to [0,1] value
        row_norm_term = attention_map.max(axis=1)
        attention_map = attention_map / row_norm_term[:, np.newaxis]
        # to [0,255]
        stack_attention_map = attention_map.sum(axis=0)
        stack_attention_map = stack_attention_map / stack_attention_map.max()
        attention_map = np.uint8(attention_map * 255)
        stack_attention_map = np.uint8(stack_attention_map * 255)
        attention_map = cv2.applyColorMap(attention_map, cv2.COLORMAP_JET)
        stack_attention_map = cv2.applyColorMap(stack_attention_map, cv2.COLORMAP_JET)
        stack_attention_map = cv2.transpose(stack_attention_map)
        visual_stack_attention_map = []
        for i in range(400):
            visual_stack_attention_map.append(stack_attention_map.squeeze(0))
        visual_stack_attention_map = np.array(visual_stack_attention_map)
        cv2.imwrite(
            self.output_path + 'figures/AttentionMapShot' + str(self.shot) + 'Receiver' + str(self.receiver) + '.jpg',
            attention_map)
        cv2.imwrite(self.output_path + 'figures/Stack_AttentionMapShot' + str(self.shot) + 'Receiver' + str(
            self.receiver) + '.jpg',
                    visual_stack_attention_map)


# ------------------------loss zoo--------------------------------------------------------------------------
class GaussianBlurConv(nn.Module):
    def __init__(self, channels=1):
        # input kernel should be shape:[1,size of kernal]
        super(GaussianBlurConv, self).__init__()
        self.channels = channels
        kernel = np.array([gauss_kernel])
        kernel_size = kernel.shape[1]
        kernel = torch.FloatTensor(kernel).unsqueeze(0)
        kernel = np.repeat(kernel, self.channels, axis=0)
        self.weight = nn.Parameter(data=kernel, requires_grad=True)
        self.padding = int(kernel_size/2)
    def __call__(self, x):
        x = F.conv1d(x, self.weight, padding=self.padding, stride=1, groups=self.channels)
        return x

def my_powerline_loss1(pre_sig, pre_noise, sig_label, input, parameters):
    # parameters are the hyperparameters:
    #     alpha = 100
    #     beta = 1
    #     lamda = 1
    #     gamma = 1

    """ All input is 3 dimensions (batch,channel,signal)
    loss 1 calculated 2 parts of loss and 2 regularizer:
loss1.MSE loss of (original trace, predicted noise + predicted signal)
loss2. Wasserstein loss of (notch filter->down sample->gaussian blur signal, predicted signal-down sample- gaussian blur)
regularizer1.L2 norm on signal time domain
regularizer2.L1 norm on the noise frequency domain"""

    import torch
    import torch.nn as nn
    import cv2
    import torch.nn.functional as F
    from scipy import fftpack
    import numpy as np
    from pytorch_stats_loss import torch_wasserstein_loss

    # parameters
    time_step = 0.002  # sample 2 ms
    fs = 1 / time_step
    f_max = int(fs/2)
    time_range = 5  # signal length 5s at the time domain
    time_vec = np.arange(0, time_range, time_step)
    trace_fft = fftpack.fft(time_vec)

    # sample freq
    sample_freq = fftpack.fftfreq(trace_fft.size, d=time_step)

    # Loss 0: signal prediction fft should be the same as the notch filter signal subtracted near suspicious power line
    # noise frequency spectrum: whole frequency spectrum is 0-500hz, the suspicious noise frequency is 5 hz.
    # Then loss0 = MSE(predicted signal fft(0-4Hz 6-500Hz), notch filter signal fft(0-4Hz 6-500Hz))
    MSEloss = nn.MSELoss()
    pre_sig_reshape = pre_sig.squeeze(0).squeeze(0)
    pre_sig_reshape = pre_sig.reshape(int(pre_sig_reshape.__len__() / 2), 2)
    pre_sig_fft = torch.fft(pre_sig_reshape, 1)

    sig_label_reshape = sig_label.squeeze(0).squeeze(0)
    sig_label_reshape = sig_label_reshape.reshape(int(sig_label_reshape.__len__() / 2), 2)
    sig_label_fft = torch.fft(sig_label_reshape, 1)

    sig_label_fft_power = torch.sqrt(sig_label_fft[:, 0] ** 2 + sig_label_fft[:, 1] ** 2)
    # sig_label_fft_power = sig_label_fft_power.cpu().detach().numpy()
    f_suspicious_noise = 21
    suspicious_radium = parameters[4]
    f_suspicious_noise_l_index = np.where(sample_freq == f_suspicious_noise - suspicious_radium)[0][0]
    f_suspicious_noise_r_index = np.where(sample_freq == f_suspicious_noise + suspicious_radium)[0][0]

    f_suspicious_noise_2 = 51
    suspicious_radium = parameters[4]
    f_2_suspicious_noise_l_index = np.where(sample_freq == f_suspicious_noise_2 - suspicious_radium)[0][0]
    f_2_suspicious_noise_r_index = np.where(sample_freq == f_suspicious_noise_2 + suspicious_radium)[0][0]

    for idx in range(f_suspicious_noise_l_index, f_suspicious_noise_r_index + 1):
        pre_sig_fft = pre_sig_fft[torch.arange(pre_sig_fft.size(0)) != idx]
        sig_label_fft = sig_label_fft[torch.arange(sig_label_fft.size(0)) != idx]

    for idx in range(f_2_suspicious_noise_l_index, f_2_suspicious_noise_r_index + 1):
        pre_sig_fft = pre_sig_fft[torch.arange(pre_sig_fft.size(0)) != idx]
        sig_label_fft = sig_label_fft[torch.arange(sig_label_fft.size(0)) != idx]

    Loss_0 = MSEloss(pre_sig_fft, sig_label_fft)


    # Loss 1: signal prediction + noise prediction = original input(real signal + real noise)

    Loss_1 = MSEloss(input, pre_sig + pre_noise)

    # Loss 2: Wasserstein loss of (notch filter->down sample->gaussian blur signal, predicted signal-down sample-
    # gaussian blur)

    # downsample
    d_sig_label = sig_label.unsqueeze(0)
    d_pre_sig = pre_sig.unsqueeze(0)
    d_shape = int(500) # size of signal after downsample, here is 500
    d_sig_label = torch.nn.functional.interpolate(d_sig_label, size=(1, d_shape), mode='bilinear')
    d_pre_sig = torch.nn.functional.interpolate(d_pre_sig, size=(1, d_shape), mode='bilinear')
    d_sig_label = d_sig_label.squeeze(0)
    d_pre_sig = d_pre_sig.squeeze(0)

    # gaussian blur
    sigma = 13 # 15
    kernel_size = 5 # 55
    cv2_kernel = cv2.getGaussianKernel(kernel_size, sigma)
    cv2_kernel = cv2_kernel.squeeze(1)
    cv2_kernel = torch.from_numpy(cv2_kernel).unsqueeze(0).unsqueeze(0).type(torch.FloatTensor).to(device=pre_sig.device)
    padding = int(kernel_size/2)
    channels = 1
    d_pre_sig = d_pre_sig.to(device=pre_sig.device)
    d_sig_label = d_sig_label.to(device=pre_sig.device)
    db_pre_sig = F.conv1d(d_pre_sig, cv2_kernel, padding=padding, stride=1, groups=channels)
    db_sig_label = F.conv1d(d_sig_label, cv2_kernel, padding=padding, stride=1, groups=channels)
    Loss_2 = MSEloss(db_sig_label, db_pre_sig) # try MSE first

    # db_sig_label_distribution = db_sig_label.squeeze(0).squeeze(0)/torch.sum(db_sig_label.squeeze(0).squeeze(0))
    # db_pre_sig_distribution = db_pre_sig.squeeze(0).squeeze(0) / torch.sum(db_pre_sig.squeeze(0).squeeze(0))
    # Loss_2 = torch_wasserstein_loss(db_sig_label_distribution, db_pre_sig_distribution)  # or use the Wasserstein loss

    # regularizer 1: L2 norm on signal time domain
    pre_sig = pre_sig.squeeze(0).squeeze(0)
    R_sig_L2 = (pre_sig**2).mean()

    # regularizer 2: L1 norm on the noise frequency domain
    pre_noise = pre_noise.squeeze(0).squeeze(0)
    pre_noise_reshape = pre_noise.reshape(int(pre_noise.__len__() / 2), 2)
    pre_noise_fft = torch.fft(pre_noise_reshape, 1)
    R_noise_L1 = torch.sqrt(pre_noise_fft[:, 0]**2 + pre_noise_fft[:, 1]**2).mean()


    # alpha = 1000
    # beta = 1
    # lamda = 100
    # gamma = 10

    alpha = parameters[0]
    beta = parameters[1]
    lamda = parameters[2]
    gamma = parameters[3]
    notch_weight = parameters[5]


    # print('alpha * Loss_1: ', alpha * Loss_1)
    # print('beta * Loss_2: ', beta * Loss_2)
    # print('lamda * R_sig_L2: ', lamda * R_sig_L2)
    # print('gamma * R_noise_L1: ', gamma * R_noise_L1)
    loss = notch_weight * Loss_0 + alpha * Loss_1 + beta * Loss_2 + lamda * R_sig_L2 + gamma * R_noise_L1
    return loss

def my_loss_f1(y_pred, y):
    """loss one to calculated 10-40 frequency MSE and L1 norm on time domain"""

    import torch
    import torch.nn as nn
    # cnn
    y_pred = y_pred.squeeze(0).squeeze(0)
    y = y.squeeze(0).squeeze(0)
    # #FNN
    # y_pred = y_pred.squeeze(0)
    # y = y.squeeze(0)

    y_pred_reshape = y_pred.reshape(int(y_pred.__len__() / 2), 2)
    y_reshape = y.reshape(int(y.__len__() / 2), 2)
    # calculate the 10-30 Hz
    y_pred_fft = torch.fft(y_pred_reshape, 1)[50:251]
    y_fft = torch.fft(y_reshape, 1)[50:251]

    lamda = 0.00005
    abs_max = torch.abs(y_fft).max()
    # abs_max_pred = torch.abs(y_pred_fft).max()
    # max = y_fft.max()
    # var = y_fft.var()
    # mean = y_fft.mean()
    # min = y_fft.min()
    y_pred_fft_norm = y_pred_fft/abs_max
    y_fft_norm = y_fft/abs_max
    MSEloss = nn.MSELoss()
    y_pred_norm = y_pred / torch.abs(y_pred).max()
    # print(' MSEloss(y_pred_fft, y_fft)',  MSEloss(y_pred_fft_norm, y_fft_norm))
    # print(' torch.abs(y_pred).mean()', torch.abs(y_pred_norm).mean())
    loss = MSEloss(y_pred_fft_norm, y_fft_norm) + lamda * torch.abs(y_pred_norm).mean()
    return loss


def my_loss_c2(y_pred, y):
    '''loss one to mitigate the phase wrapping'''

    import torch
    import numpy as np
    import torch.nn as nn
    loss_func_complex = nn.MSELoss()
    y_pred = y_pred.squeeze(0).squeeze(0)
    y = y.squeeze(0).squeeze(0)
    y_pred_phase = tophase(y_pred)
    y_phase = tophase(y)
    phase_mse1 = torch.from_numpy((y_pred_phase - y_phase) ** 2)
    phase_mse2 = torch.from_numpy((y_pred_phase - y_phase + 2 * np.pi) ** 2)
    phase_mse3 = torch.from_numpy((y_pred_phase - y_phase - 2 * np.pi) ** 2)
    phase_tensor = torch.cat((phase_mse1, phase_mse2, phase_mse3))
    phase_tensor = phase_tensor.reshape(3, y_pred_phase.shape[0])
    min_phase_mse = phase_tensor.min(dim=0)
    loss_phase = torch.mean(min_phase_mse.values)

    loss = 0.8 * loss_func_complex(y_pred, y) + 0.2 * loss_phase

    return loss


def my_loss_c3(y_pred, y):
    '''loss one to mitigate the phase wrapping'''

    import torch
    import numpy as np
    import torch.nn as nn
    loss_func_complex = nn.MSELoss()
    y_pred = y_pred.squeeze(0).squeeze(0)
    y = y.squeeze(0).squeeze(0)
    y_pred_phase = tophase(y_pred)
    y_phase = tophase(y)
    phase_mse1 = torch.from_numpy((y_pred_phase - y_phase) ** 2)
    phase_mse2 = torch.from_numpy((y_pred_phase - y_phase + 2 * np.pi) ** 2)
    phase_mse3 = torch.from_numpy((y_pred_phase - y_phase - 2 * np.pi) ** 2)
    phase_tensor = torch.cat((phase_mse1, phase_mse2, phase_mse3))
    phase_tensor = phase_tensor.reshape(3, y_pred_phase.shape[0])
    min_phase_mse = phase_tensor.min(dim=0)
    loss_phase = torch.mean(min_phase_mse.values)
    if len(y_pred.shape) == 3:
        y_pred_diff = y_pred[0][0][1:] - y_pred[0][0][:-1]
        y_diff = y[0][0][1:] - y[0][0][:-1]
    # FNN only have one dimension output
    else:
        y_pred_diff = y_pred[1:] - y_pred[:-1]
        y_diff = y[1:] - y[:-1]
    r_term = torch.mean((y_pred_diff - y_diff) ** 2)

    loss = 0.8 * loss_func_complex(y_pred, y) + 0.1 * loss_phase + 0.1 * r_term

    return loss


def my_loss_c4(y_pred, y):
    '''loss one to mitigate the phase wrapping'''

    import torch
    import numpy as np
    import torch.nn as nn
    loss_func_complex = nn.MSELoss()
    # CNN have 3 dimension output
    if len(y_pred.shape) == 3:
        y_pred_diff = y_pred[0][0][1:] - y_pred[0][0][:-1]
        y_diff = y[0][0][1:] - y[0][0][:-1]
    # FNN only have one dimension output
    else:
        y_pred_diff = y_pred[1:] - y_pred[:-1]
        y_diff = y[1:] - y[:-1]
    r_term = torch.mean((y_pred_diff - y_diff) ** 2)

    loss = 0.2 * loss_func_complex(y_pred, y) + 0.8 * r_term

    return loss


def my_loss_c5(y_pred, y):
    '''loss one to mitigate the phase wrapping'''

    import torch
    import numpy as np
    import torch.nn as nn
    # loss_func_complex = nn.MSELoss()
    # CNN have 3 dimension output
    if len(y_pred.shape) == 3:
        y_pred_diff = y_pred[0][0][1:] - y_pred[0][0][:-1]
        y_diff = y[0][0][1:] - y[0][0][:-1]
    # FNN only have one dimension output
    else:
        y_pred_diff = y_pred[1:] - y_pred[:-1]
        y_diff = y[1:] - y[:-1]
    r_term = torch.mean((y_pred_diff - y_diff) ** 2)

    loss = r_term

    return loss


def my_loss_c6(y_pred, y):
    '''loss one to mitigate the phase wrapping'''

    import torch
    import numpy as np
    import torch.nn as nn
    loss_func_complex = nn.MSELoss()
    # CNN have 3 dimension output
    if len(y_pred.shape) == 3:
        y_pred_diff = y_pred[0][0][1:] - y_pred[0][0][:-1]
        y_diff = y[0][0][1:] - y[0][0][:-1]
    # FNN only have one dimension output
    else:
        y_pred_diff = y_pred[1:] - y_pred[:-1]
        y_diff = y[1:] - y[:-1]
    r_term = torch.mean((y_pred_diff - y_diff) ** 2)

    loss = 0.8 * loss_func_complex(y_pred, y) + 0.2 * r_term
    print('MSE/Diff_term_loss: ', loss_func_complex(y_pred, y) / r_term)

    return loss


def my_loss_c7(y_pred, y):
    '''loss one to mitigate the phase wrapping'''

    import torch
    import numpy as np
    import torch.nn as nn
    loss_func_complex = nn.MSELoss()
    # CNN have 3 dimension output
    if len(y_pred.shape) == 3:
        y_pred_diff = y_pred[0][0][1:] - y_pred[0][0][:-1]
        y_diff = y[0][0][1:] - y[0][0][:-1]
    # FNN only have one dimension output
    else:
        y_pred_diff = y_pred[1:] - y_pred[:-1]
        y_diff = y[1:] - y[:-1]
    r_term = torch.mean((y_pred_diff - y_diff) ** 2)

    loss = 0.5 * loss_func_complex(y_pred, y) + 0.5 * r_term
    print('MSE/Diff_term_loss: ', loss_func_complex(y_pred, y) / r_term)

    return loss


# ------------------------loss for phase mode----------------------------------------------------------------------------
def my_loss_p1(y_pred, y):
    '''loss one to mitigate the phase wrapping'''

    import torch
    import numpy as np
    phase_mse1 = (y_pred - y) ** 2
    phase_mse2 = (y_pred - y + 2 * np.pi) ** 2
    phase_mse3 = (y_pred - y - 2 * np.pi) ** 2
    phase_tensor = torch.cat((phase_mse1, phase_mse2, phase_mse3))
    phase_tensor = phase_tensor.reshape(3, 46)  # !!! the output size
    min_phase_mse = phase_tensor.min(dim=0)
    loss = torch.mean(min_phase_mse.values)

    return loss


def my_loss_p2(y_pred, y):
    '''loss two to solve the phase wrapping, add the phase derivative regularization term'''

    import torch
    import numpy as np
    phase_mse1 = (y_pred - y) ** 2
    phase_mse2 = (y_pred - y + 2 * np.pi) ** 2
    phase_mse3 = (y_pred - y - 2 * np.pi) ** 2
    phase_tensor = torch.cat((phase_mse1, phase_mse2, phase_mse3))
    phase_tensor = phase_tensor.reshape(3, 46)  # !!! the output size
    min_phase_mse = phase_tensor.min(dim=0)
    loss = torch.mean(min_phase_mse.values)
    # CNN have 3 dimension output
    if len(y_pred.shape) == 3:
        y_pred_diff = y_pred[0][0][1:] - y_pred[0][0][:-1]
        y_diff = y[0][0][1:] - y[0][0][:-1]
    # FNN only have one dimension output
    else:
        y_pred_diff = y_pred[1:] - y_pred[:-1]
        y_diff = y[1:] - y[:-1]
    r_term = torch.mean((y_pred_diff - y_diff) ** 2)
    loss = 0.9 * loss + 0.1 * r_term

    return loss


def my_loss_p3(y_pred, y):
    '''loss two to solve the phase wrapping, add the phase derivative regularization term'''

    import torch
    import numpy as np
    phase_mse1 = (y_pred - y) ** 2
    phase_mse2 = (y_pred - y + 2 * np.pi) ** 2
    phase_mse3 = (y_pred - y - 2 * np.pi) ** 2
    phase_tensor = torch.cat((phase_mse1, phase_mse2, phase_mse3))
    phase_tensor = phase_tensor.reshape(3, 46)  # !!! the output size
    min_phase_mse = phase_tensor.min(dim=0)
    loss = torch.mean(min_phase_mse.values)
    # CNN have 3 dimension output
    if len(y_pred.shape) == 3:
        y_pred_diff = y_pred[0][0][1:] - y_pred[0][0][:-1]
        y_diff = y[0][0][1:] - y[0][0][:-1]
    # FNN only have one dimension output
    else:
        y_pred_diff = y_pred[1:] - y_pred[:-1]
        y_diff = y[1:] - y[:-1]

    diff_mse_error = (y_pred_diff - y_diff) ** 2
    diff_mse_error_plus2pi = (y_pred_diff - y_diff + 2 * np.pi) ** 2
    diff_mse_error_minus2pi = (y_pred_diff - y_diff - 2 * np.pi) ** 2
    error_buffer = torch.cat((diff_mse_error, diff_mse_error_plus2pi, diff_mse_error_minus2pi), dim=0)
    error_buffer = error_buffer.reshape(3, diff_mse_error.shape[0])
    error = error_buffer.min(dim=0)[0]  # error is the min((error),(error+2pi),(error-2pi))
    r_term = error.mean()
    # print("loss: ", loss)
    # print("r_term: ", r_term)
    loss = loss * r_term

    return loss
