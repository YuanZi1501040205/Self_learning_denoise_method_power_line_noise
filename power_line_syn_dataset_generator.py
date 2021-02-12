"""power_line_syn_dataset_generator.py: File to generate sinusoid noise and signal"""

# Example Usage: python power_line_syn_dataset_generator.py -output /homelocal/AGT_FWI_2020/output/

__author__ = "Yuan Zi"
__email__ = "yzi2@central.uh.edu"
__version__ = "1.0.0"

# monitor the time for each experiment
import time

start_time = time.time()


def main():
    """ The main function that parses input arguments, read the appropriate
     dataset and wavelet as inputs, and preprocess the dataset write to the h5 file for traning"""
    # Parse input arguments START
    from argparse import ArgumentParser
    import sys
    import numpy as np
    import h5py
    import random
    from Functions import butter_bandpass_filter
    from Functions import butter_lowpass_filter

    parser = ArgumentParser()

    parser.add_argument("-output", help="Specify the output path for storing the results")

    args = parser.parse_args()

    # Configure the output path
    if args.output is None:
        sys.exit("specify the path of output")
    else:
        path_output = args.output
        path_output_dataset = path_output + 'datasets/'
    # configure the dataset size to generate
    num_receiver = 1
    num_shot = 1
    time_step = 0.002  # sample 2 ms
    fs = 1/time_step
    time_range = 5  # signal length 5s at the time domin
    time_vec = np.arange(0, time_range, time_step)
    # configure the number of model's layer and maximum value
    num_traces = num_receiver * num_shot
    num_layers = 10
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
    # Calculate the X (all data)
    X = []
    # Convert dataset to format of (num_shot, num_receiver, len_trace)
    for i in range(num_shots):
        X.append([])
        for j in range(num_receivers):
            X[i].append([])
            # extract trace
            sig_trace = traces[i * num_receivers  + j]
            # shift
            from Functions import guassian_filter

            sig_true = butter_lowpass_filter(sig_trace, 60, fs, order=5)  # extract 1-40Hz signal as the ground truth of NN
            sig_input = butter_bandpass_filter(sig_trace, 10, 50, fs, order=5)  # extract 10-30Hz signal as the input of NN

            data_trace = np.concatenate((sig_true, sig_input), axis=0)
            X[i][j].append(data_trace)

    X = np.array(X).squeeze(2)
    # Write the dataset to HDF5 file

    name_data = 'Self_Syn_10l_192_dataset'

    f = h5py.File(path_output_dataset + 'Time_' + name_data + '.h5', 'w')
    x_freq = time_vec  # in Time case, replace the frequency component to signal's time length
    length = np.zeros(num_shots).astype(int) + int(num_receivers)
    f.create_dataset('X', data=X)
    f.create_dataset('X_freq', data=x_freq)
    f.create_dataset('len', data=length)
    f.close()
    split =True
    if split:
        X_train = []
        X_test = []
        for i in range(num_shots):
            if i % 2 == 1:
                X_test.append(X[i])
                print('write ' + str(i) + ' shot to train')
            else:
                X_train.append(X[i])

        path_train = path_output_dataset + 'Time_' + name_data + '_Train.h5'
        # write the train dataset
        f = h5py.File(path_train, 'w')
        length = np.zeros(int(num_shots / 2)).astype(int) + int(num_receivers)
        f.create_dataset('X', data=X_train)
        f.create_dataset('X_freq', data=x_freq)
        f.create_dataset('len', data=length)
        f.close()

        path_test = path_output_dataset + 'Time_' + name_data + '_Test.h5'
        # write the test dataset
        f = h5py.File(path_test, 'w')
        x_freq = x_freq
        length = np.zeros(int(num_shots / 2)).astype(int) + int(num_receivers)
        f.create_dataset('X', data=X_test)
        f.create_dataset('X_freq', data=x_freq)
        f.create_dataset('len', data=length)
        f.close()
    else:
        pass


if __name__ == "__main__":
    main()
