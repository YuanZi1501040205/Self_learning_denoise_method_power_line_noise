"""train.py: File to train the Neural Networks for Power-line Noise Removal by using Self Supervised Method"""

# Example Usage: python train.py -train /home/yzi/research/Self_learning_denoise_method_power_line_noise/output/datasets/Self_Syn_harmonic_dataset_9.h5 -model HashResUNet1 -output /home/yzi/research/Self_learning_denoise_method_power_line_noise/
# python train.py -train /homelocal/Self_learning_denoise_method_power_line_noise/output/datasets/Self_Syn_harmonic_dataset_9.h5 -test /homelocal/Self_learning_denoise_method_power_line_noise/output/datasets/Self_Syn_harmonic_dataset_9_test.h5 -model HashResUNet1 -output /homelocal/Self_learning_denoise_method_power_line_noise/output/
#python train.py -train ./output/datasets/drift_power_line_dataset_3_hard.h5 -test ./output/datasets/drift_power_line_dataset_3_hard.h5 -model HashResUNet1 -output ./output/
__author__ = "Yuan Zi"
__email__ = "yzi2@central.uh.edu"
__version__ = "1.0.0"


def main(argv):
    """ The main function that parses input arguments, calls the appropriate
     Neural Networks models, chose train dataset' paths and configure
     the output path. input dataset should be contain the original record ( true noise + true signal) as the input,
     down-sampled blured nothch filter signal result as the signal label. true signal as the ground truth(no be used
     during training)
     Output the self learning predicted signal and training loss figures to the output path folder"""
    # Parse input arguments START
    from argparse import ArgumentParser
    import sys
    import numpy as np
    from models_zoo import models
    import os
    import torch
    import h5py
    from scipy import fftpack

    # monitor the time for each experiment
    import time



    parser = ArgumentParser()
    parser.add_argument("-train", help="specify the path of the training dataset", default= '/home/yzi/research/Self_learning_denoise_method_power_line_noise/output/figures/drift_power_line_dataset_3.h5') # /home/yzi/research/Self_learning_denoise_method_power_line_noise/output/datasets/Self_Syn_harmonic_dataset_9_test.h5
    parser.add_argument("-test", help="specify the path of the testing dataset", default= '/home/yzi/research/Self_learning_denoise_method_power_line_noise/output/figures/drift_power_line_dataset_3.h5') # /home/yzi/research/Self_learning_denoise_method_power_line_noise/output/datasets/Self_Syn_harmonic_dataset_9_test.h5
    # parser.add_argument("-train", help="specify the path of the training dataset", default= '/home/yzi/research/Self_learning_denoise_method_power_line_noise/output/datasets/Self_Syn_harmonic_dataset_9_test.h5')
    # parser.add_argument("-test", help="specify the path of the testing dataset", default= '/home/yzi/research/Self_learning_denoise_method_power_line_noise/output/datasets/Self_Syn_harmonic_dataset_9_test.h5')
    parser.add_argument("-model", help="Specify the model to train", default='HashResUNet1')
    parser.add_argument("-output", help="Specify the output path for storing the results", default='/homelocal/Self_learning_denoise_method_power_line_noise/output/')
    # parser.add_argument("-output", help="Specify the output path for storing the results", default='/home/yzi/research/Self_learning_denoise_method_power_line_noise/output/')

    args = parser.parse_args()

    # Choose training dataset. dataset is a matrix, whose size is [num traces, sig_true + sig_input + sig_notch_filter]
    #[9, 2500 + 2500 + 2500]
    if args.train is None:
        sys.exit("specify the path of the training dataset")
    else:
        path_train_dataset = args.train
        name_train_dataset = path_train_dataset.split('.')[0].split('/')[-1]
        print('training dataset: ' + name_train_dataset)

    # Choose testing dataset. dataset is a matrix, whose size is [num traces, sig_true + sig_input + sig_notch_filter]
    #[9, 2500 + 2500 + 2500]
    if args.test is None:
        sys.exit("specify the path of the training dataset")
    else:
        path_test_dataset = args.test
        name_test_dataset = path_test_dataset.split('.')[0].split('/')[-1]
        print('test dataset: ' + name_test_dataset)

    # Load model
    if args.model is None:
        sys.exit("specify model for training (choose from the models.py)")
    else:
        name_model = args.model

    # Configure the output path
    if args.output is None:
        sys.exit("specify the path of output")
    else:
        path_output = args.output
        path_figures = path_output + 'figures/'
        path_models = path_output + 'models/'

    # Parse input arguments END

    # Load the training dataset

    # Read the train dataset
    f = h5py.File(path_train_dataset, 'r')

    x = f['X']

    num_traces = x.shape[0]
    len_sig = 2500
    len_sig_label = 2500

    # extract traces
    data_traces = x
    data_traces = np.array(data_traces)

    # Read the test dataset
    f = h5py.File(path_test_dataset, 'r')

    data_test = f['X']

    # extract traces
    data_test_traces = data_test
    data_test_traces = np.array(data_test_traces)

    # Create Tensors to hold inputs and outputs
    from torch.autograd import Variable
    # convert numpy array to tensor
    data_traces = torch.tensor(data_traces).type('torch.FloatTensor')

    data_train = data_traces

    # Preprocess END

    size_input = 2500
    size_output = 2500

    # Load Dataset as batch
    batch_size = 1
    from torch.utils.data import Dataset, DataLoader
    class TrainDataset(Dataset):
        def __init__(self, data_train):
            input_indices = torch.LongTensor(list(np.array(range(size_input)) + size_output))
            output_indices = torch.LongTensor(list(np.array(range(size_output))))
            sig_label_indices = torch.LongTensor(list(np.array(range(len_sig_label)) + size_output + size_input))
            self.x = torch.index_select(data_train, 1, input_indices)
            self.true = torch.index_select(data_train, 1, output_indices)
            self.sig_label = torch.index_select(data_train, 1, sig_label_indices)

        def __len__(self):
            return len(self.true)

        def __getitem__(self, idx):
            return self.x[idx], self.true[idx], self.sig_label[idx]

    train_ds = TrainDataset(data_train)
    # DataLoader
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False)

    # Train
    epochs = 300
    np.random.seed(1997)
    # random_array = np.random.randint(x.shape[0] * x.shape[1], size=(1, 25))
    score = []
    notch_score = []
    for i, (x, true, sig_label) in enumerate(train_dl):
        loss_fig = [[],
                    [], ]  # create loss_fig to store train and validation loss during the epoch (epoch, train_loss, val_loss)
        start_time = time.time()
        # run the model for 20 epochs !!! epoch can be tuned
        index_traces = i + 1
        print('trace ' + str(index_traces))

        # training part

        # Choose model from the models.py file
        model, loss_func, optimizer = models(name_model, learn_ratio=argv[-1])

        print('model: ' + name_model)
        # assign GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        print("Using GPU: " + os.environ["CUDA_VISIBLE_DEVICES"])
        model.train()

        # load to GPU
        x = Variable(x).to(device)
        true = Variable(true).to(device)
        sig_label = Variable(sig_label).to(device)

        # reshape to [batch, channel, length]
        x = x.unsqueeze(1)
        sig_label = sig_label.unsqueeze(1)
        true = true.unsqueeze(1)

        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()

            # 1. forward propagation
            pre_sig, pre_noise = model(x)

            pre_sig = pre_sig.unsqueeze(1)
            pre_noise = pre_noise.unsqueeze(1)

            # 2. loss calculation
            loss = loss_func(pre_sig, pre_noise, sig_label, x, parameters)
            loss = loss.to(device)

            # 3. backward propagation
            loss.backward()

            # 4. weight optimization
            optimizer.step()

            # print the loss function to monitor the converge
            print("Epoch:", epoch, "Training Loss: ", loss.item())

            # record loss for each epoch
            loss_fig[0].append(epoch)
            loss_fig[1].append(loss.item())
        # check the time to train this trace
        print("--- %s seconds ---" % (time.time() - start_time))
        # save the model to the output file for reload
        torch.save(model.state_dict(), path_models + name_model + '_' + name_train_dataset +'_trace_'+ str(index_traces) +  '_state_dict.pt')
        # save the loss monitor figures
        import matplotlib.pyplot as plt
        with plt.style.context(['science', 'ieee', 'no-latex']):
            fig, ax = plt.subplots()
            plt.plot(loss_fig[0], loss_fig[1], label='Loss of train ' + name_train_dataset)
            title = 'Loss of ' + name_model + ' trained on ' + name_train_dataset + ' trace ' + str(index_traces)
            plt.title(title)
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.4), shadow=True, ncol=2)
            ax.set(xlabel='Epoch')
            ax.set(ylabel='Loss')
            ax.autoscale(tight=True)
            fig.savefig(path_figures + title + '.png', dpi=300)
            plt.cla()
            plt.clf()
            plt.close()
        print('done plot loss')

        # evaluation
        trace_name = ['BP_4_93',
                       'Marmousi_108_123',
                       'Marmousi_158_147']

        time_step = 0.002  # sample 2 ms
        fs = 1 / time_step
        time_range = 5  # signal length 5s at the time domain
        time_vec = np.arange(0, time_range, time_step)
        trace_fft = fftpack.fft(time_vec)

        # sample freq
        sample_freq = fftpack.fftfreq(trace_fft.size, d=time_step)

        # recover tensor to numpy array
        prediction_sig = pre_sig.squeeze(0).squeeze(0).cpu().detach().numpy()

        # plot time domain
        fig, axes = plt.subplots(4, 1, sharex=True, sharey=True)

        axes[0].plot(time_vec, data_test_traces[i][len_sig:-len_sig])
        axes[0].set_title('Input')

        axes[1].plot(time_vec, data_test_traces[i][-len_sig:])
        axes[1].set_title('Notch Filter')

        axes[2].plot(time_vec, data_test_traces[i][0:len_sig])
        axes[2].set_title('Ground Truth')

        axes[3].plot(time_vec, prediction_sig)
        axes[3].set_title('Prediction')
        axes[3].set_xlabel('time')
        axes[3].set_ylabel('amplitude')

        title = trace_name[i]
        fig.suptitle(title, verticalalignment='center')
        fig.tight_layout()
        plt.savefig(os.path.join(path_figures, title + '.png'))
        plt.close(fig)
        plt.cla()

        # plot frequency domain
        input_fft = fftpack.fft(data_test_traces[i][len_sig:-len_sig])
        notch_fft = fftpack.fft(data_test_traces[i][-len_sig:])
        gt_fft = fftpack.fft(data_test_traces[i][0:len_sig])
        pre_sig_fft = fftpack.fft(prediction_sig)

        fig, axes = plt.subplots(4, 1, sharex=True, sharey=True)

        axes[0].plot(sample_freq[:401], abs(input_fft)[:401], label='Input')
        axes[0].set_title('Input')

        axes[1].plot(sample_freq[:401], abs(notch_fft)[:401], label='Notch Filter')
        axes[1].set_title('Notch Filter')

        axes[2].plot(sample_freq[:401], abs(gt_fft)[:401], label='Ground Truth')
        axes[2].set_title('Ground Truth')

        axes[3].plot(sample_freq[:401], abs(pre_sig_fft)[:401], label='Prediction')
        axes[3].set_title('Prediction')

        title = trace_name[i] + 'Power'
        fig.suptitle(title, verticalalignment='center')
        fig.tight_layout()
        plt.savefig(os.path.join(path_figures, title + '.png'))
        plt.close(fig)
        plt.cla()


        # plot time results overlap
        title = "Time Prediction Result " + trace_name[i]
        plt.ylabel('Amplitude')
        plt.xlabel('Time [s]')
        plt.plot(time_vec[500: 1500], data_test_traces[i][0:len_sig][500: 1500], label='ground truth')
        plt.plot(time_vec[500: 1500], data_test_traces[i][-len_sig:][500: 1500], label='notch filter')
        plt.plot(time_vec[500: 1500], prediction_sig[500: 1500], label='prediction')
        plt.title(title)
        plt.legend()
        plt.savefig(path_figures + title + '.png')
        plt.cla()
        plt.close()

        # plot frequency results overlap
        title = "Frequency Prediction Result " + trace_name[i]
        plt.ylabel('Power')
        plt.xlabel('Frequency [hz]')
        plt.plot(sample_freq[:401], abs(gt_fft)[:401], label='Ground Truth')
        plt.plot(sample_freq[:401], abs(notch_fft)[:401], label='Notch Filter')
        plt.plot(sample_freq[:401], abs(pre_sig_fft)[:401], label='Prediction')
        plt.title(title)
        plt.legend()
        plt.savefig(path_figures + title + '.png')
        plt.cla()
        plt.close()

        score.append(np.mean((prediction_sig - data_test_traces[i][0:len_sig]) ** 2))
        notch_score.append(np.mean((data_test_traces[i][-len_sig:] - data_test_traces[i][0:len_sig]) ** 2))# notch - gt
    return score, notch_score




if __name__ == "__main__":
    import sys
    alpha = [0.001, 0.01]
    beta = [0.001, 0.01]
    lamda = [0.00000001, 0.0000001, 0.000001]
    gamma = [0.001, 0.01, 0.1,]
    suspicious_radium = [0.5, 1, 5]
    notch_weight = [0.1, 1, 10]
    learn_ratio = [0.05, 0.1, 0.2]

    # best_parameter = [0.01, 0.01, 1e-06, 0.01, 1, 1, 0.1] # best_parameter = [1, 0.001, 1, 1, 0.1]
    # parameters = best_parameter
    # score, notch_score = main(parameters)
    # print('score: ', score)
    # print('notch_score: ', notch_score)

    best_parameter = [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]]
    best_score = [float("inf"), float("inf"), float("inf")]
    for a in alpha:
        for b in beta:
            for l in lamda:
                for g in gamma:
                    for s in suspicious_radium:
                        for n in notch_weight:
                            for lr in learn_ratio:
                                parameters = [a, b, l, g, s, n, lr]
                                score, notch_score = main(parameters)
                                for i in range(3):
                                    if score[i] < best_score[i]:
                                        best_score[i] = score[i]
                                        best_parameter[i] = parameters
    print('best_score: ', best_score)
    print('notch_score: ', notch_score)
    print('best_parameter: ', best_parameter)

    # f2 = open('./best_parameters_log.txt','r+')
    # f2.read()
    # f2.write('\nbest score')
    # for i in range(3):
    #     f2.write('\n' + best_score[i])
    # f2.write('\nnotch score')
    # for i in range(3):
    #     f2.write('\n' + notch_score[i])
    # f2.write('\nbest_parameter')
    # for i in range(3):
    #     f2.write('\n' + best_parameter[i])
    # f2.close()



