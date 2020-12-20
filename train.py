"""_train.py: File to train the Neural Networks for Powerline Noise Removal by using Self Supervised Method"""

# Example Usage: python train.py -train /homelocal/Self_learning_denoise_method_power_line_noise/output/datasets/Self_Syn_harmonic_dataset_9.h5 -model BlurResUNet1 -output /homelocal/Self_learning_denoise_method_power_line_noise/output/

__author__ = "Yuan Zi"
__email__ = "yzi2@central.uh.edu"
__version__ = "1.0.0"




def main():
    """ The main function that parses input arguments, calls the appropriate
     Neural Networks models and chose train and test dataset' paths and configure
     the output path. input dataset should be contain the original record (noise + signal)
     and a label of signal acquired by using notch filter. Output the comparison result of the notch filter result and the new results.
     results figure at the output path folder"""
    # Parse input arguments START
    from argparse import ArgumentParser
    import sys
    import numpy as np
    from models_zoo import models
    from Functions import load_dataset
    from Functions import extract
    import os
    import torch
    import math
    from scipy import fftpack

    # monitor the time for each experiment
    import time



    parser = ArgumentParser()
    parser.add_argument("-train", help="specify the path of the training dataset")
    parser.add_argument("-model", help="Specify the model to train")
    parser.add_argument("-output", help="Specify the output path for storing the results")

    args = parser.parse_args()

    # Choose training dataset
    if args.train is None:
        sys.exit("specify the path of the training dataset")
    else:
        path_train_dataset = args.train
        name_train_dataset = path_train_dataset.split('.')[0].split('/')[-1]
        print('training dataset: ' + name_train_dataset)

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
        path_models = path_output + 'Self_models/'

    # Parse input arguments END

    # Preprocess START
    # Load the training and validation dataset
    x, x_freq, length = load_dataset(path_train_dataset)
    Num_shot = x.shape[0]
    Num_receiver = x.shape[1]
    # extract traces
    data_traces = extract(x, length)
    data_traces = np.array(data_traces)
    data_traces = data_traces.real

    # Create Tensors to hold inputs and outputs
    from torch.autograd import Variable
    # convert numpy array to tensor
    data_traces = torch.tensor(data_traces).type('torch.FloatTensor')

    data_train = data_traces

    # Preprocess END

    num_input = 2500
    num_output = 2500

    # Load Dataset as batch
    batch_size = 1
    from torch.utils.data import Dataset, DataLoader
    class TrainDataset(Dataset):
        def __init__(self, data_train):
            input_indices = torch.LongTensor(list(np.array(range(num_input)) + num_output))
            output_indices = torch.LongTensor(list(np.array(range(num_output))))
            self.x = torch.index_select(data_train, 1, input_indices)
            self.true = torch.index_select(data_train, 1, output_indices)

        def __len__(self):
            return len(self.true)

        def __getitem__(self, idx):
            return self.x[idx], self.true[idx]

    train_ds = TrainDataset(data_train)
    # DataLoader
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False)


    # Train
    plot_count = 0
    epochs = 200
    np.random.seed(1997)
    random_array = np.random.randint(x.shape[0] * x.shape[1], size=(1, 25))
    for i, (x, true) in enumerate(train_dl):
        if i in random_array:
            loss_fig = [[],
                        []]  # create loss_fig to store train and validation loss during the epoch (epoch, train_loss, val_loss)
            start_time = time.time()
            # run the model for 20 epochs !!! epoch can be tuned
            num_shot = int(i / Num_shot) + 1
            num_receiver = i % Num_receiver + 1
            print('Shot ' + str(num_shot) + ' Receiver ' + str(num_receiver))

            # training part

            # Choose model from the models.py file
            model, loss_func, optimizer = models(name_model)

            print('model: ' + name_model)
            # assign GPU
            os.environ["CUDA_VISIBLE_DEVICES"] = "1"
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            print("Using GPU: " + os.environ["CUDA_VISIBLE_DEVICES"])
            model.train()
            #
            x = Variable(x).to(device)
            true = Variable(true).to(device)

            x = x.unsqueeze(1)
            # true = true.unsqueeze(1).to(device)
            # for data in data_train.data:
            for epoch in range(1, epochs + 1):
                optimizer.zero_grad()

                # 1. forward propagation
                y_pred = model(x)

                y_pred = y_pred.unsqueeze(1)

                # 2. loss calculation

                loss = loss_func(y_pred, true)
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
            torch.save(model.state_dict(), path_models + name_model + '_' + name_train_dataset +'_shot_'+ str(num_shot) + "_receiver_" + str(num_receiver) + '_state_dict.pt')
            # save the loss monitor figures
            import matplotlib.pyplot as plt
            with plt.style.context(['science', 'ieee', 'no-latex']):
                fig, ax = plt.subplots()
                plt.plot(loss_fig[0], loss_fig[1], label='Loss of train ' + name_train_dataset)
                title = 'Loss of ' + name_model + ' trained on ' + name_train_dataset + ' shot ' + str(num_shot) + ' receiver ' + str(num_receiver)
                plt.title(title)
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.4), shadow=True, ncol=2)
                ax.set(xlabel='Epoch')
                ax.set(ylabel='Loss')
                ax.autoscale(tight=True)
                # fig.savefig('figures/fig1.pdf')
                fig.savefig(path_figures + title + '.png', dpi=300)
                plt.cla()
                plt.clf()
                plt.close()
            print('done plot loss')

            # save the predicted signal trace's Frequency spectrum figures
            import matplotlib.pyplot as plt
            # 1. forward propagation
            path_model = path_models + name_model + '_' + name_train_dataset +'_shot_'+ str(num_shot) + "_receiver_" + str(num_receiver) + '_state_dict.pt'
            model.load_state_dict(torch.load(path_model))
            model.eval()
            y_pred = model(x)# x:[1,1,2500] y_predL[1,2500]
            # # CNN

            x = x.squeeze(0).squeeze(0)
            y_pred = y_pred.squeeze(0)
            true = true.squeeze(0)

            x = x.cpu().data.numpy()
            y_pred = y_pred.cpu().data.numpy()
            true = true.cpu().data.numpy()
            # FFT
            time_step = 0.002  # sample 2 ms
            time_range = 5  # signal length 5s at the time domain
            time_vec = np.arange(0, time_range, time_step)
            x_fft = fftpack.fft(x)
            y_pred_fft = fftpack.fft(y_pred)
            true_fft = fftpack.fft(true)
            x_fft_power = abs(x_fft)
            y_pred_fft_power = abs(y_pred_fft)
            true_fft_power = abs(true_fft)
            sample_freq = fftpack.fftfreq(true_fft.size, d=time_step)
            # Frequency Power
            with plt.style.context(['science', 'ieee', 'no-latex']):
                fig, ax = plt.subplots()
                plt.plot(sample_freq[5:301], true_fft_power[5:301], label='Ground True')
                plt.plot(sample_freq[5:301], y_pred_fft_power[5:301], label='Predicted')
                plt.plot(sample_freq[5:301], x_fft_power[5:301], label='Input')

                title = 'Frequency Spectrum of ' + name_model + '_' + name_train_dataset + "_shot_"+str(num_shot) + "_receiver"+ str(num_receiver)
                plt.title(title)
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.4), shadow=True, ncol=2)
                ax.set(xlabel='Frequency')
                ax.set(ylabel='Power')
                ax.autoscale(tight=True)
                fig.savefig(path_figures + title + '.png', dpi=300)
                plt.cla()
                plt.clf()
                plt.close()
            # Time domain
            with plt.style.context(['science', 'ieee', 'no-latex']):
                fig, ax = plt.subplots()
                plt.plot(time_vec, true, label='Ground True')
                plt.plot(time_vec, y_pred, label='Predicted')
                plt.plot(time_vec, x, label='input')
                title = 'Time Signal of ' + name_model + '_' + name_train_dataset + '_shot'+str(num_shot) + "_receiver" + str(
                    num_receiver)
                plt.title(title)
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.4), shadow=True, ncol=2)
                ax.set(xlabel='Time')
                ax.set(ylabel='Amplitude')
                ax.autoscale(tight=True)
                # fig.savefig('figures/fig1.pdf')
                fig.savefig(path_figures + title + '.png', dpi=300)
                plt.cla()
                plt.clf()
                plt.close()
            # Zoomed Time domain
            with plt.style.context(['science', 'ieee', 'no-latex']):
                fig, ax = plt.subplots()
                plt.plot(time_vec[1250:1750], true[1250:1750], label='Ground True')
                plt.plot(time_vec[1250:1750], y_pred[1250:1750], label='Predicted')
                plt.plot(time_vec[1250:1750], x[1250:1750], label='input')
                title = 'Zoomed Time Signal of ' + name_model + '_' + name_train_dataset + '_shot'+str(num_shot) + "_receiver" + str(
                    num_receiver)
                plt.title(title)
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.4), shadow=True, ncol=2)
                ax.set(xlabel='Time')
                ax.set(ylabel='Amplitude')
                ax.autoscale(tight=True)
                # fig.savefig('figures/fig1.pdf')
                fig.savefig(path_figures + title + '.png', dpi=300)
                plt.cla()
                plt.clf()
                plt.close()
            # Phase Spectrum domain
            y_pred_fft = y_pred_fft[5:301]
            true_fft = true_fft[5:301]
            y_pred_phase = []
            true_phase = []
            for i in range(true_fft.shape[0]):
                y_pred_phase.append(math.atan2(y_pred_fft[i].imag, y_pred_fft[i].real))
                true_phase.append(math.atan2(true_fft[i].imag, true_fft[i].real))

            with plt.style.context(['science', 'ieee', 'no-latex']):
                    fig, ax = plt.subplots()
                    plt.plot(sample_freq[5:301], true_phase, label='Ground True')
                    plt.plot(sample_freq[5:301], y_pred_phase, label='Predicted')
                    title = 'Phase of ' + name_model + '_' + name_train_dataset + '_shot'+str(num_shot) + "_receiver" + str(
                        num_receiver)
                    plt.title(title)
                    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.4), shadow=True, ncol=2)
                    ax.set(xlabel='Frequency')
                    ax.set(ylabel='Phase')
                    ax.autoscale(tight=True)
                    # fig.savefig('figures/fig1.pdf')
                    fig.savefig(path_figures + title + '.png', dpi=300)
                    plt.cla()
                    plt.clf()
                    plt.close()
            with plt.style.context(['science', 'ieee', 'no-latex']):
                fig, ax = plt.subplots()
                plt.plot(sample_freq[5:101], true_phase[:96], label='Ground True')
                plt.plot(sample_freq[5:101], y_pred_phase[:96], label='Predicted')
                title = 'Zoom Phase of ' + name_model + '_' + name_train_dataset +'_shot'+ str(num_shot) + "_receiver" + str(
                    num_receiver)
                plt.title(title)
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.4), shadow=True, ncol=2)
                ax.set(xlabel='Frequency')
                ax.set(ylabel='Phase')
                ax.autoscale(tight=True)
                # fig.savefig('figures/fig1.pdf')
                fig.savefig(path_figures + title + '.png', dpi=300)
                plt.cla()
                plt.clf()
                plt.close()
            print("done plot shot" + str(num_shot) + " receiver " + str(num_receiver))
            plot_count =  plot_count + 1
            if plot_count == 5:
                0/0


if __name__ == "__main__":
    main()
