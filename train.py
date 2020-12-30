"""train.py: File to train the Neural Networks for Power-line Noise Removal by using Self Supervised Method"""

# Example Usage: python train.py -train /homelocal/Self_learning_denoise_method_power_line_noise/output/datasets/Self_Syn_harmonic_dataset_9.h5 -model BlurResUNet1 -output /homelocal/Self_learning_denoise_method_power_line_noise/output/
# Example Usage: python train.py -train /home/yzi/research/Self_learning_denoise_method_power_line_noise/output/datasets/Self_Syn_harmonic_dataset_9.h5 -model HashResUNet1 -output /home/yzi/research/Self_learning_denoise_method_power_line_noise/

__author__ = "Yuan Zi"
__email__ = "yzi2@central.uh.edu"
__version__ = "1.0.0"





def main():
    """ The main function that parses input arguments, calls the appropriate
     Neural Networks models, chose train and test dataset' paths and configure
     the output path. input dataset should be contain the original record ( true noise + true signal) as the input,
     true signal as the ground truth. And a label of signal acquired by using notch filter.
     Output the self learning predicted signal and training loss figures to the output path folder"""
    # Parse input arguments START
    from argparse import ArgumentParser
    import sys
    import numpy as np
    from models_zoo import models
    import os
    import torch
    import h5py

    # monitor the time for each experiment
    import time


    parser = ArgumentParser()
    parser.add_argument("-train", help="specify the path of the training dataset", default= '/home/yzi/research/Self_learning_denoise_method_power_line_noise/output/datasets/Self_Syn_harmonic_dataset_9.h5')
    parser.add_argument("-model", help="Specify the model to train", default='HashResUNet1')
    parser.add_argument("-output", help="Specify the output path for storing the results", default='/home/yzi/research/Self_learning_denoise_method_power_line_noise/')

    args = parser.parse_args()

    # Choose training dataset. dataset is a matrix, whose size is [num traces, sig_true + sig_input + sig_ds_blur]
    #[9, 2500 + 2500 + 500]
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

    # Load the training dataset

    # Read the train dataset
    f = h5py.File(path_train_dataset, 'r')

    x = f['X']

    num_traces = x.shape[0]
    len_sig = 2500
    len_sig_label = 500

    # extract traces
    data_traces = x
    data_traces = np.array(data_traces)

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
    epochs = 200
    np.random.seed(1997)
    # random_array = np.random.randint(x.shape[0] * x.shape[1], size=(1, 25))
    for i, (x, true, sig_label) in enumerate(train_dl):
        loss_fig = [[],
                    [], ]  # create loss_fig to store train and validation loss during the epoch (epoch, train_loss, val_loss)
        start_time = time.time()
        # run the model for 20 epochs !!! epoch can be tuned
        index_traces = i + 1
        print('trace ' + str(index_traces))

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

            loss = loss_func(pre_sig, pre_noise, sig_label, x)
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

if __name__ == "__main__":
    train  = '/home/yzi/research/Self_learning_denoise_method_power_line_noise/output/datasets/Self_Syn_harmonic_dataset_9.h5'
    model = 'HashResUNet1'
    output =  '/home/yzi/research/Self_learning_denoise_method_power_line_noise/'
    main()
