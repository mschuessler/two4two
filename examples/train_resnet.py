import json
import optparse
import os
import time


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import io
import torch
from torchvision import models, transforms


def create_log_path(opt):
    unique_path = False

    if opt.save_dir != "":
        opt.log_path = os.path.join(opt.data_output_dir, opt.save_dir)
        unique_path = True

    # do not overwrite
    while os.path.exists(opt.log_path) and not unique_path:
        opt.log_path += "_" + str(np.random.randint(100))

    if not os.path.exists(opt.log_path):
        os.makedirs(opt.log_path)


def test_model(opt, model, dataloader, criterion):
    total_step = len(dataloader)

    print('Num steps', total_step)

    model.eval()

    epoch_acc = 0

    running_loss = 0.0
    running_corrects = 0

    for step, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(opt.device)
        labels = labels.to(opt.device)

        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data).item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = running_corrects / float(len(dataloader) * opt.batch_size)

    print('Val/Test Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

    return epoch_loss, epoch_acc


def train_model(opt, model,
                dataloader, criterion, optimizer,
                scheduler, validation_dataloader=None,
                plot=False):
    since = time.time()

    losses = []
    accuracies = []
    val_losses = []
    val_accuracies = []

    print('Num steps', len(dataloader))

    for epoch in range(opt.num_epochs):
        print('Epoch {}/{}'.format(epoch, opt.num_epochs - 1))
        print('-' * 10)

        model.train()

        running_loss = 0.0
        running_corrects = 0

        for step, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(opt.device)
            labels = labels.to(opt.device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloader)

        losses.append(epoch_loss)

        epoch_acc = running_corrects.double(
        ) / (len(dataloader) * opt.batch_size)

        accuracies.append(epoch_acc.item())

        print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        if epoch % opt.ckpt_freq == 0:
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'loss': loss},
                       os.path.join(opt.log_path, 'model_{}.ckpt'.format(epoch)))
            if validation_dataloader is not None:
                # evaluate on validation set
                val_loss, val_acc = test_model(
                    opt, model, validation_dataloader, criterion)
                val_losses.append(val_loss)
                val_accuracies.append(val_acc)

        scheduler.step()

    time_elapsed = time.time() - since

    print('Time elapsed:', time_elapsed)

    if plot:
        plt.plot(losses, color='blue', label='training_loss')
        plt.plot(val_losses, color='red', label='validation_loss')
        plt.legend(loc='upper left')
        plt.show()

    return model, losses, accuracies, val_losses, val_accuracies


def initialize_model(opt, num_classes):
    model = None
    use_pretrained = False
    if opt.type == 'pretrained':
        use_pretrained = True

    if opt.model_name == "resnet50":
        model = models.resnet50(pretrained=use_pretrained)
    elif opt.model_name == "resnet34":
        model = models.resnet34(pretrained=use_pretrained)
    elif opt.model_name == "resnet18":
        model = models.resnet18(pretrained=use_pretrained)

    # 4 channel
    model.conv1 = torch.nn.Conv2d(
        opt.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)

    return model


def reload_model(opt, model):
    print('LOADING', os.path.join(opt.model_path,
                                  "model_{}.ckpt".format(opt.model_num)))

    model.load_state_dict(
        torch.load(
            os.path.join(opt.model_path,
                         "model_{}.ckpt".format(opt.model_num)),
            map_location=opt.device.type))

    return model


class Two4TwoDataset(torch.utils.data.Dataset):
    def __init__(self,
                 opt,
                 mode='train',
                 transform=transforms.ToTensor()):
        self.opt = opt

        self.root_dir = os.path.join(self.opt.data_input_dir, mode)
        self.parameters_file = os.path.join(self.root_dir, 'parameters.jsonl')

        self.parameters = self.create_df(mode)
        self.id_col_idx = self.parameters.columns.get_loc("id")
        self.label_col_idx = self.parameters.columns.get_loc("label")

        self.transform = transform

    def __len__(self):
        return len(self.parameters)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.parameters.iloc[idx, self.id_col_idx] + '.png')
        image = io.imread(str(img_name))

        label = self.parameters.iloc[idx, self.label_col_idx]

        image = self.transform(transforms.ToPILImage()(image))

        sample = (image / 255., label)

        return sample

    def create_df(self, mode):

        label_data = pd.read_json(self.parameters_file, lines=True)
        label_data['label'] = label_data['obj_name'].apply(
            lambda x: 0 if x == 'sticky' else 1)

        return label_data


def run_for_seed(opt):

    # set random seeds
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    train_dataset = Two4TwoDataset(opt, mode='train')
    test_dataset = Two4TwoDataset(opt, mode='test')
    validation_dataset = Two4TwoDataset(opt, mode='validation')

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=opt.batch_size, shuffle=True,
                                               num_workers=opt.num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=opt.batch_size, shuffle=False,
                                              num_workers=opt.num_workers)
    validation_loader = torch.utils.data.DataLoader(validation_dataset,
                                                    batch_size=opt.batch_size, shuffle=False,
                                                    num_workers=opt.num_workers)

    model = initialize_model(opt, 10)

    model = model.to(opt.device)

    params_to_update = model.parameters()

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=20, gamma=0.1, verbose=True)

    criterion = torch.nn.CrossEntropyLoss()

    # TRAIN
    model, losses, accuracies, val_losses, val_accuracies = train_model(opt,
                                                                        model,
                                                                        train_loader,
                                                                        criterion,
                                                                        optimizer,
                                                                        scheduler,
                                                                        validation_loader)

    final_loss, final_acc = test_model(opt, model,
                                       test_loader,
                                       criterion)

    out = dict()
    out['losses'] = losses
    out['accuracies'] = accuracies
    out['val_losses'] = val_losses
    out['val_accuracies'] = val_accuracies
    out['final_loss'] = final_loss
    out['final_accuracy'] = final_acc

    with open(os.path.join(opt.log_path, 'results.txt'), 'w') as f:
        json.dump(out, f)


if __name__ == '__main__':
    seeds = [1, 12, 7]

    parser = optparse.OptionParser()

    parser.add_option('--batch_size', default=128,
                      type=int, help='Batch size per GPU')
    parser.add_option('--learning_rate', default=1e-3,
                      type=float, help='Learning rate')
    parser.add_option('--num_epochs', default=50, type=int,
                      help='Number of epochs to train')
    parser.add_option('--in_channels', default=4, type=int,
                      help='Number of epochs to train')
    parser.add_option('--num_workers', default=4, type=int,
                      help='Number of data loader threads')
    parser.add_option("--resume", action="store_true", default=False,
                      help="Resume training")
    parser.add_option('--seed', type=int, default=0,
                      help='Random seed for reproducibility')
    parser.add_option('--ckpt_freq', type=int, default=10,
                      help='Checkpoint frequency')
    parser.add_option('--model_name', type=str,
                      default='resnet18', help='Model name resnet18/34/50')
    parser.add_option('--type', type=str,
                      default='pretrained', help='scratch or pretrained')
    parser.add_option('--model_path', type=str, default='./',
                      help='Load model path. Use when resuming')
    parser.add_option('--model_num', type=int, default=100,
                      help='Load model number. Use when resuming')
    parser.add_option('--data_output_dir', type=str,
                      default='/media/oanaucs/Stuff/inn_models', help='Output model dir')
    parser.add_option('--save_dir', type=str, default='celeba_r18',
                      help='Training name for save folder')
    parser.add_option('--data_input_dir', type=str,
                      default='/media/oanaucs/Stuff/datasets/medVarBgBias', help='Data input dir')

    opt, _ = parser.parse_args()

    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for i in range(len(seeds)):
        opt.seed = seeds[i]

        opt.save_dir = 'two4two_' + opt.type + '_' + str(opt.seed)
        create_log_path(opt)

        with open(os.path.join(opt.log_path, 'opt.txt'), 'w') as f:
            f.write(str(opt))

        run_for_seed(opt)
