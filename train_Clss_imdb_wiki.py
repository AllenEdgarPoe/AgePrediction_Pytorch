import torch
from Dataloader import AgeGenderDataset
import argparse
from torch.utils.data import DataLoader
from util import *
import shutil
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from model import *
from tqdm import tqdm
import os, shutil, pickle
from tensorboardX import SummaryWriter
from model import resnet101

def train(config):

    train_set = AgeGenderDataset(config.train_csv_path, transform=image_transformer()['train'], LAP=config.LAP)
    trainloader = DataLoader(train_set, batch_size = config.batch_size, shuffle=True, num_workers=8)
    # weights = make_weights_for_balanced_classes(train_set, (config.max - config.min) // config.interval)
    # with open ('./img_weight/LAP_10_60.pkl','rb') as f:
    #     weights = pickle.load(f)
    # weights = torch.DoubleTensor(weights)
    # sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    # trainloader = DataLoader(train_set, batch_size=config.batch_size, sampler=sampler, shuffle=False)
    valid_set = AgeGenderDataset(config.test_csv_path, transform=image_transformer()['val'], LAP=config.LAP)
    validloader = DataLoader(valid_set, batch_size=config.batch_size, shuffle=True, num_workers=8)

    device = torch.device("cuda", index=1)

    if config.checkpoint is None:
        start_epoch = 0
        model = resnet101(100)
        # optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)
        optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
    else:
        checkpoint = torch.load(config.checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        # optimizer = optim.Adam(model.parameters(), lr=config.lr)

    model = model.to(device)
    age_criterion = nn.L1Loss().to(device)

    train_loss_idx = 0
    test_loss_idx = 0
    least_mae = 100
    epsilon_error =0.0
    exp_least_mae = 100
    exp_epsilon_error =0.0
    for epoch in range(start_epoch, start_epoch + config.num_epochs):
        if epoch in config.lr_decay_epochs:
            for param_group in optimizer.param_groups:
                pre_lr = param_group['lr']
                new_lr = pre_lr * config.lr_decay_rate
                param_group['lr'] = new_lr
            print('lr decay: %f --> %f (rate: %f)' % (pre_lr, new_lr, config.lr_decay_rate))

        for mode in config.mode:
            if mode == 'train':
                model.train()
                train_loader_pbar = tqdm(trainloader)
                train_loss_sum = 0.0
                total = 0
                total_acc = 0.0
                maes = 0.0
                exp_maes = 0.0
                total_epsilon = 0.0
                exp_total_epsilon = 0.0

                for batch_idx, data in enumerate(train_loader_pbar):
                    if config.LAP:
                        img, age, std = data
                        std = torch.autograd.Variable(std.to(device))
                    else:
                        img, age = data

                    img = torch.autograd.Variable(img.float().to(device), requires_grad=True)
                    age = torch.autograd.Variable(age.float().to(device), requires_grad=True)
                    #forward
                    age_pred = model(img)
                    a = np.arange(0, 100)
                    a = np.tile(a, ((age_pred.size(0), 1)))
                    exp_age_pred = torch.sum(age_pred * torch.tensor(a).to(device), 1).data
                    #Loss
                    # age_loss = age_criterion(torch.log(age_pred), age)
                    age_loss = age_criterion(exp_age_pred, age)
                    #Backward
                    optimizer.zero_grad()
                    age_loss.backward()
                    #Update Model
                    optimizer.step()

                    train_loss_sum += age_loss.item()
                    total += 1
                    #Calculate Accuracy

                    # print(exp_age_pred)
                    int_age_pred = torch.argmax(age_pred, dim=1).data
                    int_age = age
                    mae = abs(int_age - int_age_pred).sum()/config.batch_size
                    exp_mae = abs(int_age - exp_age_pred).sum()/config.batch_size
                    exp_maes+=exp_mae
                    maes+=mae
                    acc = (int_age == int_age_pred).sum().item()/config.batch_size
                    total_acc +=acc

                    if config.LAP:
                        epsilon = 1 - torch.exp(-(torch.square(int_age_pred - int_age)) / (2 * torch.square(std)))
                        total_epsilon += sum(epsilon.data) / config.batch_size

                        exp_epsilon = 1 - torch.exp(-(torch.square(exp_age_pred - int_age)) / (2 * torch.square(std)))
                        exp_total_epsilon += sum(exp_epsilon.data) / config.batch_size

                        train_loader_pbar.set_description(
                            '[training] epoch:%d/%d, ' % (epoch, config.num_epochs+start_epoch) +
                            'train_loss:%.3f, ' % (train_loss_sum / total) +
                            'Accuracy:%.3f ' % (total_acc / total) +
                            'MAE:%.3f(%.3f) ' % (mae, maes / total) +
                            'Epsilon error:%.3f ' % (total_epsilon / total) +
                            'Expected MAE:%.3f(%.3f) ' % (exp_mae, exp_maes / total) +
                            'Expected Epsilon error:%.3f ' % (exp_total_epsilon / total) +
                            'predicted:%d ' % (int_age_pred[0]) +
                            'Answer:%d ' % (int_age[0]))

                    else:
                        train_loader_pbar.set_description(
                            '[training] epoch:%d/%d, ' % (epoch, config.num_epochs+start_epoch) +
                            'train_loss:%.3f, ' % (train_loss_sum/total) +
                            'Accuracy:%.3f ' % (total_acc/total) +
                            'MAE:%.3f(%.3f) ' % (mae, maes/total) +
                            'Expected MAE:%.3f(%.3f) ' % (exp_mae, exp_maes / total) +
                            'predicted:%d ' % (int_age_pred[0]) +
                            'Answer:%d ' % (int_age[0]))

                summary.add_scalar('train mae', maes/total, train_loss_idx)
                train_loss_idx+=1

                if config.LAP:
                    f = open(os.path.join(config.logger_path, 'new_train.txt'), 'a')
                    f.write(
                        'Epoch: {}\t Loss: {:.3f}\t Accuracy: {:.3f}\t Mae: {:.3f}\t Epsilon error: {:.3f}\t Expected Mae: {:.3f}\t Exp_Epsilon error: {:.3f}\n'.format(
                            epoch, train_loss_sum / total, total_acc / total, maes / total, total_epsilon / total, exp_maes / total, exp_total_epsilon/total))
                else:
                    f = open(os.path.join(config.logger_path, 'new2_train.txt'), 'a')
                    f.write('Epoch: {}\t Loss: {:.3f}\t Accuracy: {:.3f}\t Mae: {:.3f}\t Expected Mae: {:.3f}\n'.format(epoch,train_loss_sum/total, total_acc/total, maes/total, exp_maes/total))


            elif mode == 'eval':
                model.eval()
                val_loss_sum = 0.0
                total = 0
                total_acc = 0.0
                maes = 0.0
                exp_maes =0.0
                val_loader_pbar = tqdm(validloader)
                total_epsilon = 0.0
                exp_total_epsilon =0.0
                for batch_idx, data in enumerate(val_loader_pbar):
                    if config.LAP:
                        img, age, std = data
                        std = torch.autograd.Variable(std.to(device))
                    else:
                        img, age = data

                    img = torch.autograd.Variable(img.float().to(device))
                    age = torch.autograd.Variable(age.long().to(device))
                    # forward
                    age_pred= model(img)
                    a = np.arange(0, 100)
                    a = np.tile(a, ((age_pred.size(0), 1)))
                    exp_age_pred = torch.sum(age_pred * torch.tensor(a).to(device), 1).data

                    # Loss
                    # age_loss = age_criterion(torch.log(age_pred), age)
                    age_loss = age_criterion(exp_age_pred, age)

                    optimizer.zero_grad()
                    val_loss_sum += age_loss.item()
                    total += 1
                    # Calculate Accuracy

                    int_age_pred = torch.argmax(age_pred, dim=1).data
                    int_age = age
                    acc = (int_age == int_age_pred).sum().item() /config.batch_size
                    mae = abs(int_age - int_age_pred).sum() /config.batch_size
                    exp_mae = abs(int_age - exp_age_pred).sum() /config.batch_size
                    exp_maes +=exp_mae
                    maes += mae
                    total_acc +=acc

                    if config.LAP:
                        epsilon = 1 - torch.exp(-(torch.square(int_age_pred - int_age)) / (2 * torch.square(std)))
                        total_epsilon += sum(epsilon.data) / config.batch_size
                        exp_epsilon = 1 - torch.exp(-(torch.square(exp_age_pred - int_age)) / (2 * torch.square(std)))
                        exp_total_epsilon += sum(exp_epsilon.data) / config.batch_size

                        val_loader_pbar.set_description(
                            '[testing] epoch:%d/%d, ' % (epoch, config.num_epochs+start_epoch) +
                            'valid_loss:%.3f, ' % (val_loss_sum / total) +
                            'Accuracy:%.3f ' % (total_acc / total) +
                            'MAE:%.3f(%.3f) ' % (mae, maes/ total) +
                            'Epsilon error:%.3f ' % (total_epsilon / total) +
                            'Expected MAE:%.3f(%.3f) ' % (exp_mae, exp_maes / total) +
                            'Expected Epsilon error:%.3f ' % (exp_total_epsilon / total) +
                            'predicted:%d ' % (int_age_pred[0]) +
                            'Answer:%d ' % (int_age[0]))

                    else:
                        val_loader_pbar.set_description(
                            '[testing] epoch:%d/%d, ' % (epoch, config.num_epochs+start_epoch) +
                            'valid_loss:%.3f, ' % (val_loss_sum / total) +
                            'Accuracy:%.3f ' % (total_acc / total) +
                            'MAE:%.3f(%.3f) ' % (mae, maes/ total) +
                            'Expected MAE:%.3f(%.3f) ' % (exp_mae, exp_maes / total) +
                            'predicted:%d ' % (int_age_pred[0]) +
                            'Answer:%d ' % (int_age[0]))

                if maes/total < least_mae:
                    least_mae = maes/total
                    if config.LAP:
                        epsilon_error = total_epsilon/total

                if exp_maes/total < exp_least_mae:
                    exp_least_mae = exp_maes/total
                    if config.LAP:
                        exp_epsilon_error = total_epsilon/total

                summary.add_scalar('valid mae', maes/total, test_loss_idx)
                test_loss_idx+=1

                if config.LAP:
                    print("===Smallest MAE: {:.3f}, Epsilon error: {:.3f}, Exp_MAE: {:.3f}, Exp_epsilon_error: {:.3f}".format(least_mae, epsilon_error, exp_least_mae, exp_epsilon_error))
                else:
                    print("===Smallest MAE: {:.3f}, Exp_MAE: {:.3f}".format(least_mae, exp_least_mae))

                if config.LAP:
                    f = open(os.path.join(config.logger_path, 'new_val.txt'), 'a')
                    f.write(
                        'Epoch: {}\t Loss: {:.3f}\t Accuracy: {:.3f}\t Mae: {:.3f}\t Epsilon error: {:.3f}\t Expected Mae: {:.3f}\t Exp_Epsilon error: {:.3f}\n'.format(
                            epoch, train_loss_sum / total, total_acc / total, maes / total, total_epsilon / total,
                                   exp_maes / total, exp_total_epsilon / total))
                else:
                    f = open(os.path.join(config.logger_path, 'new2_val.txt'), 'a')
                    f.write('Epoch: {}\t Loss: {:.3f}\t Accuracy: {:.3f}\t Mae: {:.3f}\t Expected Mae: {:.3f}\n'.format(
                        epoch, train_loss_sum / total, total_acc / total, maes / total, exp_maes / total))

    if config.model_save == True:
        model_out_path = config.model_save_path + str(epoch) + '.pth'
        state = {'epoch': epoch, 'model': model, 'optimizer': optimizer}
        # torch.save(model.module.state_dict(), model_out_path)
        torch.save(state, model_out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='imdb_crop')
    parser.add_argument('--LAP', type=bool, default=False)
    parser.add_argument('--train_csv_path', default='./data/LAP/LAP_train.csv')
    # parser.add_argument('--train_csv_path', default='./data/imdb_wiki_train.csv')
    parser.add_argument('--test_csv_path', default='./data/LAP/LAP_test.csv')
    # parser.add_argument('--model_save_path', type=str, default="model_load/model1_LAP/Clss/resent101_epoch_")
    parser.add_argument('--model_save_path', type=str, default="model_load/model2_IMDB_LAP/Clss/new2_resnet101_epoch")
    # parser.add_argument('--checkpoint', default='model_load/model2_IMDB_LAP/Clss/new_resnet101_epoch29.pth')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--summary_write_path', default='./log_save/model2/Clss/resnet101')
    parser.add_argument('--logger_path', default='./logger/model2/Clss/resnet101/')
    parser.add_argument('--img_resize', type=int, default=48)
    parser.add_argument('--model_save', default=True)

    parser.add_argument('--mode', default=['train','eval'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=int, default=0.01)
    parser.add_argument('--num_epochs', type=int, default=30)

    parser.add_argument('--lr_decay_epochs', default=[15])
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)

    parser.add_argument('--min', type=int, default=1, help='minimum age')
    parser.add_argument('--max', type=int, default=101, help='maximum age')
    parser.add_argument('--interval', type=int, default=1)
    config = parser.parse_args()


    shutil.rmtree(config.summary_write_path)
    summary = SummaryWriter(config.summary_write_path)
    ##converting .mat file into .csv file.. 맷랩 형식 귀찮아서 그냥 csv파일로 만들어서 갖고 옴
    # preprocess(config)
    # assert False
    train(config)
    # eval(config)