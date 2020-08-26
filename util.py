import numpy as np
from scipy.io import loadmat
import pandas as pd
import datetime as date
from dateutil.relativedelta import relativedelta
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import torch
import math

def visualize_cam2(mask, img):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask.squeeze()), cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
    b, g, r = heatmap.split(1)
    # zero = torch.zeros((1,224,224))
    heatmap = torch.cat([r, r, r])
    img = img.squeeze(0)
    # result = heatmap + img.cpu()
    # result = result.div(result.max()).squeeze()
    img = torch.tensor(np.where(heatmap<0.8, img.cpu(), 0))
    return img

def visualize_cam(mask, img):
    """Make heatmap from mask and synthesize GradCAM result image using heatmap and img.
    Args:
        mask (torch.tensor): mask shape of (1, 1, H, W) and each element has value in range [0, 1]
        img (torch.tensor): img shape of (1, 3, H, W) and each pixel value is in range [0, 1]

    Return:
        heatmap (torch.tensor): heatmap img shape of (3, H, W)
        result (torch.tensor): synthesized GradCAM result of same shape with heatmap.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask.squeeze()), cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b])
    # zero = torch.zeros((1,224,224))

    result = heatmap + img.cpu()
    result = result.div(result.max()).squeeze()

    return heatmap, result.squeeze(0)


def find_resnet_layer(arch, target_layer_name):
    """Find resnet layer to calculate GradCAM and GradCAM++

    Args:
        arch: default torchvision densenet models
        target_layer_name (str): the name of layer with its hierarchical information. please refer to usages below.
            target_layer_name = 'conv1'
            target_layer_name = 'layer1'
            target_layer_name = 'layer1_basicblock0'
            target_layer_name = 'layer1_basicblock0_relu'
            target_layer_name = 'layer1_bottleneck0'
            target_layer_name = 'layer1_bottleneck0_conv1'
            target_layer_name = 'layer1_bottleneck0_downsample'
            target_layer_name = 'layer1_bottleneck0_downsample_0'
            target_layer_name = 'avgpool'
            target_layer_name = 'fc'

    Return:
        target_layer: found layer. this layer will be hooked to get forward/backward pass information.
    """
    if 'layer' in target_layer_name:
        hierarchy = target_layer_name.split('_')
        layer_num = int(hierarchy[0].lstrip('layer'))
        if layer_num == 1:
            target_layer = arch.layer1
        elif layer_num == 2:
            target_layer = arch.layer2
        elif layer_num == 3:
            target_layer = arch.layer3
        elif layer_num == 4:
            target_layer = arch.layer4
        else:
            raise ValueError('unknown layer : {}'.format(target_layer_name))

        if len(hierarchy) >= 2:
            bottleneck_num = int(hierarchy[1].lower().lstrip('bottleneck').lstrip('basicblock'))
            target_layer = target_layer[bottleneck_num]

        if len(hierarchy) >= 3:
            target_layer = target_layer._modules[hierarchy[2]]

        if len(hierarchy) == 4:
            target_layer = target_layer._modules[hierarchy[3]]

    else:
        target_layer = arch._modules[target_layer_name]

    return target_layer


def find_densenet_layer(arch, target_layer_name):
    """Find densenet layer to calculate GradCAM and GradCAM++

    Args:
        arch: default torchvision densenet models
        target_layer_name (str): the name of layer with its hierarchical information. please refer to usages below.
            target_layer_name = 'features'
            target_layer_name = 'features_transition1'
            target_layer_name = 'features_transition1_norm'
            target_layer_name = 'features_denseblock2_denselayer12'
            target_layer_name = 'features_denseblock2_denselayer12_norm1'
            target_layer_name = 'features_denseblock2_denselayer12_norm1'
            target_layer_name = 'classifier'

    Return:
        target_layer: found layer. this layer will be hooked to get forward/backward pass information.
    """

    hierarchy = target_layer_name.split('_')
    target_layer = arch._modules[hierarchy[0]]

    if len(hierarchy) >= 2:
        target_layer = target_layer._modules[hierarchy[1]]

    if len(hierarchy) >= 3:
        target_layer = target_layer._modules[hierarchy[2]]

    if len(hierarchy) == 4:
        target_layer = target_layer._modules[hierarchy[3]]

    return target_layer


def find_vgg_layer(arch, target_layer_name):
    """Find vgg layer to calculate GradCAM and GradCAM++

    Args:
        arch: default torchvision densenet models
        target_layer_name (str): the name of layer with its hierarchical information. please refer to usages below.
            target_layer_name = 'features'
            target_layer_name = 'features_42'
            target_layer_name = 'classifier'
            target_layer_name = 'classifier_0'

    Return:
        target_layer: found layer. this layer will be hooked to get forward/backward pass information.
    """
    hierarchy = target_layer_name.split('_')

    if len(hierarchy) >= 1:
        target_layer = arch.features

    if len(hierarchy) == 2:
        target_layer = target_layer[int(hierarchy[1])]

    return target_layer


def find_alexnet_layer(arch, target_layer_name):
    """Find alexnet layer to calculate GradCAM and GradCAM++

    Args:
        arch: default torchvision densenet models
        target_layer_name (str): the name of layer with its hierarchical information. please refer to usages below.
            target_layer_name = 'features'
            target_layer_name = 'features_0'
            target_layer_name = 'classifier'
            target_layer_name = 'classifier_0'

    Return:
        target_layer: found layer. this layer will be hooked to get forward/backward pass information.
    """
    hierarchy = target_layer_name.split('_')

    if len(hierarchy) >= 1:
        target_layer = arch.features

    if len(hierarchy) == 2:
        target_layer = target_layer[int(hierarchy[1])]

    return target_layer


def find_squeezenet_layer(arch, target_layer_name):
    """Find squeezenet layer to calculate GradCAM and GradCAM++

    Args:
        arch: default torchvision densenet models
        target_layer_name (str): the name of layer with its hierarchical information. please refer to usages below.
            target_layer_name = 'features_12'
            target_layer_name = 'features_12_expand3x3'
            target_layer_name = 'features_12_expand3x3_activation'

    Return:
        target_layer: found layer. this layer will be hooked to get forward/backward pass information.
    """
    hierarchy = target_layer_name.split('_')
    target_layer = arch._modules[hierarchy[0]]

    if len(hierarchy) >= 2:
        target_layer = target_layer._modules[hierarchy[1]]

    if len(hierarchy) == 3:
        target_layer = target_layer._modules[hierarchy[2]]

    elif len(hierarchy) == 4:
        target_layer = target_layer._modules[hierarchy[2] + '_' + hierarchy[3]]

    return target_layer


def denormalize(tensor, mean, std):
    if not tensor.ndimension() == 4:
        raise TypeError('tensor should be 4D')

    mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)

    return tensor.mul(std).add(mean)


def normalize(tensor, mean, std):
    if not tensor.ndimension() == 4:
        raise TypeError('tensor should be 4D')

    mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)

    return tensor.sub(mean).div(std)


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return self.do(tensor)

    def do(self, tensor):
        return normalize(tensor, self.mean, self.std)

    def undo(self, tensor):
        return denormalize(tensor, self.mean, self.std)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def preprocess(config=None):
    print("=============Start making csv file...=================")
    cols = ['age', 'gender', 'path', 'face_score1', 'face_score2']

    if config.dataset=='imdb_crop':
        imdb_mat = 'data/imdb_crop/imdb.mat'
        imdb_data = loadmat(imdb_mat)
        del imdb_mat
        imdb = imdb_data['imdb']
        imdb_photo_taken = imdb[0][0][1][0]
        imdb_full_path = imdb[0][0][2][0]
        imdb_gender = imdb[0][0][3][0]
        imdb_face_score1 = imdb[0][0][6][0]
        imdb_face_score2 = imdb[0][0][7][0]

        imdb_path = []
        for path in imdb_full_path:
            imdb_path.append('data/imdb_crop/' + path[0])

        imdb_genders = []
        for n in range(len(imdb_gender)):
            if imdb_gender[n] == 1:
                imdb_genders.append('male')
            else:
                imdb_genders.append('female')

        imdb_dob = []
        for file in imdb_path:
            temp = file.split('_')[3]
            temp = temp.split('-')
            if len(temp[1]) == 1:
                temp[1] = '0' + temp[1]
            if len(temp[2]) == 1:
                temp[2] = '0' + temp[2]

            if temp[1] == '00':
                temp[1] = '01'
            if temp[2] == '00':
                temp[2] = '01'

            imdb_dob.append('-'.join(temp))

        imdb_age = []
        for i in range(len(imdb_dob)):
            try:
                d1 = date.datetime.strptime(imdb_dob[i][0:10], '%Y-%m-%d')
                d2 = date.datetime.strptime(str(imdb_photo_taken[i]), '%Y')
                rdelta = relativedelta(d2, d1)
                diff = rdelta.years
            except Exception as ex:
                # print(ex)
                diff = -1
            imdb_age.append(diff)

        final_imdb = np.vstack((imdb_age, imdb_genders, imdb_path, imdb_face_score1, imdb_face_score2)).T
        final_imdb_df = pd.DataFrame(final_imdb)
        final_imdb_df.columns = cols

        meta = final_imdb_df
        meta = meta[meta['face_score1'] != '-inf']
        meta = meta[meta['face_score2'] == 'nan']

        meta = meta.drop(['face_score1', 'face_score2'], axis=1)

        meta = meta.sample(frac=1)

        data_len = len(meta)
        train_meta = meta[:int(data_len * 0.7)]
        test_meta = meta[int(data_len * 0.7):]

        train_meta.to_csv(config.train_csv_path, index=False)
        test_meta.to_csv(config.test_csv_path, index=False)

    elif config.dataset == 'wiki_crop':
        wiki_mat = 'data/wiki_crop/wiki.mat'


        wiki_data = loadmat(wiki_mat)


        del wiki_mat


        wiki = wiki_data['wiki']



        wiki_photo_taken = wiki[0][0][1][0]
        wiki_full_path = wiki[0][0][2][0]
        wiki_gender = wiki[0][0][3][0]
        wiki_face_score1 = wiki[0][0][6][0]
        wiki_face_score2 = wiki[0][0][7][0]


        wiki_path = []



        for path in wiki_full_path:
            wiki_path.append('data/wiki_crop/' + path[0])


        wiki_genders = []



        for n in range(len(wiki_gender)):
            if wiki_gender[n] == 1:
                wiki_genders.append('male')
            else:
                wiki_genders.append('female')


        wiki_dob = []



        for file in wiki_path:
            wiki_dob.append(file.split('_')[2])


        wiki_age = []



        for i in range(len(wiki_dob)):
            try:
                d1 = date.datetime.strptime(wiki_dob[i][0:10], '%Y-%m-%d')
                d2 = date.datetime.strptime(str(wiki_photo_taken[i]), '%Y')
                rdelta = relativedelta(d2, d1)
                diff = rdelta.years
            except Exception as ex:
                print(ex)
                diff = -1
            wiki_age.append(diff)


        final_wiki = np.vstack((wiki_age, wiki_genders, wiki_path, wiki_face_score1, wiki_face_score2)).T


        final_wiki_df = pd.DataFrame(final_wiki)


        final_wiki_df.columns = cols

        meta = final_wiki_df
        meta = meta[meta['face_score1'] != '-inf']
        meta = meta[meta['face_score2'] == 'nan']

        meta = meta.drop(['face_score1', 'face_score2'], axis=1)

        meta = meta.sample(frac=1)

        data_len = len(meta)
        train_meta = meta[:int(data_len * 0.7)]
        test_meta = meta[int(data_len * 0.7):]

        train_meta.to_csv(config.train_csv_path, index=False)
        test_meta.to_csv(config.test_csv_path, index=False)



    print("===============Finish making....==========")


def image_transformer():
  """
  :return:  A transformer to convert a PIL image to a tensor image
            ready to feed into a neural network
  """
  return {
      'train': transforms.Compose([
        transforms.Resize((256, 256), interpolation=2),
          transforms.RandomHorizontalFlip(),
          transforms.RandomCrop(224),
          transforms.RandomGrayscale(),
          transforms.RandomRotation([30, 60]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
      'val': transforms.Compose([
        transforms.Resize((256, 256), interpolation=2),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
    }



class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=None, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss



class MeanVarianceLoss(nn.Module):
    def __init__(self, lambda_1, lambda_2, start_age, end_age):
        super().__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.start_age = start_age
        self.end_age = end_age

    def forward(self, input, target):
        N = input.size()[0]
        target = target.type(torch.FloatTensor).cuda()
        m = nn.Softmax(dim=1)
        p = m(input)
        # mean loss
        a = torch.arange(self.start_age, self.end_age, 10, dtype=torch.float32).cuda()
        mean = torch.squeeze((p * a).sum(1, keepdim=True), dim=1)
        mse = (mean - target) ** 2
        mean_loss = mse.mean() / 2.0

        # variance loss
        b = (a[None, :] - mean[:, None]) ** 2
        variance_loss = (p * b).sum(1, keepdim=True).mean()

        return self.lambda_1 * mean_loss, self.lambda_2 * variance_loss