#機械学習に必要な関数

import os
from torchvision import transforms
import torch
import torch.nn as nn

#クラス名とインデックスのラベル付け
class_to_idx = {
#ここに学習する画像フォルダのフォルダ目にラベルをつける
}

# クラス名とインデックスのラベル付け

#画像のロード

def load_image(image_path):
    from PIL import Image

    return Image.open(image_path).convert("RGB")

#画像のリサイズ

def resize_image(image, size):
    return image.resize(size)

#画像の前処理

def preprocess_image(image, size):
    resize_transform = transforms.Resize(size)
    image = resize_transform(image)

    tensor_transform = transforms.ToTensor()
    image = tensor_transform(image)

    normalize_transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    image = normalize_transform(image)
    return image

#画像の前処理を行うパイプライン

def image_preprocessing(image_path, label):
    image = load_image(image_path)
    preprocessed_image = preprocess_image(image, (224, 224))
    label = class_to_idx[label]
    return preprocessed_image, label


#画像を走査して学習用データを作る


def make_data(source_folder):

    makedata = []
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            data = image_preprocessing(os.path.join(root,file),os.path.basename(root))
            makedata.append(data)

    return makedata

#3ブロックのCNN

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        #特徴抽出
        #入力3,出力32,カーネルサイズ3,パディング1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)


        #特徴抽出
        #入力32,出力64,カーネルサイズ3,パディング1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        #特徴抽出
        #入力64,出力128,カーネルサイズ3,パディング1
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        #プーリング
        #カーネルサイズ2,ストライド2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        #全結合層

        #入力　128 * 28 * 28,出力128
        self.fc1 = nn.Linear(128 * 28 * 28, 128)

        #入力128,出力10
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        #特徴抽出とプーリング
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))

        size = x.size(0)
        x = x.view(size,-1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
