import csv
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os
import cv2
import numpy as np
from skimage.feature import greycomatrix
from sklearn.cluster import KMeans

class MoodModel(nn.Module):
    def __init__(self, num_classes):
        super(MoodModel, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.extra_fc = nn.Linear(2048, 256)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x = torch.relu(self.extra_fc(x))
        x = self.fc(x)
        return x

def extract_dominant_color(image):
    pixels = np.array(image).reshape((-1, 3))
    kmeans = KMeans(n_clusters=1)
    kmeans.fit(pixels)
    dominant_color = kmeans.cluster_centers_[0]
    return dominant_color / 255.0  # 归一化到 [0, 1]

def extract_color_combinations(image):
    pixels = np.array(image).reshape((-1, 3))
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(pixels)
    color_combinations = kmeans.cluster_centers_ / 255.0  # 归一化到 [0, 1]
    return color_combinations

def extract_shape_feature(image):
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray_image, threshold1=30, threshold2=100)
    shape_feature = np.sum(edges) / (edges.shape[0] * edges.shape[1])
    return shape_feature


def extract_texture_feature(image):
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    glcm = greycomatrix(gray_image, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)

    texture_feature = np.sum(glcm ** 2)  # 例如，可以使用GLCM的平方和作为纹理特征

    return texture_feature

def extract_contrast_feature(image):
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    contrast_feature = np.std(gray_image)
    return contrast_feature

def extract_saturation_feature(image):
    hsv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)
    saturation_feature = np.mean(hsv_image[:, :, 1] / 255.0)  # 归一化到 [0, 1]
    return saturation_feature

model = MoodModel(num_classes=3)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image_path = '/上的各种课/上课爬虫/豆瓣/douban'
afiles = os.listdir(image_path)
files = [file for file in afiles if file.lower().endswith('.jpg')]
with open('features.csv', 'w') as fp:
    writer=csv.writer(fp)
    writer.writerow(['imgid','mood','main_color','color_matrix',
                     'shape','texture','contrast_ratio','saturation'])

    for file in files:
        image_path = os.path.join('/上的各种课/上课爬虫/豆瓣/douban')
        image_path = os.path.join('/上的各种课/上课爬虫/豆瓣/douban', file)
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image)
        image_tensor = torch.unsqueeze(image_tensor, 0)

        with torch.no_grad():
            outputs = model(image_tensor)


        _, predicted = torch.max(outputs, 1)
        emotion_classes = ['-1', '0', '1']
        predicted_emotion = emotion_classes[predicted.item()]

        dominant_color = extract_dominant_color(image)
        color_combinations = extract_color_combinations(image)
        shape_feature = extract_shape_feature(image)
        texture_feature = extract_texture_feature(image)
        contrast_feature = extract_contrast_feature(image)
        saturation_feature = extract_saturation_feature(image)
        writer.writerow([file, predicted_emotion, dominant_color.tolist(), color_combinations.tolist(),
                         shape_feature, texture_feature, contrast_feature, saturation_feature])
        print(file.index)
