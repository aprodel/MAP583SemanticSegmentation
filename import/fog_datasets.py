import torch
import torch.utils.data
from PIL import Image

import numpy as np
import cv2
import os

train_dirs = ["jena/", "zurich/", "weimar/", "ulm/", "tubingen/", "stuttgart/",
              "strasbourg/", "monchengladbach/", "krefeld/", "hanover/",
              "hamburg/", "erfurt/", "dusseldorf/", "darmstadt/", "cologne/",
              "bremen/", "bochum/", "aachen/"]
val_dirs = ["frankfurt/", "munster/", "lindau/"]
test_dirs = ["berlin", "bielefeld", "bonn", "leverkusen", "mainz", "munich"]

class DatasetTrain_fog(torch.utils.data.Dataset):
    def __init__(self, cityscapes_data_path):
        self.img_dir = cityscapes_data_path + "/leftImg8bit/train/"
        self.label_dir = cityscapes_data_path + "/leftImg8bit/train/"

        self.img_h = 512
        self.img_w = 1024

        self.examples = []
        for train_dir in train_dirs:
            train_img_dir_path = self.img_dir + train_dir

            file_names = os.listdir(train_img_dir_path)
            for file_name in file_names:
                img_id = file_name.split("_leftImg8bit.png")[0]

                img_path = train_img_dir_path + file_name

                label_img_path = train_img_dir_path + file_name

                example = {}
                example["img_path"] = img_path
                example["label_img_path"] = label_img_path
                example["img_id"] = img_id
                self.examples.append(example)

        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        img_id=example["img_id"]
        img_path = example["img_path"]
        img = cv2.imread(img_path, -1)

        label_img_path = example["label_img_path"]
        label_img = cv2.imread(label_img_path, -1)

        img = img/255.0
        img = img - np.array([0.485, 0.456, 0.406])
        img = img/np.array([0.229, 0.224, 0.225])
        img = np.transpose(img, (2, 0, 1))
        img = img.astype(np.float32)

        img=np.asarray(img)
        img = torch.from_numpy(img)
        label_img=np.asarray(label_img)
        label_img = torch.from_numpy(label_img)

        return (img, label_img,img_id)

    def __len__(self):
        return self.num_examples

class DatasetVal_fog(torch.utils.data.Dataset):
    def __init__(self, cityscapes_data_path):
        self.img_dir = cityscapes_data_path + "/leftImg8bit/val/"
        self.label_dir = cityscapes_data_path + "/leftImg8bit/val/"

        self.img_h = 512
        self.img_w = 1024

        self.examples = []
        for val_dir in val_dirs:
            val_img_dir_path = self.img_dir + val_dir

            file_names = os.listdir(val_img_dir_path)
            for file_name in file_names:
                img_id = file_name.split("_leftImg8bit.png")[0]

                img_path = val_img_dir_path + file_name

                label_img_path = val_img_dir_path + file_name
                label_img = cv2.imread(label_img_path, -1)

                example = {}
                example["img_path"] = img_path
                example["label_img_path"] = label_img_path
                example["img_id"] = img_id
                self.examples.append(example)

        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        img_id = example["img_id"]

        img_path = example["img_path"]
        img = cv2.imread(img_path, -1)
        
        label_img_path = example["label_img_path"]
        label_img = cv2.imread(label_img_path, -1)

        
        img = img/255.0
        img = img - np.array([0.485, 0.456, 0.406])
        img = img/np.array([0.229, 0.224, 0.225])
        img = np.transpose(img, (2, 0, 1))
        img = img.astype(np.float32)
        
        img = torch.from_numpy(img)
        label_img = torch.from_numpy(label_img)
        
        return (img, label_img,img_id)

    def __len__(self):
        return self.num_examples

