import sys
import cv2
import numpy as np
from torch import torch, utils
import yaml


class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class datagen(utils.data.Dataset):
    def __init__(self, file_ids, config, input_dir):
        self.file_ids = file_ids
        self.config = config
        self.input_dir = input_dir

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        file_id = self.file_ids[idx]

        ddir = self.input_dir

        #INPUT
        dvs_f = cv2.imread(ddir+self.config['input'][0]+"/"+file_id) / 255
        dvs_l = cv2.imread(ddir+self.config['input'][1]+"/"+file_id) / 255
        dvs_ri = cv2.imread(ddir+self.config['input'][2]+"/"+file_id) / 255
        dvs_r = cv2.imread(ddir+self.config['input'][3]+"/"+file_id) / 255
        dvs_f = np.delete(dvs_f.transpose(2, 0, 1), obj=1, axis=0)
        dvs_l = np.delete(dvs_l.transpose(2, 0, 1), obj=1, axis=0)
        dvs_ri = np.delete(dvs_ri.transpose(2, 0, 1), obj=1, axis=0)
        dvs_r = np.delete(dvs_r.transpose(2, 0, 1), obj=1, axis=0)

        bgr_f = cv2.imread(ddir+self.config['input'][4]+"/"+file_id) / 255
        bgr_l = cv2.imread(ddir+self.config['input'][5]+"/"+file_id) / 255
        bgr_ri = cv2.imread(ddir+self.config['input'][6]+"/"+file_id) / 255
        bgr_r = cv2.imread(ddir+self.config['input'][7]+"/"+file_id) / 255
        bgr_f = bgr_f.transpose(2, 0, 1)
        bgr_l = bgr_l.transpose(2, 0, 1)
        bgr_ri = bgr_ri.transpose(2, 0, 1)
        bgr_r = bgr_r.transpose(2, 0, 1)
        
        if self.config['tensor_dim'][1][2] == 1: 
            lid_t = cv2.imread(ddir+self.config['input'][8]+"/"+file_id, cv2.COLOR_BGR2GRAY) / 255
            lid_t = np.expand_dims(lid_t, axis=0)
        else: 
            lid_t = np.load(ddir+self.config['input'][8]+"/"+file_id[:-4]+"_128.npy")

        inp = (dvs_f, dvs_l, dvs_ri, dvs_r, bgr_f, bgr_l, bgr_ri, bgr_r, lid_t)
        

        #OUTPUT
        depth_f = cv2.imread(ddir+self.config['task'][0]+"/"+file_id, cv2.COLOR_BGR2GRAY) / 255
        depth_l = cv2.imread(ddir+self.config['task'][1]+"/"+file_id, cv2.COLOR_BGR2GRAY) / 255
        depth_ri = cv2.imread(ddir+self.config['task'][2]+"/"+file_id, cv2.COLOR_BGR2GRAY) / 255
        depth_r = cv2.imread(ddir+self.config['task'][3]+"/"+file_id, cv2.COLOR_BGR2GRAY) / 255
        depth_f = np.expand_dims(depth_f, axis=0)
        depth_l = np.expand_dims(depth_l, axis=0)
        depth_ri = np.expand_dims(depth_ri, axis=0)
        depth_r = np.expand_dims(depth_r, axis=0)
        
        segmentation_f = np.load(ddir+self.config['task'][4]+"/"+file_id[:-4]+"_128.npy")
        segmentation_l = np.load(ddir+self.config['task'][5]+"/"+file_id[:-4]+"_128.npy")
        segmentation_ri = np.load(ddir+self.config['task'][6]+"/"+file_id[:-4]+"_128.npy")
        segmentation_r = np.load(ddir+self.config['task'][7]+"/"+file_id[:-4]+"_128.npy")

        lid_seg = np.load(ddir+self.config['task'][8]+"/"+file_id[:-4]+"_128.npy")

        bird_view = np.load(ddir+self.config['task'][9]+"/"+file_id[:-4]+"_128.npy")
     
        out = (depth_f, depth_l, depth_ri, depth_r, segmentation_f, segmentation_l, segmentation_ri, segmentation_r, lid_seg, bird_view)

        
        return inp, out, {'img_id': file_id}



