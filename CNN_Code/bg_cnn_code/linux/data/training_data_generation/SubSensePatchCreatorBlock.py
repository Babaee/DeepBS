import numpy as np
import matplotlib.image as mpimg
import scipy
from scipy import ndimage
from scipy import misc
import argparse, os
import matplotlib.pyplot as plt
from skimage.color.colorconv import rgb2gray
from math import floor
from PIL.ImageOps import grayscale
import h5py
import random


class PatchCreatorBlock:
    
    def __init__(self, sequence,patchSize, size, stride, grayScale=True):
        assert(patchSize % 2 != 0), 'patchSize should be odd!'
               
        self.sequences = sequence # dictionary of sequences
        self.bg_images_color = dict()
        self.bg_images = dict()
        self.ROI    = dict()
        # hack with absolute paths
        self.bg_root = '/usr/home/rez/ZM/BG_Model1' #TODO
        self.data_root = '/usr/home/rez/ZM/CDNet_Dataset/dataset' #TODO

        self.pSize = patchSize
        self.grayScale = grayScale
        self.stride = stride
        self.height, self.width = size[0], size[1]
        self.x_max = (self.width - pSize) / self.stride
        self.y_max = (self.height - pSize) / self.stride

        # generate color and grayscale background images
        for id in self.sequences.keys():
            self.ROI.update({id: misc.imresize(np.array(rgb2gray(mpimg.imread(self.sequences[id]['root'] + '/ROI.png', np.uint8)) * 255, np.uint8), [self.height, self.width], 'nearest')} )

    def generateBgImage(self, start, end, sequence_id):
        
        im_sequence = np.zeros(( self.height, self.width, 3, end - start + 1), np.uint8)
        for idx in range(start, end + 1):
            im_sequence[:, :, :, idx - start] = misc.imresize(mpimg.imread(self.sequences[sequence_id]['root'] + '/input/in%06d.jpg' % (idx), np.uint8),\
                                                              [self.height, self.width, 3], 'bilinear') # read sequence image and store in tensor
        im_sequence = np.uint8(np.median(im_sequence, axis=3)) # true: overwrites input
        return im_sequence
        
    def createPatchXY(self, im, gt, bg, bg_id, x, y):
        # builds a patch that consists of bg, input image, assume that image is already padded
        gt_treshold = 255
        ignore_val = 85
    
        x_ = x + self.pSize
        y_ = y + self.pSize
        
        in_patch = im[y:y_, x:x_]
        bg_patch = bg[y:y_, x:x_]
        gt_patch = gt[y:y_, x:x_]
        
        if self.grayScale:
            in_patch = im[y:y_, x:x_]
            bg_patch = bg[y:y_, x:x_]
            patch_tensor = np.zeros((2, self.pSize, self.pSize, 1), np.uint8)
            patch_tensor[0,:,:,0] = in_patch
            patch_tensor[1,:,:,0] = bg_patch
        else:
            in_patch = im[y:y_, x:x_,:]
            bg_patch = bg[y:y_, x:x_,:]
            patch_tensor = np.zeros((2, self.pSize, self.pSize, 3), np.uint8)
            patch_tensor[0,:,:,:] = in_patch
            patch_tensor[1,:,:,:] = bg_patch

        
        roi_max_y = np.max((self.ROI[bg_id].shape[0] - 1, y_))
        roi_max_x = np.max((self.ROI[bg_id].shape[1] - 1, x_))
        inRoi = np.any(self.ROI[bg_id][y:roi_max_y , x:roi_max_x] == 255)

        gt_patch = np.bitwise_and(gt_patch > 20, gt_patch < 200)*2 + (gt_patch>240) # ignore hard shadows (all gray areas in gt)
        
        return patch_tensor, np.array((gt_patch), np.uint8), inRoi
    

    def im2patches(self, id):
        # extracts all patches from image and saves in destination folder
        ims = dict()
        gts = dict()
        bgs = dict()
        self.patchTensors = dict()
        self.gtTensors = dict()
        y_max = (self.height - self.pSize) / self.stride
        x_max = (self.width - self.pSize) / self.stride
        Y = range(0, y_max * self.stride + 1, self.stride)
        X = range(0, x_max * self.stride + 1, self.stride)
        n_patches = len(Y) * len(X)
    
        for k in self.sequences.keys():
        
            image = self.sequences[k]['root']  + '/input/in%06d.jpg' % (self.sequences[k]['start'] + id)
            data_root = self.sequences[k]['root'] 
            bg_im = data_root.replace(self.data_root, self.bg_root) + '/in%06d.jpg' % (self.sequences[k]['start'] + id)
            ground_truth = self.sequences[k]['root']  + '/groundtruth/gt%06d.png' % (self.sequences[k]['start'] + id)       
            
    
            # read image and prepare patch tensor
            if self.grayScale:
                ims[k] = misc.imresize(np.array(rgb2gray(mpimg.imread(image))*255,np.uint8), [self.height, self.width], 'bilinear')
                bgs[k] = misc.imresize(np.array(rgb2gray(mpimg.imread(bg_im))*255,np.uint8), [self.height, self.width], 'bilinear')
                patchContainer = np.zeros((1, 2, self.pSize, self.pSize, 1), np.uint8)
                
                
            else:
                ims[k] = misc.imresize(mpimg.imread(image, np.uint8), [self.height, self.width], 'bilinear')
                bgs[k] = misc.imresize(mpimg.imread(bg_im, np.uint8), [self.height, self.width], 'bilinear')
                patchContainer = np.zeros((1, 2, self.pSize, self.pSize, 3), np.uint8)
                
                
            gtContainer = np.zeros((1,self.pSize, self.pSize), np.uint8)
            # prepare groundtruth to label tensor
            gts[k] =  misc.imresize(np.array(rgb2gray(mpimg.imread(ground_truth, np.uint8))*255, np.uint8), [self.height, self.width], 'bilinear')

            for y in range(0, y_max * self.stride + 1, self.stride):
                for x in range(0, x_max * self.stride + 1, self.stride):
                    patchContainer[0,:,:,:,:], gtContainer[0,:,:], inROI = self.createPatchXY(ims[k], gts[k], bgs[k], k, x, y)        
                    if inROI:
                        self.patchTensors[k] = np.vstack([self.patchTensors[k], patchContainer]) if k in self.patchTensors.keys() else patchContainer
                        self.gtTensors[k] = np.vstack([self.gtTensors[k], gtContainer]) if k in self.gtTensors.keys() else gtContainer

    
    def savePatchTensor(self, destination, id):
        h5_out = destination + '/data%06d.h5' % (id)
        
        pt = np.array([])
        gt = np.array([])
        
        for k in self.sequences.keys():
            # stack patch and gt tensors
            if k in self.patchTensors.keys():
                pt = np.vstack([pt, self.patchTensors[k]]) if pt.size else self.patchTensors[k]
                gt = np.vstack([gt, self.gtTensors[k]]) if gt.size else self.gtTensors[k]
        

        with h5py.File(h5_out, 'w') as f:
            print 'writing file to ' + h5_out
            f.create_dataset('patches', data=pt, dtype=np.uint8)
            f.create_dataset('labels', data=gt, dtype=np.uint8)
            
    def getCategoryVideoFromId(self,id):
        
        splitted_path = self.sequences[id]['root'].split('/')
        category, video = splitted_path[-2], splitted_path[-1]
        return category,video


if __name__ == "__main__":
    
    sequences = []

    # baseline
    sequences.append({'root' : '/usr/home/rez/ZM/CDNet_Dataset/dataset/baseline/highway', 'start': 790})
    sequences.append({'root' : '/usr/home/rez/ZM/CDNet_Dataset/dataset/baseline/office', 'start': 580})
    sequences.append({'root' : '/usr/home/rez/ZM/CDNet_Dataset/dataset/baseline/pedestrians', 'start': 422})
    sequences.append({'root' : '/usr/home/rez/ZM/CDNet_Dataset/dataset/baseline/PETS2006', 'start': 420 })
    
    # dynamic background
    sequences.append({'root' : '/usr/home/rez/ZM/CDNet_Dataset/dataset/dynamicBackground/canoe', 'start': 900})
    sequences.append({'root' : '/usr/home/rez/ZM/CDNet_Dataset/dataset/dynamicBackground/fall', 'start': 1465})
    sequences.append({'root' : '/usr/home/rez/ZM/CDNet_Dataset/dataset/dynamicBackground/fountain02', 'start': 657})
    sequences.append({'root' : '/usr/home/rez/ZM/CDNet_Dataset/dataset/dynamicBackground/fountain01', 'start': 702})   
    sequences.append({'root' : '/usr/home/rez/ZM/CDNet_Dataset/dataset/dynamicBackground/overpass', 'start': 2600})       
    sequences.append({'root' : '/usr/home/rez/ZM/CDNet_Dataset/dataset/dynamicBackground/boats', 'start': 7600})
    
    
    # bad weather
    sequences.append({'root' : '/usr/home/rez/ZM/CDNet_Dataset/dataset/badWeather/blizzard', 'start': 1180})
    sequences.append({'root' : '/usr/home/rez/ZM/CDNet_Dataset/dataset/badWeather/skating', 'start': 850})    
    sequences.append({'root' : '/usr/home/rez/ZM/CDNet_Dataset/dataset/badWeather/wetSnow', 'start': 520})        
    sequences.append({'root' : '/usr/home/rez/ZM/CDNet_Dataset/dataset/badWeather/snowFall', 'start': 1300})            
    
    
    # nightvideo
    sequences.append({'root' : '/usr/home/rez/ZM/CDNet_Dataset/dataset/nightVideos/streetCornerAtNight', 'start': 1220})
    sequences.append({'root': '/usr/home/rez/ZM/CDNet_Dataset/dataset/nightVideos/bridgeEntry', 'start' : 1550})
    sequences.append({'root' : '/usr/home/rez/ZM/CDNet_Dataset/dataset/nightVideos/winterStreet', 'start': 940})
    sequences.append({'root': '/usr/home/rez/ZM/CDNet_Dataset/dataset/nightVideos/tramStation', 'start' : 1462})
    sequences.append({'root' : '/usr/home/rez/ZM/CDNet_Dataset/dataset/nightVideos/fluidHighway', 'start': 586})
    sequences.append({'root': '/usr/home/rez/ZM/CDNet_Dataset/dataset/nightVideos/busyBoulvard', 'start' : 1141})    
    
    
    # thermal
    sequences.append({'root' : '/usr/home/rez/ZM/CDNet_Dataset/dataset/thermal/corridor', 'start': 1616})
    sequences.append({'root' : '/usr/home/rez/ZM/CDNet_Dataset/dataset/thermal/diningRoom', 'start': 800})
    sequences.append({'root' : '/usr/home/rez/ZM/CDNet_Dataset/dataset/thermal/park', 'start': 260})
    #sequences.append({'root' : '/usr/home/rez/ZM/CDNet_Dataset/dataset/thermal/library', 'start': 990})
    sequences.append({'root' : '/usr/home/rez/ZM/CDNet_Dataset/dataset/thermal/lakeSide', 'start': 1734})
     
    
    # lowframerate
    sequences.append({'root' : '/usr/home/rez/ZM/CDNet_Dataset/dataset/lowFramerate/port_0_17fps', 'start': 1135})
    sequences.append({'root' : '/usr/home/rez/ZM/CDNet_Dataset/dataset/lowFramerate/tramCrossroad_1fps', 'start': 401})
    sequences.append({'root' : '/usr/home/rez/ZM/CDNet_Dataset/dataset/lowFramerate/tunnelExit_0_35fps', 'start': 2170})
    sequences.append({'root' : '/usr/home/rez/ZM/CDNet_Dataset/dataset/lowFramerate/turnpike_0_5fps', 'start': 850})
    

    # hard shadows
    sequences.append({'root' : '/usr/home/rez/ZM/CDNet_Dataset/dataset/shadow/cubicle', 'start': 6400})
    sequences.append({'root' : '/usr/home/rez/ZM/CDNet_Dataset/dataset/shadow/copyMachine', 'start': 1000})
    sequences.append({'root' : '/usr/home/rez/ZM/CDNet_Dataset/dataset/shadow/busStation', 'start': 366})
    sequences.append({'root' : '/usr/home/rez/ZM/CDNet_Dataset/dataset/shadow/bungalows', 'start': 915})
    sequences.append({'root' : '/usr/home/rez/ZM/CDNet_Dataset/dataset/shadow/backdoor', 'start': 1800})
    sequences.append({'root' : '/usr/home/rez/ZM/CDNet_Dataset/dataset/shadow/peopleInShade', 'start': 287})
    
    # intermittent
    sequences.append({'root' : '/usr/home/rez/ZM/CDNet_Dataset/dataset/intermittentObjectMotion/sofa', 'start': 800})
    
    
    # turbulence
    sequences.append({'root' : '/usr/home/rez/ZM/CDNet_Dataset/dataset/turbulence/turbulence0', 'start': 2300})
    sequences.append({'root' : '/usr/home/rez/ZM/CDNet_Dataset/dataset/turbulence/turbulence1', 'start': 1600})
    sequences.append({'root' : '/usr/home/rez/ZM/CDNet_Dataset/dataset/turbulence/turbulence2', 'start': 615})
    sequences.append({'root' : '/usr/home/rez/ZM/CDNet_Dataset/dataset/turbulence/turbulence3', 'start': 1114})


    # cameraJitter
    sequences.append({'root' : '/usr/home/rez/ZM/CDNet_Dataset/dataset/cameraJitter/badminton', 'start': 855})
    sequences.append({'root' : '/usr/home/rez/ZM/CDNet_Dataset/dataset/cameraJitter/boulevard', 'start': 810})
    sequences.append({'root' : '/usr/home/rez/ZM/CDNet_Dataset/dataset/cameraJitter/sidewalk', 'start': 800})
    sequences.append({'root' : '/usr/home/rez/ZM/CDNet_Dataset/dataset/cameraJitter/traffic', 'start': 954})

    sequences = dict(enumerate(sequences))
    
    output = '/usr/home/rez/ZM/CNN/Up_Proj'
    size = (240, 320)
    N = 170
    train_length = 150
    stride = 10
    pSize = 37
    pg = PatchCreatorBlock(sequences ,pSize, size, stride, grayScale=False)


    for id in range(N):
        pg.im2patches(id)
        if id < train_length:
            pg.savePatchTensor(output + '/train', id)
        else:
            pg.savePatchTensor(output + '/val', id)

