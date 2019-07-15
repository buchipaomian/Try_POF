import torch
from torch.utils.data import Dataset
import numpy as np
import torch.utils.data.dataloader as DataLoader
#import tools
from torchvision import transforms
import os
import glob
from PIL import Image
import pandas as pd
import json
import datasets.PanoPicConvert as ppc
import random

"""
this is the function for convert the image with camera positions
"""
def projectPoints(X, K, R, t, Kd):
    """ Projects points X (3xN) using camera intrinsics K (3x3),
    extrinsics (R,t) and distortion parameters Kd=[k1,k2,p1,p2,k3].
    
    Roughly, x = K*(R*X + t) + distortion
    
    See http://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html
    or cv2.projectPoints
    """
    
    x = np.asarray(R*X + t)
    
    x[0:2,:] = x[0:2,:]/x[2,:]
    
    r = x[0,:]*x[0,:] + x[1,:]*x[1,:]
    
    x[0,:] = x[0,:]*(1 + Kd[0]*r + Kd[1]*r*r + Kd[4]*r*r*r) + 2*Kd[2]*x[0,:]*x[1,:] + Kd[3]*(r + 2*x[0,:]*x[0,:])
    x[1,:] = x[1,:]*(1 + Kd[0]*r + Kd[1]*r*r + Kd[4]*r*r*r) + 2*Kd[3]*x[0,:]*x[1,:] + Kd[2]*(r + 2*x[1,:]*x[1,:])

    x[0,:] = K[0,0]*x[0,:] + K[0,1]*x[1,:] + K[0,2]
    x[1,:] = K[1,0]*x[0,:] + K[1,1]*x[1,:] + K[1,2]
    
    return x

def cameramix(data,camera):
    """
    this funtion will convert the 3d model into a position which would point to the camera
    based on camera position.
    I don't know if this would work,but maybe this operation will come out with better result
    """
    result = projectPoints(data,np.matrix(camera["K"]),np.matrix(camera["R"]),np.array(camera["t"]).reshape((3,1)),np.array(camera["distCoef"]))
    return result

class PairedDataset(Dataset):
    """
    Composed of
    (origanal image,output json)
    """
    def __init__(self,root = './data/',mode = 'train',transform = None,):
        """
        @param root: data root
        @param mode: set mode (train, test, val)
        @param transform: Image Processing
        """
        available_angel = ['00_11','00_17','00_18']
        img_root = os.path.join(root,mode+'/image')#+'/00_18'
        json_root = os.path.join(root,mode+'/json_body')
        self.camera_root = os.path.join(root,mode+'/camera')
        #self.camera_file_name = glob.glob(camera_root+'/camera.json')
        self.is_train = (mode == 'train')
        self.transform = transform
        self.image_files = []
        for nums in available_angel:
            self.image_files += glob.glob(os.path.join(img_root,nums, '*.jpg'))
        #self.json_files = glob.glob(os.path.join(json_root, '*.json'))
        
        #here would random average the dataset
        self.image_files = random.sample(self.image_files,len(self.image_files)//len(available_angel))
        if len(self.image_files) == 0:
            # no png file, use jpg
            self.image_files = glob.glob(os.path.join(img_root, '*.png'))
    def __len__(self):
        return len(self.image_files)
    def __getitem__(self,index):
        """
        CMUPoseCapture Dataset Get Item
        @param index: index
        Returns:
            image
            paired json
        """
        if self.is_train:
            img_filename = self.image_files[index]
            #json_filename = self.json_files[index]
            json_filename = img_filename[0:12]+"/json_body/body3DScene_"+img_filename[31:39]+".json"
            #print(img_filename)
            #print(json_filename)
            camera_root = self.camera_root+"/"+img_filename[25:30]
            camera_file_name = glob.glob(camera_root+'/camera.json')
            camera_file_name = camera_file_name[0]
        else:
            img_filename = self.image_files[index]
            #json_filename = self.json_files[index]
            #json_filename = img_filename[0:10]+"/json_body/body3DScene_"+img_filename[17:25]+".json"
            #print(img_filename)
            #print(json_filename)
            #camera_file_name = self.camera_file_name[0]
            image = Image.open(img_filename)
            if self.transform is not None:
                image = self.transform(image)
            return image
        
        image = Image.open(img_filename)
        jsonfile = pd.read_json(json_filename)
        k = jsonfile.to_dict()
        try:
            joint = k["bodies"][0]["joints19"]
        except:
            #print(json_filename)
            return self.transform(image),-1,-1,-1
        #print(camera_file_name)
        with open(camera_file_name,'r') as jsonfile:
            camera = json.load(jsonfile)
        #joint = cameramix(joint,camera)
        joint_result = np.array(joint).reshape((-1,4)).transpose()
        if self.transform is not None:
            image = self.transform(image)
        joint_result = cameramix(joint_result[0:3,:],camera)
        heatmap,pof = ppc.get_target(joint_result.T,56)
        #print(target.shape)
        return image,1,heatmap,pof
