import torch
from torchvision import transforms
from PIL import Image
import numpy
import cv2
from scipy.ndimage.filters import gaussian_filter,maximum_filter
from scipy.ndimage.morphology import generate_binary_structure
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
plt.rcParams['image.interpolation'] = 'nearest'
from mpl_toolkits.mplot3d import Axes3D


joint_num = 19
linenum = [[17,15],[15,1],[1,0],[0,2],[0,3],[3,4],[4,5],[2,6],[6,7],[7,8],[1,16],[16,18],[0,9],[9,10],[10,11],[2,12],[12,13],[13,14]]#this is about the pof

def restorenet(img):
    transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])
    img = transform(img)
    img = img.unsqueeze(0).to(torch.device('cuda:3' if torch.cuda.is_available() else 'cpu'))
    net = torch.load('multiepoch12.pkl')
    heatmap,pof = net(img)
    return heatmap[0],pof[0]

def find_peaks(param,img):
    peak_binary = (maximum_filter(img,footprint = generate_binary_structure(2,1))==img)*(img>param['thre1'])
    return np.array(np.nonzero(peak_binary)[::-1]).T

def compute_resized_coords(coords,resizeFactor):
    return (np.array(coords,dtype=float)+0.5)*resizeFactor - 0.5

def generate_heatmap_joint(img,upsamFactor = 1):
    """
    img should be origanal size
    subset: n*20 array
    heatmap = [index][y][x]
    pof = [index][y][x]
    """
    heatmap,pof = restorenet(img)
    stickwidth = 4#not sure this is a relative param or just diy setting
    #linenum = [[17,15],[15,1],[1,0],[0,2],[0,3],[3,4],[4,5],[2,6],[6,7],[7,8],[18,16],[16,1],[0,9],[9,10],[10,11],[2,12],[12,13],[13,14]]#this is about the pof
    boxsize = 224
    stride = 4
    
    img_num = np.array(img)
    
    heatmap_avg = np.zeros((img_num.shape[0],img_num.shape[1],20))
    pof_avg = np.zeros((img_num.shape[0],img_num.shape[1],54))
    
    x_resize = img_num.shape[0] / boxsize
    y_resize = img_num.shape[1] / boxsize
    z_resize = min(x_resize,y_resize)
    
    img_input = img.resize((boxsize,boxsize),Image.ANTIALIAS)
    
    heatmaps,pofs = restorenet(img_input)
    
    heatmaps = heatmaps.cpu().detach().numpy()#this is 20*56*56
    pofs = pofs.cpu().detach().numpy()#this is 54*56*56
    
    shadow = 2#this means the point shadow should be 5*5
    upsamFactor = 224//56#this is the ratio between heatmap and input image
    param = {'thre1':0.1,'thre2':0.05,'thre3':0.5}#this is the background of heatmap and pof,which would let the counter faster
    
    joint_list_per_joint_type = []
    cnt_total_joints = 0
    for joint in range(joint_num):
        map_orig = heatmaps[joint,:,:]
        peak_coords = find_peaks(param,map_orig)
        peaks = np.zeros((len(peak_coords),4))
        for i,peak in enumerate(peak_coords):
            x_min,y_min = np.maximum(0,peak-shadow)
            x_max,y_max = np.minimum(np.array(map_orig.T.shape)-1,peak+shadow)
            
            patch = map_orig[y_min:y_max+1,x_min:x_max+1]
            map_upsamp = cv2.resize(patch,None,fx=upsamFactor,fy = upsamFactor,interpolation = cv2.INTER_CUBIC)
            #map_upsamp = gaussian_filter(map_upsamp,sigma=2)   skip this for now
            location_of_max = np.unravel_index(map_upsamp.argmax(),map_upsamp.shape)
            location_of_patch_center = compute_resized_coords(peak[::-1]-[y_min,x_min],upsamFactor)
            refined_center = (location_of_max - location_of_patch_center)
            peak_score = map_upsamp[location_of_max]
            peaks[i,:] = tuple([int(round(x)) for x in compute_resized_coords(peak_coords[i],upsamFactor)+refined_center[::-1]])+(peak_score,cnt_total_joints)
            cnt_total_joints +=1
        joint_list_per_joint_type.append(peaks)
    joint_list = np.array([tuple(peak) + (joint_type,) for joint_type, joint_peaks in enumerate(joint_list_per_joint_type) for peak in joint_peaks])
    print(joint_list.shape)
    #pof_upsamp = cv2.resize(pofs,(img.shape[1],img.shape[0]),interpolation = cv2.INTER_CUBIC)
    
    
    #here is get the z part
    result = np.zeros((19,3))
    result[17] = [joint_list[17][0],joint_list[17][1],0.1]
    for index,pairs in enumerate(linenum):
        start_point_idx = pairs[0]
        end_point_idx = pairs[1]
        vector_x = max(map(max,pof[3*index]))
        vector_y = max(map(max,pof[3*index+1]))
        vector_z = max(map(max,pof[3*index+2]))
        result[start_point_idx][0] = joint_list[start_point_idx][0]*1080/224
        result[start_point_idx][1] = joint_list[start_point_idx][1]*1920/224
        if result[start_point_idx][2] != 0:
            result[end_point_idx][2] = (result[start_point_idx][2] + vector_z *4)*1080/224#resize size ==> 4
        else:
            result[start_point_idx][2] = (joint_list[end_point_idx][2] - vector_z*4)*1080/224
        result[end_point_idx][0] = joint_list[end_point_idx][0]*1080/224
        result[end_point_idx][1] = joint_list[end_point_idx][1]*1920/224
        
    return result,img_input

def generate_joints_directly(img):
    heatmap,pof = restorenet(img)
    heatmap = heatmap.cpu().detach().numpy()
    pof = pof.cpu().detach().numpy()
    result = np.zeros((19,3))
    temp_result = np.zeros((19,3))
    for i in range(19):
        point_map = heatmap[i]
        points = np.where(point_map > (np.max(point_map))/2)
        for j in range(len(points[0])):
            temp_result[i][1] = points[1][j]/(len(points[0])) + temp_result[i][1]
            temp_result[i][0] = points[0][j]/(len(points[0])) + temp_result[i][0]
    result[17][2] =0.1
    for index,pairs in enumerate(linenum):
        start_point_idx = pairs[0]
        end_point_idx = pairs[1]
        vector_x = max(map(max,pof[3*index]))
        vector_y = max(map(max,pof[3*index+1]))
        vector_z = max(map(max,pof[3*index+2]))
        result[start_point_idx][0] = temp_result[start_point_idx][0]*1080/224*4
        result[start_point_idx][1] = temp_result[start_point_idx][1]*1920/224*4
        if result[start_point_idx][2] != 0:
            result[end_point_idx][2] = result[start_point_idx][2] + vector_z *4*1080/56#resize size ==> 4
        else:
            result[start_point_idx][2] = result[end_point_idx][2] - vector_z*4*1080/56
        result[end_point_idx][0] = temp_result[end_point_idx][0]*1080/224*4
        result[end_point_idx][1] = temp_result[end_point_idx][1]*1920/224*4
    return result



def generate_pof_joint(heatmap,pof):
    return 0

def draw_2d_img(img):
    result = generate_joints_directly(img)
    fig = plt.figure()
    ax = fig.add_subplot(111,projection = '3d')
    ax.view_init(elev = -90,azim = -90)
    ax.set_ylabel('Y label')
    ax.axis('equal')
    
    body_edges = np.array([[17,15],[15,1],[1,0],[0,2],[0,3],[3,4],[4,5],[2,6],[6,7],[7,8],[1,16],[16,18],[0,9],[9,10],[10,11],[2,12],[12,13],[13,14]])
    for edge in body_edges:
        ax.plot(result[edge,2],result[edge,0],result[edge,1],color="red")
    plt.show()
    return 0
def convert_joints(source):
    source_joint = ['neck','nose','bodycenter','lshoulder','lelbow','lwrist','lhip','lknee','lankle','rshoulder','relbow','rwrist','rhip','rknee','rankle','leye','reye','lear','rear']
    #19 joints
    target_joint = ['pelvis','lhip','rhip','spine1','lknee','rknee','spine2','lankle','rankle','spine3','lfoot','rfoot','neck','lcollar','rcollar','head','lshoulder','rshoulder','lelbow','relbow','lwrist','rwrist','lhand','rhand']
    #24 joints





def draw_smpl_model(img):
    result = generate_joints_directly(img)
    
    return 0


"""
this area contains several tools
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


#here is whaat would happen if you directly run, pretend the name of target image is "test1.jpg"

img = Image.open("test1.jpg")
draw_2d_img(img)
result = generate_joints_directly(img)
print(result)
json_filename = "test1.json"
camera_file_name = "camera.json"
jsonfile = pd.read_json(json_filename)
k = jsonfile.to_dict()
joint = k["bodies"][0]["joints19"]
with open(camera_file_name,'r') as jsonfile:
    camera = json.load(jsonfile)
joint_result = np.array(joint).reshape((-1,4)).transpose()
joint_result = cameramix(joint_result[0:3,:],camera)
"""
"""
fig = plt.figure()
ax = fig.add_subplot(111,projection = '3d')
ax.view_init(elev = -90,azim = -90)
ax.set_ylabel('Y label')
ax.axis('equal') 
body_edges = np.array([[17,15],[15,1],[1,0],[0,2],[0,3],[3,4],[4,5],[2,6],[6,7],[7,8],[1,16],[16,18],[0,9],[9,10],[10,11],[2,12],[12,13],[13,14]])
for edge in body_edges:
    ax.plot(joint_result[2,edge]-300,joint_result[1,edge],joint_result[0,edge],color="green")
plt.show()
"""
"""
print(joint_result)
