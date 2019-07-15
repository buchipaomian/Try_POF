import numpy as np
import math
import matplotlib.pyplot as plt
import cv2

linenum = [[17,15],[15,1],[1,0],[0,2],[0,3],[3,4],[4,5],[2,6],[6,7],[7,8],[1,16],[16,18],[0,9],[9,10],[10,11],[2,12],[12,13],[13,14]]
"""
the input joint is 19*[x,y,z]
the heatmap should be size*size*19
range size set as 3(value/2/2/2)
the pof should be size*size*54
"""



def generate_heatmap(joint,size,orisize = 1080):
    """
    this will generate heatmap,which would paint through the joint point
    the range is 3,which would be 5*5 area(point in center)
    """
    heatmap = np.zeros((size,size,19))
    for i in range(19):
        x,y = joint[0][i],joint[1][i]
        x = int(x*size/1920)
        y = int(y*size/orisize)
        for j in range(-3,4):
            if x-j >=0 and x-j < size:
                for k in range(-3,4):
                    if y-k >=0 and y-k < size:
                        heatmap[x-j][y-k][i] =  x*y / max(2**abs(j),2**abs(k))
    return heatmap

def put_heatmap(joints,target_size,size = 28,orisize = 1080):
    """
    this is a new heatmap generater, based on https://blog.csdn.net/m0_37477175/article/details/81236115
    """
    heatmap = np.zeros((20,size,size),dtype = np.float32)
    for idx,point in enumerate(joints):
        if point[0] < 0 or point[1] <0:
            continue
        center_x,center_y,_ = point * size/orisize
        center_x = center_x*orisize/1920
        th = 4.6052
        delta = math.sqrt(th*2)
        sigma = 4
        
        x0 = int(max(0,center_x - delta * sigma))
        y0 = int(max(0,center_y - delta * sigma))
        
        x1 = int(min(size,center_x+delta*sigma))
        y1 = int(min(size,center_y+delta*sigma))
        
        for y in range(y0,y1):
            for x in range(x0,x1):
                d = (x-center_x) ** 2 + (y-center_y) **2
                exp = d/2.0/sigma/sigma
                if exp >th:
                    continue
                heatmap[idx][y][x] = max(heatmap[idx][y][x],math.exp(-exp))
                heatmap[idx][y][x] = min(heatmap[idx][y][x],1.0)
    heatmap = heatmap.transpose((1, 2, 0))
    heatmap[:, :, -1] = np.clip(np.amax(heatmap, axis=2), 0.0, 1.0)
    if target_size != size:
        heatmap = cv2.resize(heatmap,(target_size,target_size),interpolation= cv2.INTER_AREA)
    heatmap = heatmap.transpose((2, 0, 1))
    return heatmap.astype(np.float32)

def put_vectormap(size,vectormap,countmap,idx,center_from,center_to,threshold = 4):
    vec_x = center_to[0] - center_from[0]
    vec_y = center_to[1] - center_from[1]
    vec_z = center_to[2] - center_from[2]
    min_x = max(0, int(min(center_from[0], center_to[0]) - threshold))
    min_y = max(0, int(min(center_from[1], center_to[1]) - threshold))
    
    max_x = min(size, int(max(center_from[0], center_to[0]) + threshold))
    max_y = min(size, int(max(center_from[1], center_to[1]) + threshold))
    norm = math.sqrt(vec_x ** 2 + vec_y ** 2+ vec_z **2)
    if norm == 0:
        return
    vec_x /= norm
    vec_y /= norm
    vec_z /= norm
    
    for y in range(min_y,max_y):
        for x in range(min_x,max_x):
            bec_x = x-center_from[0]
            bec_y = y-center_from[1]
            dist = abs(bec_x * vec_y - bec_y * vec_x)
            if dist > threshold:
                continue
            
            countmap[idx][y][x]+=1
            vectormap[idx*3][y][x] = vec_x
            vectormap[idx*3+1][y][x] = vec_y
            vectormap[idx*3+2][y][x] = vec_z

def get_vectormap(joints,target_size,size = 28,orisize = 1080):
    vectormap = np.zeros((54,size,size),dtype = np.float32)
    countmap = np.zeros((18,size,size),dtype = np.int16)
    for idx,(j_idx1,j_idx2) in enumerate(linenum):
        center_from = joints[j_idx1]
        center_to = joints[j_idx2]
        center_from_change = [center_from[0]*size/1920,center_from[1]*size/orisize,center_from[2]*size/orisize]
        center_to_change = [center_to[0]*size/1920,center_to[1]*size/orisize,center_to[2]*size/orisize]
        put_vectormap(size,vectormap,countmap,idx,center_from_change,center_to_change)
    nonzeros = np.nonzero(countmap)
    for p,y,x in zip(nonzeros[0],nonzeros[1],nonzeros[2]):
        if countmap[p][y][x] <=0:
            continue
        vectormap[p*3][y][x] /= countmap[p][y][x]
        vectormap[p*3+1][y][x] /= countmap[p][y][x]
        vectormap[p*3+2][y][x] /= countmap[p][y][x]
    vectormap = vectormap.transpose((1,2,0))
    if target_size != size:
        vectormap = cv2.resize(vectormap,(target_size,target_size),interpolation= cv2.INTER_AREA)
    vectormap = vectormap.transpose((2,0,1))
    return vectormap.astype(np.float32)


def generate_pof(joint,size,orisize = 1080):
    pof = np.zeros((size,size,54))
    limb_width = 3
    stride = size/orisize
    for i in range(len(linenum)):
        jointk = np.array(joint)
        start_point,end_point = jointk[:,linenum[i][0]],joint[:,linenum[i][1]]
        #start_point is [x1,y1,z1],end_point is [x2,y2,z2]
        start_point[0] = start_point[0]*1080/1920
        end_point[0] = end_point[0]*1080/1920
        start_point = start_point/stride
        end_point = end_point/stride
        
        limb_vec = start_point-end_point
        norm = np.linalg.norm(limb_vec)
        if (norm == 0.0):
            continue
        limb_vec_unit = limb_vec/norm
        
        min_x = max(int(round(min(start_point[0],end_point[0])-limb_width)),0)
        max_x = min(int(round(max(start_point[0],end_point[0])+limb_width)),size)
        min_y = max(int(round(min(start_point[1],end_point[1])-limb_width)),0)
        max_y = min(int(round(max(start_point[1],end_point[1])+limb_width)),size)
        min_z = max(int(round(min(start_point[2],end_point[2])-limb_width)),0)
        max_z = min(int(round(max(start_point[2],end_point[2])+limb_width)),size)
        range_x = list(range(int(min_x),int(max_x),1))
        range_y = list(range(int(min_y),int(max_y),1))
        range_z = list(range(int(min_z),int(max_z),1))
        for j in  range_x:
            for k in range_y:
                pof[j][k][3*i] = start_point[0]-end_point[0]
                pof[j][k][3*i+1] = start_point[1]-end_point[1]
                pof[j][k][3*i+2] = start_point[2]-end_point[2]
    return pof

def imshow(inp,title = None):
    plt.imshow(inp)

def get_target(joint,size,orisize = 1080):
    heatmap = put_heatmap(joint,size,224)
    #imshow(heatmap[0])
    #imshow(heatmap[1])
    #imshow(heatmap[2])
    pof = get_vectormap(joint,size,224,orisize = 1080)
    #result = np.concatenate((heatmap,pof),axis = 0)
    #print(result.shape)
    return heatmap,pof
