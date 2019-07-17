from basicusage import generate_joints_directly
import pandas as pd
import cv2
from PIL import Image
import os.path
import glob
import numpy
import csv

source_video = "test1.mp4"


cap = cv2.VideoCapture(source_video)
c = 0
rval = cap.isOpened()



while rval:
    c = c+1
    rval,frame = cap.read()
    rval,frame = cap.read()
    rval,frame = cap.read()
    rval,frame = cap.read()
    rval,frame = cap.read()
    if rval:
        img = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        img = img.resize((224,224),Image.ANTIALIAS)
        result = generate_joints_directly(img)#here get the 3d joints(19*3)
        #here we need to re-order the joints information to fit the bvh situation
        right_order = [14,13,12,6,7,8,11,10,9,3,4,5,0,1,1,15,16,17,18,2]#this order means the index in result ordered by joint_names,there was an additional joint called head,I replaed it with nose.
        result_csv_0 = []
        for index in right_order:
            result_csv_0.append((result[index][1]-result[1][1])/100)
            result_csv_0.append((result[index][0]-result[1][0])/100)
            result_csv_0.append((result[index][2]-result[1][2])/100)
        result_csv = []
        result_csv.append(result_csv_0)
        #result_temp = result.to('cpu').detech().numpy()
        #result_csv = [result_temp[14][0:-1]]
        #result_csv = result.to('cpu').detach().numpy().reshape(1,57)
        joints_names = ['Ankle.R_x', 'Ankle.R_y', 'Ankle.R_z',
                   'Knee.R_x', 'Knee.R_y', 'Knee.R_z',
                   'Hip.R_x', 'Hip.R_y', 'Hip.R_z',
                   'Hip.L_x', 'Hip.L_y', 'Hip.L_z',
                   'Knee.L_x', 'Knee.L_y', 'Knee.L_z', 
                   'Ankle.L_x', 'Ankle.L_y', 'Ankle.L_z',
                   'Wrist.R_x', 'Wrist.R_y', 'Wrist.R_z', 
                   'Elbow.R_x', 'Elbow.R_y', 'Elbow.R_z', 
                   'Shoulder.R_x', 'Shoulder.R_y', 'Shoulder.R_z', 
                   'Shoulder.L_x', 'Shoulder.L_y', 'Shoulder.L_z',
                   'Elbow.L_x', 'Elbow.L_y', 'Elbow.L_z',
                   'Wrist.L_x', 'Wrist.L_y', 'Wrist.L_z', 
                   'Neck_x', 'Neck_y', 'Neck_z', 
                   'Head_x', 'Head_y', 'Head_z', 
                   'Nose_x', 'Nose_y', 'Nose_z', 
                   'Eye.L_x', 'Eye.L_y', 'Eye.L_z', 
                   'Eye.R_x', 'Eye.R_y', 'Eye.R_z', 
                   'Ear.L_x', 'Ear.L_y', 'Ear.L_z', 
                   'Ear.R_x', 'Ear.R_y', 'Ear.R_z',
                   'hip.Center_x','hip.Center_y','hip.Center_z']
        joints_export = pd.DataFrame(result_csv, columns=joints_names)
        joints_export.index.name = 'frame'
        joints_export.iloc[:, 1::3] = joints_export.iloc[:, 1::3]*-1
        joints_export.iloc[:, 2::3] = joints_export.iloc[:, 2::3]*-1
        print(str(c))
        joints_export.to_csv("output/test"+str(c)+".csv")
        #this part we have got the wanted result,now let combine the results into bvh
        """
        path = 'output/'                   
        all_files = glob.glob(os.path.join(path, "*.csv"))
        df_from_each_file = (pd.read_csv(f) for f in sorted(all_files))
        concatenated_df = pd.concat(df_from_each_file, ignore_index=True)
        concatenated_df['frame'] = concatenated_df.index+1
        concatenated_df.to_csv("result.csv", index=False)
        """
    else:
        break
path = 'output/'                   
all_files = glob.glob(os.path.join(path, "*.csv"))
df_from_each_file = (pd.read_csv(f) for f in sorted(all_files))
concatenated_df = pd.concat(df_from_each_file, ignore_index=True)
concatenated_df['frame'] = concatenated_df.index+1
concatenated_df.to_csv("result.csv", index=False)
