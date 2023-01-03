#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
from numpy import array, exp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def func1(x, a, b, c):
    return a*x**2+b*x+c
def func1(x, a, b,c,d):
    return a*x**3+b*x**2+c*x+d


abnormal_thr=1000

wedge_name=['Fe','Al','PMMA','Metal curve','Organic curve']
color_code=['bo','go','yo','co','ro']
color_code2=['b','g','y','c','r']
plt.figure(figsize=(20,20))
i=0
y_dense_all=[]
for wedge in wedge_name[:3]:
    img1=cv2.imread('D:/workspace/DataBoucher_Xray_Materials_Discrimination_Information/sample/example_2/'+ wedge +'/high.png',-1)/65535*255
    img2=cv2.imread('D:/workspace/DataBoucher_Xray_Materials_Discrimination_Information/sample/example_2/'+ wedge +'/low.png',-1)/65535*255
#     img1 = (img1/256).astype('uint8')
#     img2 = (img1/256).astype('uint8')
#     img_y=(cv2.absdiff(img1, img2))
#     img_x=((img1+img2)/2)



    crop_img1=np.zeros((50,img1.shape[1]-20),dtype='uint16')
    crop_img1[:,:]=img1[150:200,20:img1.shape[1]]

    crop_img2=np.zeros((50,img1.shape[1]-20),dtype='uint16')
    crop_img2[:,:]=img2[150:200,20:img1.shape[1]]

    # crop_img1=np.zeros((50,150-20),dtype='uint16')
    # crop_img1[:,:]=img1[150:200,20:150]

    # crop_img2=np.zeros((50,150-20),dtype='uint16')
    # crop_img2[:,:]=img2[150:200,20:150]

    img_y=(cv2.absdiff(crop_img1, crop_img2))
    img_x=((crop_img1+crop_img2)/2)
    x =np.mean(img_x,0)
    y = np.mean(img_y,0)

    d=x[int(len(y)/2)+int(len(y)/4):len(y)]
    d1=d-np.median(d)
    y_abnom=np.where(d1>abnormal_thr)
    d=x[int(len(x)/2)+int(len(x)/4):len(x)]
    d1=d-np.median(d)
    x_abnom=np.where(d1>abnormal_thr)
    xy_abnom=np.unique([y_abnom,x_abnom]).tolist()
    xy_abnom.sort(reverse=True)
    x=x.tolist()
    y=y.tolist()
    xy_abnom1 = list(map(lambda k: k +int(len(x)/2)+int(len(x)/4), xy_abnom))
    for ii in xy_abnom1:
        del x[ii]
        del y[ii]

    x=np.array(x)
    y=np.array(y)
   
    params, _ = curve_fit(func1, x, y)
    a, b, c, d = params[0], params[1], params[2], params[3]
    # yfit = a*x**2+b*x+c
    yfit =  a*x**3+b*x**2+c*x+d
    x_dense=np.linspace(0,255,256)
#     if wedge =='Fe':
#         y_dense=a*x_dense**3+b*0.98*x_dense**2+c*x_dense+d
#     if wedge =='PMMA':
#         y_dense=a*x_dense**3+b*0.95*x_dense**2+c*x_dense+d    
    y_dense=a*x_dense**3+b*x_dense**2+c*x_dense+d
    y_dense_all.append(y_dense)
    plt.plot(x, y, color_code[i], label=wedge)
    plt.plot(x, yfit,color_code2[i],  label= 'y={0:0.9f}*x^3 + {1:0.4f}*x^2 + {2:0.4f}*x + {3:0.4f}'.format(a, b,c,d))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='best', fancybox=True, shadow=True, fontsize=15)
    plt.grid(True)
    i=i+1

    
metal_curve=(y_dense_all[0]+y_dense_all[1])/2    
organic_curve=(y_dense_all[1]+y_dense_all[2])/2   
median_curve=[metal_curve,organic_curve]
metal_organic_params=[]
for ii in range(2):
    print(ii)
    x_dense=np.linspace(0,255,256)
    y_dense=median_curve[ii]
    params, _ = curve_fit(func1, x_dense, y_dense)
    a, b, c, d = params[0], params[1], params[2], params[3]
    if ii==0:
        a=a*1
        b=b*1
        c=c*1
        d=d+0
    else:
        a=a*1.3
        b=b*1.3
        c=c*1.3
        d=d+0
    metal_organic_params.append([a,b,c,d])
    # yfit = a*x**2+b*x+c
    yfit_dense =  a*x_dense**3+b*1*x_dense**2+c*x_dense+d
  
    plt.plot(x_dense, y_dense, color_code[ii+3], label=wedge_name[ii+3])
    plt.plot(x_dense, yfit_dense,color_code2[ii+3],  label= 'y={0:0.9f}*x^3 + {1:0.4f}*x^2 + {2:0.4f}*x + {3:0.4f}'.format(a, b,c,d))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='best', fancybox=True, shadow=True, fontsize=15)
    plt.grid(True)


# In[2]:


import os
import json
import numpy as np
import cv2
import matplotlib.cm as cm
from PIL import ImageFont, ImageDraw, Image
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt


from skimage import data, img_as_float
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.metrics import peak_signal_noise_ratio
from skimage.util import random_noise


################################################## RGB Graph ##################################################

# New Color code 적용
pg=[[[130,253],[130,130],[130,54]],[[130,20],[130,240],[130,60]],[[130,20],[130,100],[130,240]]] # parameter graph (순서: organic[R,G,B], inorganic[R,G,B], metal[R,G,B])
yy_mid_graph=np.linspace(0,255,256).tolist()
# 두점 좌표를 이용한 1차방정식 구하기
# y=a*x+b
x=np.linspace(0,255,256)
# x_1=np.linspace(0,150,150+1)
# x_2=np.linspace(151,255,255-151+1)
# y=(y2-y1)/(x2-x1)*(x-x1)-y1
materials_RGB=[]
for m in range(3):
    #################################  Red       #####################################################
    x1=0
    y1=0
    x_1=np.linspace(0,pg[m][0][0],pg[m][0][0]+1)
    x_2=np.linspace(pg[m][0][0]+1,255,255-pg[m][0][0])
    x2=pg[m][0][0]
    y2=pg[m][0][1]
    yy_1=[]
    for ii in x_1:
        x=ii
        y=(y2-y1)/(x2-x1)*(x-x1)+y1
        yy_1.append(y)

    x1=pg[m][0][0]
    y1=pg[m][0][1]
    x2=255
    y2=255
    yy_2=[]
    for ii in x_2:
        x=ii
        y=(y2-y1)/(x2-x1)*(x-x1)+y1
        yy_2.append(y)
    yy=[yy_1,yy_2]
    yy_11=sum(yy,[])


     #################################  Green         #####################################################
    x1=0
    y1=0
    x_1=np.linspace(0,pg[m][1][0],pg[m][1][0]+1)
    x_2=np.linspace(pg[m][1][0]+1,255,255-pg[m][1][0])
    x2=pg[m][1][0]
    y2=pg[m][1][1]
    yy_1=[]
    for ii in x_1:
        x=ii
        y=(y2-y1)/(x2-x1)*(x-x1)+y1
        yy_1.append(y)

    x1=pg[m][1][0]
    y1=pg[m][1][1]
    x2=255
    y2=255
    yy_2=[]
    for ii in x_2:
        x=ii
        y=(y2-y1)/(x2-x1)*(x-x1)+y1
        yy_2.append(y)
    # yR=(pg[0][0][1]/pg[0][0][0])*x1+x2
    yy=[yy_1,yy_2]
    yy_22=sum(yy,[])
    
    
     #################################   Blue       #####################################################
    x1=0
    y1=0
    x_1=np.linspace(0,pg[m][2][0],pg[m][2][0]+1)
    x_2=np.linspace(pg[m][2][0]+1,255,255-pg[m][2][0])
    x2=pg[m][2][0]
    y2=pg[m][2][1]
    yy_1=[]
    for ii in x_1:
        x=ii
        y=(y2-y1)/(x2-x1)*(x-x1)+y1
        yy_1.append(y)

    x1=pg[m][2][0]
    y1=pg[m][2][1]
    x2=255
    y2=255
    yy_2=[]
    for ii in x_2:
        x=ii
        y=(y2-y1)/(x2-x1)*(x-x1)+y1
        yy_2.append(y)
    # yR=(pg[0][0][1]/pg[0][0][0])*x1+x2
    yy=[yy_1,yy_2]
    yy_33=sum(yy,[])

    
   
    
   
    yy_m=[yy_11,yy_22,yy_33]
#     elif m==1: # inorganic
#         yy_m=[yy_22,yy_33,yy_11]
#     elif m==2: # metal
#         yy_m=[yy_33,yy_22,yy_11]
    #yy_Red=yy_11
    #yy_Blue=yy_22
    #yy_Green=yy_mid_graph
    materials_RGB.append(yy_m) # 순서: organic, inorganic, metal


# In[7]:


categories=[{'id': 1,
'name': 'artknife',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 2,
'name': 'fruitknife',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 3,
'name': 'chefknife',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 4,
'name': 'jackknife',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 5,
'name': 'officeutilityknife',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 6,
'name': 'scissors',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 7,
'name': 'steakknife',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 8,
'name': 'swissarmyknife',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 9,
'name': 'battery',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 10,
'name': 'laserpointer',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 11,
'name': 'gass',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 12,
'name': 'lighter',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 13,
'name': 'gun',
'supercategory': 'item',
'color': '040439',
'metadata': ''},

{'id': 14,
'name': 'container',
'supercategory': 'item',
'color': '040439',
'metadata': ''}]   
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import data
from skimage.morphology import disk
from skimage.filters.rank import mean_bilateral
import random
import os
def coordinate_segm_x(each_segm_x):
    return [each_segm_x[0][i] for i in range(0,len(each_segm_x[0]),2)]

def coordinate_segm_y(each_segm_y):
    return [each_segm_y[0][i+1] for i in range(0,len(each_segm_y[0]),2)]

def coordinate_segm_container(each_segm,random_rocation_x,random_rocation_y):
    return [((each_segm[0][i]+random_rocation_x,each_segm[0][i+1]+random_rocation_y)) for i in range(0,len(each_segm[0]),2)]
def coordinate_segm(each_segm):
    return [((each_segm[0][i],each_segm[0][i+1])) for i in range(0,len(each_segm[0]),2)]


mask_colors=[[1*255,1*255,0],[1*255,0,1*255],[0,0,1*255],[0,1*255,0],[1*255,0,0],[0,1*255,1*255],[0.5*255,0.5*255,0],[0.7*255,0.3*255,0.5*255],[1*255,0.5*255,0.1*255],[0.3*255,0.5*255,0.8*255]]
################################           Load item image ######################################################
imgimg=0

# dsfl=os.listdir('D:/KJE_Airiss/Police_data/xray/new_data_select/')
# for ds in dsfl:
ds='steakknife'
img_root='D:/KJE_Airiss/Police_data/xray/data_select_new_categ/'+ds+'/all_augment/crop/'
img_RGB_root='D:/KJE_Airiss/Police_data/xray/data_select_new_categ/'+ds+'/all_augment/crop_RGB/'

try:
    os.listdir(img_RGB_root) 
except:
    os.mkdir(img_RGB_root)

file_list=os.listdir(img_root)
img_num=0
# for ss in range(0,len(file_list),2):
for img_num in range(0,len(file_list),2):    
#     sl=file_list[ss]
#     bg_select=sl[-25:-4]
#     bg_select1=[]
#     for bgbg in file_list:
#         if bgbg.find(bg_select)>=0:
#             bg_select1.append(bgbg)
#     bg_name_high=bg_select1[np.where(np.array([bg_select1[0].find('_H'),bg_select1[1].find('_H')])>0)[0][0].tolist()]
#     bg_name_low=bg_select1[np.where(np.array([bg_select1[0].find('_L'),bg_select1[1].find('_L')])>0)[0][0].tolist()]
#     print(bg_name_high,bg_name_low)
#     high_energy_bg_img=bg_name_high
#     low_energy_bg_img= bg_name_low
#     img1=cv2.imread(img_root+high_energy_bg_img,-1)/65535*255  # High energy
#     img2=cv2.imread(img_root+low_energy_bg_img,-1)/65535*255 # Low energy
##################################################################
    random_item_high1 = img_num
    random_item_low1=img_num+1
    for fn in file_list:

        if int(fn[:fn.find('-')])==random_item_high1:
            random_item_high2=fn
        if int(fn[:fn.find('-')])==random_item_low1:
            random_item_low2=fn

    high_energy_img=(random_item_high2)
    low_energy_img=(random_item_low2)
###################################################################


    print(high_energy_img,low_energy_img)

    img1=cv2.imread(img_root+high_energy_img,-1)/65535*255  # High energy
    img2=cv2.imread(img_root+low_energy_img,-1)/65535*255  # Low energy

#     plt.figure(figsize=(20,20))
#     plt.subplot(121)
#     plt.imshow(img1_bg)
#     plt.subplot(122)
#     plt.imshow(img2_bg)
    indexX=img1.shape[1]
    indexY=img1.shape[0]
    arrayWidth=img1.shape[1]
    index_1d=0
    index_1d_2d=[]
    for column in range((indexY)):
        for row in range((indexX)):
            index_1d_2d.append([index_1d,row,column])
            index_1d=index_1d+1
    img_y=cv2.absdiff(img1, img2)
                # img_y=img1-img2
    img_x=((img1+img2)/2)
    def func1(x, a, b,c,d):
        return a*x**3+b*x**2+c*x+d

    imgx=img_x.flatten()
    imgy=img_y.flatten()
    color_img=img1*0+255
    color_img1=color_img.reshape(-1)
    color_img2=np.reshape(color_img1,img1.shape)
    color_img3=np.zeros((color_img2.shape[0],color_img2.shape[1],3),dtype='uint8')+255
    for ii in range(len(imgx)):
        ix=imgx[ii]
        iy=imgy[ii]
        #### check whether metal or mixture or organic ######## 
        # check metal #
        a,b,c,d=metal_organic_params[0]
        cm=func1(ix, a, b,c,d)
        # check organic #
        a,b,c,d=metal_organic_params[1]
        co=func1(ix, a, b,c,d)
        if  0<=iy < co:
            color_img3[index_1d_2d[ii][2],index_1d_2d[ii][1],0]=materials_RGB[0][0][int(imgx[ii])]
            color_img3[index_1d_2d[ii][2],index_1d_2d[ii][1],1]=materials_RGB[0][1][int(imgx[ii])]
            color_img3[index_1d_2d[ii][2],index_1d_2d[ii][1],2]=materials_RGB[0][2][int(imgx[ii])]
        elif co<=iy<cm:
            color_img3[index_1d_2d[ii][2],index_1d_2d[ii][1],0]=materials_RGB[1][0][int(imgx[ii])]
            color_img3[index_1d_2d[ii][2],index_1d_2d[ii][1],1]=materials_RGB[1][1][int(imgx[ii])]
            color_img3[index_1d_2d[ii][2],index_1d_2d[ii][1],2]=materials_RGB[1][2][int(imgx[ii])]


        elif iy>=cm:
            color_img3[index_1d_2d[ii][2],index_1d_2d[ii][1],0]=materials_RGB[2][0][int(imgx[ii])]
            color_img3[index_1d_2d[ii][2],index_1d_2d[ii][1],1]=materials_RGB[2][1][int(imgx[ii])]
            color_img3[index_1d_2d[ii][2],index_1d_2d[ii][1],2]=materials_RGB[2][2][int(imgx[ii])]

#         plt.figure(figsize=(20,20))
#         plt.imshow(color_img3)
    color_img3_RGB = cv2.cvtColor(color_img3, cv2.COLOR_BGR2RGB)


    cv2.imwrite(img_RGB_root+high_energy_img,color_img3_RGB )
    img_num=img_num+1
    imgimg=imgimg+1
    
    


# # Xray Gray img to RGB (make image size=700x700)

# In[227]:


categories=[{'id': 1,
'name': 'artknife',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 2,
'name': 'fruitknife',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 3,
'name': 'chefknife',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 4,
'name': 'jackknife',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 5,
'name': 'officeutilityknife',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 6,
'name': 'scissors',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 7,
'name': 'steakknife',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 8,
'name': 'swissarmyknife',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 9,
'name': 'battery',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 10,
'name': 'laserpointer',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 11,
'name': 'gass',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 12,
'name': 'lighter',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 13,
'name': 'gun',
'supercategory': 'item',
'color': '040439',
'metadata': ''},

{'id': 14,
'name': 'container',
'supercategory': 'item',
'color': '040439',
'metadata': ''}]   
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import data
from skimage.morphology import disk
from skimage.filters.rank import mean_bilateral
import random
import os
def coordinate_segm_x(each_segm_x):
    return [each_segm_x[0][i] for i in range(0,len(each_segm_x[0]),2)]

def coordinate_segm_y(each_segm_y):
    return [each_segm_y[0][i+1] for i in range(0,len(each_segm_y[0]),2)]

def coordinate_segm_container(each_segm,random_rocation_x,random_rocation_y):
    return [((each_segm[0][i]+random_rocation_x,each_segm[0][i+1]+random_rocation_y)) for i in range(0,len(each_segm[0]),2)]
def coordinate_segm(each_segm):
    return [((each_segm[0][i],each_segm[0][i+1])) for i in range(0,len(each_segm[0]),2)]


mask_colors=[[1*255,1*255,0],[1*255,0,1*255],[0,0,1*255],[0,1*255,0],[1*255,0,0],[0,1*255,1*255],[0.5*255,0.5*255,0],[0.7*255,0.3*255,0.5*255],[1*255,0.5*255,0.1*255],[0.3*255,0.5*255,0.8*255]]
################################           Load item image ######################################################
imgimg=0

img_root='D:/KJE_Airiss/Police_data/xray/20211124_verification_data/20211124_verification_data_reorder/image_reorder/'

img_RGB_root='D:/KJE_Airiss/Police_data/xray/20211124_verification_data/20211124_verification_data_reorder/image_reorder_RGB/'
img_gray_root_High='D:/KJE_Airiss/Police_data/xray/20211124_verification_data/20211124_verification_data_reorder/image_reorder_High/'
img_gray_root_Low='D:/KJE_Airiss/Police_data/xray/20211124_verification_data/20211124_verification_data_reorder/image_reorder_Low/'
img_gray_root_Avg='D:/KJE_Airiss/Police_data/xray/20211124_verification_data/20211124_verification_data_reorder/image_reorder_Avg/'
try:
    os.listdir(img_RGB_root)
    os.listdir(img_gray_root_High)
    os.listdir(img_gray_root_Low)
    os.listdir(img_gray_root_Avg)
except:
    os.mkdir(img_RGB_root)
    os.mkdir(img_gray_root_High)
    os.mkdir(img_gray_root_Low)
    os.mkdir(img_gray_root_Avg)


file_list=os.listdir(img_root)
for img_num in range(0,len(file_list),2):

###################################################################
#     random_item_high1 = img_num
#     random_item_low1=img_num+1
#     for fn in file_list:

#         if int(fn[:fn.find('-')])==random_item_high1:
#             random_item_high2=fn
#         if int(fn[:fn.find('-')])==random_item_low1:
#             random_item_low2=fn

#     high_energy_img=(random_item_high2)
#     low_energy_img=(random_item_low2)
####################################################################
    
    high_energy_img=str(img_num)+'.png'
    low_energy_img=str(img_num+1)+'.png'
    
    
    print(high_energy_img,low_energy_img)
    
    img1=cv2.imread(img_root+high_energy_img,-1)  # High energy
    img2=cv2.imread(img_root+low_energy_img,-1)  # Low energy
    
    
    img1=img1/65535*255
    img2 =img2/65535*255
#     plt.figure(figsize=(20,20))
#     plt.subplot(121)
#     plt.imshow(img1)
#     plt.subplot(122)
#     plt.imshow(img2)
    
    # img1=cv2.imread(img_root+high_energy_bg_img,-1)/65535*255  # High energy
    # img2=cv2.imread(img_root+low_energy_bg_img,-1)/65535*255 # Low energy
    #     plt.figure(figsize=(20,20))
    #     plt.subplot(121)
    #     plt.imshow(img1_bg)
    #     plt.subplot(122)
    #     plt.imshow(img2_bg)
    indexX=img1.shape[1]
    indexY=img1.shape[0]
    arrayWidth=img1.shape[1]
    index_1d=0
    index_1d_2d=[]
    for column in range((indexY)):
        for row in range((indexX)):
            index_1d_2d.append([index_1d,row,column])
            index_1d=index_1d+1
    img_y=cv2.absdiff(img1, img2)
                # img_y=img1-img2
    img_x=((img1+img2)/2)
    def func1(x, a, b,c,d):
        return a*x**3+b*x**2+c*x+d

    imgx=img_x.flatten()
    imgy=img_y.flatten()
    color_img=img1*0+255
    color_img1=color_img.reshape(-1)
    color_img2=np.reshape(color_img1,img1.shape)
    color_img3=np.zeros((color_img2.shape[0],color_img2.shape[1],3),dtype='uint8')+255
    for ii in range(len(imgx)):
        ix=imgx[ii]
        iy=imgy[ii]
        #### check whether metal or mixture or organic ######## 
        # check metal #
        a,b,c,d=metal_organic_params[0]
        cm=func1(ix, a, b,c,d)
        # check organic #
        a,b,c,d=metal_organic_params[1]
        co=func1(ix, a, b,c,d)
        if  0<=iy < co:
            color_img3[index_1d_2d[ii][2],index_1d_2d[ii][1],0]=materials_RGB[0][0][int(imgx[ii])]
            color_img3[index_1d_2d[ii][2],index_1d_2d[ii][1],1]=materials_RGB[0][1][int(imgx[ii])]
            color_img3[index_1d_2d[ii][2],index_1d_2d[ii][1],2]=materials_RGB[0][2][int(imgx[ii])]
        elif co<=iy<cm:
            color_img3[index_1d_2d[ii][2],index_1d_2d[ii][1],0]=materials_RGB[1][0][int(imgx[ii])]
            color_img3[index_1d_2d[ii][2],index_1d_2d[ii][1],1]=materials_RGB[1][1][int(imgx[ii])]
            color_img3[index_1d_2d[ii][2],index_1d_2d[ii][1],2]=materials_RGB[1][2][int(imgx[ii])]


        elif iy>=cm:
            color_img3[index_1d_2d[ii][2],index_1d_2d[ii][1],0]=materials_RGB[2][0][int(imgx[ii])]
            color_img3[index_1d_2d[ii][2],index_1d_2d[ii][1],1]=materials_RGB[2][1][int(imgx[ii])]
            color_img3[index_1d_2d[ii][2],index_1d_2d[ii][1],2]=materials_RGB[2][2][int(imgx[ii])]

    #         plt.figure(figsize=(20,20))
    #         plt.imshow(color_img3)
    color_img3_RGB = cv2.cvtColor(color_img3, cv2.COLOR_BGR2RGB)
    
    img_Avg=((img1+img2)/2)

##################### change size 700x700 #####################################    
    bg_w=img1.shape[1]
    bg_h=img1.shape[0]
    fix_size_bg_w=int((700-bg_w)/2)
    fix_size_bg_h=int((700-bg_h)/2)

    img11=np.zeros((700,700),dtype='uint8')+255
    img11[fix_size_bg_h:fix_size_bg_h+bg_h,fix_size_bg_w:fix_size_bg_w+bg_w]=img1
    img22=np.zeros((700,700),dtype='uint8')+255
    img22[fix_size_bg_h:fix_size_bg_h+bg_h,fix_size_bg_w:fix_size_bg_w+bg_w]=img2
    img_Avgg=np.zeros((700,700),dtype='uint8')+255
    img_Avgg[fix_size_bg_h:fix_size_bg_h+bg_h,fix_size_bg_w:fix_size_bg_w+bg_w]=img_Avg
    img_RGB=np.zeros((700,700,3),dtype='uint8')+255
    img_RGB[fix_size_bg_h:fix_size_bg_h+bg_h,fix_size_bg_w:fix_size_bg_w+bg_w,:]=color_img3_RGB 
    
    cv2.imwrite(img_RGB_root+str(imgimg)+'.png',img_RGB )
    cv2.imwrite(img_gray_root_High+str(imgimg)+'.png',img11)
    cv2.imwrite(img_gray_root_Low+str(imgimg)+'.png',img22)
    cv2.imwrite(img_gray_root_Avg+str(imgimg)+'.png',img_Avgg)

    imgimg=imgimg+1

    


# In[223]:


img1= (img1).astype('uint8')
img2 = (img2).astype('uint8')
img_Avg=((img1+img2)/2)


# In[224]:


img1


# In[226]:


(img1+img2)


# # X ray Gray image to RGB, make json file

# In[31]:


categories=[{'id': 1,
'name': 'artknife',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 2,
'name': 'fruitknife',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 3,
'name': 'chefknife',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 4,
'name': 'jackknife',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 5,
'name': 'officeutilityknife',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 6,
'name': 'scissors',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 7,
'name': 'steakknife',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 8,
'name': 'swissarmyknife',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 9,
'name': 'battery',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 10,
'name': 'laserpointer',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 11,
'name': 'gass',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 12,
'name': 'lighter',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 13,
'name': 'gun',
'supercategory': 'item',
'color': '040439',
'metadata': ''},

{'id': 14,
'name': 'container',
'supercategory': 'item',
'color': '040439',
'metadata': ''}]   
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import data
from skimage.morphology import disk
from skimage.filters.rank import mean_bilateral
import random
import os


def coordinate_segm_x(each_segm_x):
    return [each_segm_x[0][i] for i in range(0,len(each_segm_x[0]),2)]

def coordinate_segm_y(each_segm_y):
    return [each_segm_y[0][i+1] for i in range(0,len(each_segm_y[0]),2)]

def coordinate_segm_container(each_segm,random_rocation_x,random_rocation_y):
    return [((each_segm[0][i]+random_rocation_x,each_segm[0][i+1]+random_rocation_y)) for i in range(0,len(each_segm[0]),2)]
def coordinate_segm(each_segm):
    return [((each_segm[0][i],each_segm[0][i+1])) for i in range(0,len(each_segm[0]),2)]


mask_colors=[[1*255,1*255,0],[1*255,0,1*255],[0,0,1*255],[0,1*255,0],[1*255,0,0],[0,1*255,1*255],[0.5*255,0.5*255,0],[0.7*255,0.3*255,0.5*255],[1*255,0.5*255,0.1*255],[0.3*255,0.5*255,0.8*255]]
################################           Load item image ######################################################
imgimg=0

json_root='D:/KJE_Airiss/Police_data/xray/20211124_verification_data/json/data_reorder.json'
with open(json_root,"r") as pc_train_json:  # pc : poice
    augm_json_train = json.load(pc_train_json)
 


idx1=0
annot_image_id=[]
for ii in augm_json_train['annotations']:

    tmp=augm_json_train['annotations'][idx1]
    idx1=idx1+1
    annot_image_id.append(tmp['image_id'])
    
img_root='D:/KJE_Airiss/Police_data/xray/20211124_verification_data/image_reorder/'
img_RGB_root='D:/KJE_Airiss/Police_data/xray/20211124_verification_data/RGB_image/'
file_list=os.listdir(img_root)
idx_item=0
image_id=0
img_annot_id_new=0
augm_json_val_images={'images':[]}   
augm_json_val_annot={'annotations':[]}
for ss in range(0,len(file_list),2):
    high_energy_bg_img=str(ss)+'.png'
    low_energy_bg_img=str(ss+1)+'.png'
   
    print(high_energy_bg_img,low_energy_bg_img)
    
    img1=cv2.imread(img_root+high_energy_bg_img,-1)/65535*255  # High energy
    img2=cv2.imread(img_root+low_energy_bg_img,-1)/65535*255 # Low energy
#     plt.figure(figsize=(20,20))
#     plt.subplot(121)
#     plt.imshow(img1_bg)
#     plt.subplot(122)
#     plt.imshow(img2_bg)
    indexX=img1.shape[1]
    indexY=img1.shape[0]
    arrayWidth=img1.shape[1]
    index_1d=0
    index_1d_2d=[]
    for column in range((indexY)):
        for row in range((indexX)):
            index_1d_2d.append([index_1d,row,column])
            index_1d=index_1d+1
    img_y=cv2.absdiff(img1, img2)
                # img_y=img1-img2
    img_x=((img1+img2)/2)
    def func1(x, a, b,c,d):
        return a*x**3+b*x**2+c*x+d

    imgx=img_x.flatten()
    imgy=img_y.flatten()
    color_img=img1*0+255
    color_img1=color_img.reshape(-1)
    color_img2=np.reshape(color_img1,img1.shape)
    color_img3=np.zeros((color_img2.shape[0],color_img2.shape[1],3),dtype='uint8')+255
    for ii in range(len(imgx)):
        ix=imgx[ii]
        iy=imgy[ii]
        #### check whether metal or mixture or organic ######## 
        # check metal #
        a,b,c,d=metal_organic_params[0]
        cm=func1(ix, a, b,c,d)
        # check organic #
        a,b,c,d=metal_organic_params[1]
        co=func1(ix, a, b,c,d)
        if  0<=iy < co:
            color_img3[index_1d_2d[ii][2],index_1d_2d[ii][1],0]=materials_RGB[0][0][int(imgx[ii])]
            color_img3[index_1d_2d[ii][2],index_1d_2d[ii][1],1]=materials_RGB[0][1][int(imgx[ii])]
            color_img3[index_1d_2d[ii][2],index_1d_2d[ii][1],2]=materials_RGB[0][2][int(imgx[ii])]
        elif co<=iy<cm:
            color_img3[index_1d_2d[ii][2],index_1d_2d[ii][1],0]=materials_RGB[1][0][int(imgx[ii])]
            color_img3[index_1d_2d[ii][2],index_1d_2d[ii][1],1]=materials_RGB[1][1][int(imgx[ii])]
            color_img3[index_1d_2d[ii][2],index_1d_2d[ii][1],2]=materials_RGB[1][2][int(imgx[ii])]


        elif iy>=cm:
            color_img3[index_1d_2d[ii][2],index_1d_2d[ii][1],0]=materials_RGB[2][0][int(imgx[ii])]
            color_img3[index_1d_2d[ii][2],index_1d_2d[ii][1],1]=materials_RGB[2][1][int(imgx[ii])]
            color_img3[index_1d_2d[ii][2],index_1d_2d[ii][1],2]=materials_RGB[2][2][int(imgx[ii])]

#         plt.figure(figsize=(20,20))
#         plt.imshow(color_img3)
    color_img3_RGB = cv2.cvtColor(color_img3, cv2.COLOR_BGR2RGB)
   
  
    cv2.imwrite(img_RGB_root+str(imgimg)+'.png',color_img3_RGB )
   
    
    for kk in range(len(augm_json_train['images'])):


        if augm_json_train['images'][kk]['file_name']==high_energy_bg_img:
        #         print('idx cow image=',ii)
            img_id1=(augm_json_train['images'][kk]['id'])
            # print(img_id1)
            idx_segm=[ i for i,img_id in enumerate(annot_image_id) if img_id==img_id1]
            #                 print(len(idx_segm))

            segm=[]
            bbox=[]
            area=[]
            categ=[]
            for kkk in idx_segm:
                    json_df_annotation=[]

                    json_df_annotation=  {'id': img_annot_id_new,
                    'image_id': image_id,
                    'category_id': augm_json_train['annotations'][kkk]['category_id'],
                    'bbox': augm_json_train['annotations'][kkk]['bbox'],
                    'segmentation': augm_json_train['annotations'][kkk]['segmentation'],                    
                    'area':augm_json_train['annotations'][kkk]['area'],
                    'iscrowd': False,
                    'color': 'Unknown',
                    'unitID': 1,
                    'registNum': 1,
                    'number1': 4,
                    'number2': 4,
                    'weight': None}
                    augm_json_val_annot['annotations'].append(json_df_annotation)
                    img_annot_id_new=img_annot_id_new+1
            json_df_images= {'id': image_id,
                           'dataset_id': 1,
                           'path': str(idx_item)+'.png',
                           'file_name': str(idx_item)+'.png',
                           'width': augm_json_train['images'][kk]['width'],
                           'height': augm_json_train['images'][kk]['height']}  
            augm_json_val_images['images'].append(json_df_images)
    image_id=image_id+1 
    idx_item=idx_item+1
    imgimg=imgimg+1
    
    
    
    

val_json1=[]

val_json1={'images':[], 'annotations':[], 'categories':[]}

val_json1['images']=augm_json_val_images['images']
val_json1['annotations']=augm_json_val_annot['annotations']

val_json1['categories']=augm_json_train['categories']  
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
file_path='D:/KJE_Airiss/Police_data/xray/20211124_verification_data/json/data_reorder_RGB.json'
with open(file_path, 'w') as outfile:
    json.dump(val_json1, outfile, cls=NpEncoder)


# # image Resize(700x700)

# In[90]:


blue=np.uint8([[[255,0,0]]])
green=np.uint8([[[0,255,0]]])
red=np.uint8([[[0,0,255]]])
white=np.uint8([[[255,255,255]]])
hsv_blue=cv2.cvtColor(blue,cv2.COLOR_BGR2HSV)
hsv_green=cv2.cvtColor(green,cv2.COLOR_BGR2HSV)
hsv_red=cv2.cvtColor(red,cv2.COLOR_BGR2HSV)
hsv_white=cv2.cvtColor(white,cv2.COLOR_BGR2HSV)

print(hsv_blue)
print(hsv_green)
print(hsv_red)
print(hsv_white)


# In[118]:


img=cv2.imread(img_root_RGB+fn)


img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_Gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret,threshold_img=cv2.threshold(img_Gray,245,255,cv2.THRESH_BINARY)
plt.figure(figsize=(20,20))

result = cv2.bitwise_and(img_RGB, img_RGB, mask = threshold_img)

img[threshold_img==255]=0
img[threshold_img==255]=0

plt.imshow(img)


# In[124]:


import cv2
import numpy as np
img_root_RGB='D:/KJE_Airiss/Police_data/xray/20211124_verification_data/20211124_verification_data_reorder_RGB_High_Low_Separate/RGB_image_size(700x700)/'
 
fl=os.listdir(img_root_RGB)
for fn in fl:
#     if fn=='28.png':
        img=cv2.imread(img_root_RGB+fn)
        
        
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img_Gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         ret,threshold_img=cv2.threshold(img_Gray,245,255,cv2.THRESH_BINARY)
#         plt.figure(figsize=(20,20))
#         plt.subplot(141)
#         plt.imshow(img_Gray,cmap='gray')
#         plt.subplot(142)
#         plt.imshow(threshold_img,cmap='gray')
#         result = cv2.bitwise_and(img_RGB, img_RGB, mask = threshold_img)
#         plt.subplot(143)
#         plt.imshow(result)
#         img[threshold_img==255]=0
#         plt.subplot(144)
#         plt.imshow(img_RGB)
#         # It converts the BGR color space of image to HSV color space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Threshold of blue in HSV space
#         lower_white= np.array([0, 1, 230])
#         upper_white = np.array([0, 15, 255])
        # preparing the mask to overlay
#         mask = cv2.inRange(hsv, lower_white, upper_white)
#         mask_inv = cv2.bitwise_not(mask)

        lower_blue = np.array([20, 100, 10])
        upper_blue = np.array([130, 255, 255])
        
        # preparing the mask to overlay
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        mask_inv = cv2.bitwise_not(mask)
        # The black region in the mask has the value of 0,
        # so when multiplied with original image removes all non-blue regions
        result = cv2.bitwise_and(img_RGB, img_RGB, mask = mask)
        plt.figure(figsize=(20,20))
        plt.subplot(221)
        plt.imshow(img)
        plt.subplot(222)
        plt.imshow(img_RGB)
        plt.subplot(223)
        plt.imshow(mask,cmap='gray')
        plt.subplot(224)
#         plt.figure()    
        plt.imshow(result)
        plt.title(fn)
     


# In[36]:


img_root_high='D:/KJE_Airiss/Police_data/xray/20211124_verification_data/20211124_verification_data_reorder_RGB_High_Low_Separate/High_energy_image_size(700x700)_16bit/'
img_root_low='D:/KJE_Airiss/Police_data/xray/20211124_verification_data/20211124_verification_data_reorder_RGB_High_Low_Separate/Low_energy_image_size(700x700)_16bit/'
img_root_avg='D:/KJE_Airiss/Police_data/xray/20211124_verification_data/20211124_verification_data_reorder_RGB_High_Low_Separate/Avg_energy_image_size(700x700)_24bit/'
fl=os.listdir(img_root_high)

for ii in range(len(fl)):
    fn=fl[ii]
    print(fn)
    img_high=cv2.imread(img_root_high+fn)
    img_low=cv2.imread(img_root_low+fn)
    img_avg=(img_high/2+img_low/2)
#     img_avg=np.zeros((700,700),-1),dtype='uint8')+255
#     img_avg[:,:,0]=(img_high[:,:,0]+img_low[:,:,0])/2
#     img_avg[:,:,1]=(img_high[:,:,1]+img_low[:,:,1])/2
#     img_avg[:,:,2]=(img_high[:,:,2]+img_low[:,:,2])/2
#     plt.imshow(img_avg)
    cv2.imwrite(img_root_avg+fn,img_avg )


# In[ ]:


img_high


# In[35]:


img_avg


# In[122]:


# img2_bg=cv2.imread(img_root+low_energy_bg_img,-1) # Low energy
# aa=65535/(65535+random.randrange(int(-1*np.mean(img2_bg)*0.02),int(np.mean(img2_bg)*0.02)))
# img2_bg=img2_bg*aa
img_root='D:/KJE_Airiss/Police_data/xray/xray_real_data_20211126_RGB/'
img_RGB_resize_root='D:/KJE_Airiss/Police_data/xray/xray_real_data_20211126_RGB_size(700x700)/'
fl=os.listdir('D:/KJE_Airiss/Police_data/xray/xray_real_data_20211126_RGB/')

for ii in range(len(fl)):
    fn=fl[ii]
    print(fn)
    img2_bg_16bit=cv2.imread(img_root+fn,-1) # Low energy
#     img2_bg_gray = cv2.cvtColor(img2_bg_8bit, cv2.COLOR_BGR2GRAY)
    bg_w=img2_bg_16bit.shape[1]
    bg_h=img2_bg_16bit.shape[0]
    fix_size_bg_w=int((700-bg_w)/2)
    fix_size_bg_h=int((700-bg_h)/2)

    size_700_mask_RGB=np.zeros((700,700,3),dtype='uint8')+255
    size_700_mask_RGB[fix_size_bg_h:fix_size_bg_h+bg_h,fix_size_bg_w:fix_size_bg_w+bg_w,:]=img2_bg_16bit
    cv2.imwrite(img_RGB_resize_root+fn,size_700_mask_RGB )


# # Augment data(make wide image =1400x700)

# In[351]:


img_root_RGB='D:/KJE_Airiss/Police_data/xray/train/Random_jitter_augment/synthesis_RGB_10items5_RGB/'
img_root_High='D:/KJE_Airiss/Police_data/xray/train/Random_jitter_augment/synthesis_Gray_10items5_High/'
img_root_Low='D:/KJE_Airiss/Police_data/xray/train/Random_jitter_augment/synthesis_Gray_10items5_Low/'
img_root_Avg='D:/KJE_Airiss/Police_data/xray/train/Random_jitter_augment/synthesis_Gray_10items5_Avg/'

fl_RGB=os.listdir(img_root_RGB)
fl_High=os.listdir(img_root_High)
fl_Low=os.listdir(img_root_Low)
fl_Avg=os.listdir(img_root_Avg)


root_save='D:/KJE_Airiss/Police_data/xray/train/Random_jitter_augment/synthesis_RGB_10items5_RGB_High_low_Avg_wide_image/'


# In[352]:


from tifffile import imread, imwrite
from tifffile import imsave
for ii in range(len(fl_RGB)):
    
    fn=fl_RGB[ii]
    if int(fn[fn.find('_')+1:-4])>53214:
        print(fn)
        img_RGB=cv2.imread(img_root_RGB+fn) # Low energy
    #     img_RGB = cv2.cvtColor(img_RGB, cv2.COLOR_BGR2RGB)
    #     print(img_RGB.shape)
        fn=fl_High[ii]
        print(fn)
        img_High=cv2.imread(img_root_High+fn,-1) # Low energy
        fn=fl_Low[ii]
        print(fn)
        img_Low=cv2.imread(img_root_Low+fn,-1) # Low energy
        fn=fl_Avg[ii]
        print(fn)
        img_Avg=cv2.imread(img_root_Avg+fn,-1) # Low energy


        size_700_mask_RGB=np.zeros((700,700,3),dtype='uint8')+255
        size_700_mask_RGB[:,:,0]=img_High
        size_700_mask_RGB[:,:,1]=img_Low
        size_700_mask_RGB[:,:,2]=img_Avg
    #     img2_bg_gray = cv2.cvtColor(img2_bg_8bit, cv2.COLOR_BGR2GRAY)
    #     merged = np.concatenate((img_RGB[:,:,0],img_RGB[:,:,1],img_RGB[:,:,2]), axis=1) # creates a numpy array with 6 channels 
    #     merged1 = np.concatenate((img_High,img_Low,img_Avg), axis=1) # creates a numpy array with 6 channels 
        merged2 = np.concatenate((img_RGB,size_700_mask_RGB), axis=1) # creates a numpy array with 6 channels 
    #     cv2.imwrite(root_save+fn[:-4]+'.tiff', merged)

    #     size_700_mask_RGB=np.zeros((700,700,5),dtype='uint8')+255
    #     size_700_mask_RGB[:,:,0]=img_RGB[:,:,0]
    #     size_700_mask_RGB[:,:,1]=img_RGB[:,:,1]
    #     size_700_mask_RGB[:,:,2]=img_RGB[:,:,2]
    #     size_700_mask_RGB[:,:,3]=img_High
    #     size_700_mask_RGB[:,:,4]=img_Low
    #     size_700_mask_RGB=Image.fromarray(size_700_mask_RGB)
    #     size_700_mask_RGB.save(root_save+fn[:-4]+'.tif')
    #     imwrite(root_save+fn[:-4]+'.tif', size_700_mask_RGB, planarconfig='CONTIG')
        cv2.imwrite(root_save+fn,merged2)


# # Make Json for augment data(make wide image =1400x700)

# In[8]:


import copy
json_root='D:/KJE_Airiss/Police_data/xray/train/json/synthesis_police_10items_train_random_jitter_augment_86.json'
with open(json_root,"r") as pc_train_json:  # pc : poice
    augm_json_train = json.load(pc_train_json)
json_root='D:/KJE_Airiss/Police_data/xray/train/json/synthesis_police_10items_train_random_jitter_augment_5_5.json'
with open(json_root,"r") as pc_train_json:  # pc : poice
    augm_json_train1 = json.load(pc_train_json)    
# category_1=[]
# for ii in range(len(augm_json_train['annotations'])):   
#     if augm_json_train['annotations'][ii]['category_id']==1:
#         category_1.append(ii)


# In[355]:


import copy
# json_root='D:/KJE_Airiss/Police_data/xray/train/json/synthesis_police_10items_train_random_jitter_augment_final.json'
# with open(json_root,"r") as pc_train_json:  # pc : poice
#     augm_json_train = json.load(pc_train_json)
 


idx1=0
annot_image_id=[]
for ii in range(len(augm_json_train['annotations'])):

    tmp=augm_json_train['annotations'][idx1]
    idx1=idx1+1
    annot_image_id.append(tmp['image_id'])

idx1=0
annot_id=[]
for ii in range(len(augm_json_train['annotations'])):

    tmp=augm_json_train['annotations'][idx1]
    idx1=idx1+1
    annot_id.append(tmp['id'])
img_annot_id_new=max(annot_id)+1  
# imgimg=max(annot_image_id)+1
# img_id_new=max(annot_image_id)+1

    
augm_json_val_images={'images':[]}   
augm_json_val_annot={'annotations':[]}

for iii in range(len(augm_json_train['images'])):
        img_id1=(augm_json_train['images'][iii]['id'])

        
        idx_segm=[ i for i,img_id in enumerate(annot_image_id) if img_id==img_id1]
        for dd in idx_segm:
            segm=augm_json_train['annotations'][dd]['segmentation']
            segm1=copy.deepcopy(segm)
            area=augm_json_train['annotations'][dd]['area']
            bbox_item=augm_json_train['annotations'][dd]['bbox']
            bbox_item1=copy.deepcopy(bbox_item)
            bbox_item1[0]=bbox_item1[0]+700
            print(bbox_item1)
            category_item=augm_json_train['annotations'][dd]['category_id']
#             augm_json_train['annotations'][kk]['bbox'][1]=augm_json_train['annotations'][kk]['bbox'][1]+fix_size_bg_h
            for d in range(0,len(segm1[0]),2):
                segm1[0][d]=segm1[0][d]+700
            json_df_annotation=[]

            json_df_annotation=  {'id': img_annot_id_new,
            'image_id': img_id1,
            'category_id': category_item,
            'bbox': bbox_item1,
            'segmentation':segm1,                    
            'area':area,
            'iscrowd': False,
            'color': 'Unknown',
            'unitID': 1,
            'registNum': 1,
            'number1': 4,
            'number2': 4,
            'weight': None}
            augm_json_val_annot['annotations'].append(json_df_annotation)
            img_annot_id_new=img_annot_id_new+1



#             json_df_images= {'id': img_id_new,
#                        'dataset_id': 1,
#                        'path': 'synthesis_'+str(imgimg)+'.png',
#                        'file_name': 'synthesis_'+str(imgimg)+'.png',
#                        'width': bg_w,
#                        'height':bg_h}
#             augm_json_val_images['images'].append(json_df_images)
#             img_id_new=img_id_new+1
for kk in range(len(augm_json_train['images'])):            
    augm_json_train['images'][kk]['width']=1400
    augm_json_train['images'][kk]['height']=700
    

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
file_path='D:/KJE_Airiss/Police_data/xray/train/json/synthesis_police_10items_train_random_jitter_augment_5_5_wide.json'

val_json1=[]
# print(file_path)

val_json1={'images':[], 'annotations':[], 'categories':[]}

val_json1['images']=augm_json_train['images']
val_json1['annotations']=augm_json_train['annotations']+augm_json_val_annot['annotations']
val_json1['categories']=augm_json_train['categories']
with open(file_path, 'w') as outfile:
    json.dump(val_json1, outfile, cls=NpEncoder)


# In[10]:


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
file_path='D:/KJE_Airiss/Police_data/xray/train/json/synthesis_police_10items_train_random_jitter_augment_5_5_86.json'

val_json2=[]
# print(file_path)

val_json2={'images':[], 'annotations':[], 'categories':[]}

val_json2['images']=augm_json_train['images']+augm_json_train1['images']
val_json2['annotations']=augm_json_train['annotations']+augm_json_train1['annotations']
val_json2['categories']=augm_json_train1['categories']
with open(file_path, 'w') as outfile:
    json.dump(val_json2, outfile, cls=NpEncoder)


# In[319]:


categories=[
{'id': 2,
'name': 'fruitknife',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 3,
'name': 'chefknife',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 4,
'name': 'jackknife',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 5,
'name': 'officeutilityknife',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 6,
'name': 'scissors',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 7,
'name': 'steakknife',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 8,
'name': 'swissarmyknife',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 9,
'name': 'battery',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 10,
'name': 'laserpointer',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 11,
'name': 'gass',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 12,
'name': 'lighter',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 13,
'name': 'gun',
'supercategory': 'item',
'color': '040439',
'metadata': ''},

{'id': 14,
'name': 'container',
'supercategory': 'item',
'color': '040439',
'metadata': ''}]   


# In[320]:


# val_json1=[]
# # print(file_path)

# val_json1={'images':[], 'annotations':[], 'categories':[]}

# val_json1['images']=augm_json_train['images']
# val_json1['annotations']=augm_json_train['annotations']+augm_json_val_annot['annotations']
val_json1['categories']=categories
with open(file_path, 'w') as outfile:
    json.dump(val_json1, outfile, cls=NpEncoder)


# In[242]:


import copy
json_root='D:/KJE_Airiss/Police_data/xray/train/json/synthesis_police_10items_train_random_jitter_augment_7_3_wide.json'
with open(json_root,"r") as pc_train_json:  # pc : poice
    augm_json_train = json.load(pc_train_json)


# In[246]:


augm_json_train['images'][540]


# In[250]:


def get_mask(segm,bbox,categ):
        h=700
        w=1400
        c = 3

        mask =  np.zeros((h,w,c), np.uint8)
        all_mask_with_image=[]
        all_masks1=[]
        all_color_img_mask=[]

        for ii in range(len(segm)):
#         for ii in [1]:    
            each_segm=   segm[ii]  
#                 print(len(each_segm))    
            all_mask=[]
            out=[]

            all_masks=[]
            shapes = np.zeros((h,w,c), np.uint8)
            for jj in range(len(each_segm)):
#                 print(jj)

                coodinate1=coordinate_segm([each_segm[jj]])
                arr = np.array(coodinate1, np.int32)
                mask = cv2.fillPoly(mask, [arr], mask_colors[ii])
                shapes = cv2.fillPoly( shapes, [arr], mask_colors[ii])
#                 out = image.copy()
                alpha = 0.1
                mask1 = shapes.astype(bool)
                shapes1 = cv2.fillPoly( shapes, [arr], [255,255,255])
                all_masks.append(shapes1)

#                 out[mask1] = cv2.addWeighted(image, alpha, shapes, 1 - alpha, 0)[mask1]
#             all_mask_with_image.append(out) 
            all_masks1.append(all_masks)
            all_color_img_mask.append(shapes)  
#             all_mask_with_image1.append(all_mask_with_image)
        return mask, mask, all_masks1, bbox,all_color_img_mask, categ


# In[290]:




for ii in range(len(augm_json_train['annotations'])):
    if augm_json_train['annotations'][ii]['category_id']==14 and augm_json_train['annotations'][ii]['bbox'][0]<700:
        fn='synthesis_'+ str(augm_json_train['annotations'][ii]['image_id']) +'.png'
#         if fn=='synthesis_0.png':
        segm=augm_json_train['annotations'][ii]['segmentation']
        bbox=augm_json_train['annotations'][ii]['bbox']
        categ=augm_json_train['annotations'][ii]['category_id']
        all_mask=get_mask([segm], bbox,categ)

        for ii in range(len(all_mask[2])):
#                     if all_mask[5][ii]==2:

#                         plt.subplot(5,2,ii+1)
            shapes = all_mask[2][ii][0]
        #     shapes[:,:,0]=0
        #     shapes[:,:,2]=0
            shapes[[shapes>0][0][:,:,0],0]=  255
            shapes[[shapes>0][0][:,:,0],1]= 255
            shapes[[shapes>0][0][:,:,0],2]= 255




        mask1 = shapes.astype(bool)
#         plt.figure()
#         plt.imshow(shapes)
        img=cv2.imread('D:/KJE_Airiss/Police_data/xray/train/Random_jitter_augment/synthesis_RGB_10items3_RGB_High_low_Avg_wide_image/'+fn)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img2=img*mask1
        if np.mean(img2[:,:,0])>np.mean(img2[:,:,2]):
            augm_json_train['annotations'][ii]['category_id']=1
            augm_json_train['annotations'][ii+10]['category_id']=1
#             plt.figure()
#             plt.imshow(img2[:,:,0],cmap='gray')
#             plt.title(fn)


# In[286]:


categories=[{'id': 1,
'name': 'water',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 2,
'name': 'fruitknife',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 3,
'name': 'chefknife',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 4,
'name': 'jackknife',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 5,
'name': 'officeutilityknife',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 6,
'name': 'scissors',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 7,
'name': 'steakknife',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 8,
'name': 'swissarmyknife',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 9,
'name': 'battery',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 10,
'name': 'laserpointer',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 11,
'name': 'gass',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 12,
'name': 'lighter',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 13,
'name': 'gun',
'supercategory': 'item',
'color': '040439',
'metadata': ''},

{'id': 14,
'name': 'container',
'supercategory': 'item',
'color': '040439',
'metadata': ''}]   


# In[284]:


plt.figure(figsize=(20,20))
plt.subplot((131))
plt.imshow(img2[:,:,0],cmap='gray')
plt.title(fn)
plt.subplot((132))
plt.imshow(img2[:,:,1],cmap='gray')
plt.title(fn)
plt.subplot((133))
plt.imshow(img2[:,:,2],cmap='gray')
plt.title(fn)


# In[266]:


img2.shape


# In[257]:


mask1


# In[264]:


plt.figure()
plt.imshow(img2)


# In[254]:


len(all_mask[2])


# In[252]:


segm


# In[145]:


json_root='D:/KJE_Airiss/Police_data/xray/train/json/synthesis_police_10items_train_random_jitter_augment_6_3.json'
with open(json_root,"r") as pc_train_json:  # pc : poice
    augm_json_train = json.load(pc_train_json)


# In[152]:


augm_json_train['annotations'][25]


# In[142]:


json_df_annotation


# In[138]:


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
file_path='D:/KJE_Airiss/Police_data/xray/train/json/synthesis_police_10items_train_random_jitter_augment_6_3_wide.json'
with open(file_path, 'w') as outfile:
    json.dump(val_json1, outfile, cls=NpEncoder)


# In[134]:


len(val_json1['images'])


# In[136]:


len(augm_json_train['annotations'])


# In[135]:


len(val_json1['annotations'])


# In[25]:


for d in range(1,len(augm_json_train['annotations'][kk]['segmentation'][0]),2):
    print(d)


# In[19]:


augm_json_train['annotations'][kk]['bbox']


# In[16]:


plt.imshow(img2_bg_8bit)


# In[6]:


categories=[{'id': 1,
'name': 'water',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 2,
'name': 'fruitknife',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 3,
'name': 'chefknife',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 4,
'name': 'jackknife',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 5,
'name': 'officeutilityknife',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 6,
'name': 'scissors',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 7,
'name': 'steakknife',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 8,
'name': 'swissarmyknife',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 9,
'name': 'battery',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 10,
'name': 'laserpointer',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 11,
'name': 'gass',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 12,
'name': 'lighter',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 13,
'name': 'gun',
'supercategory': 'item',
'color': '040439',
'metadata': ''},

{'id': 14,
'name': 'container',
'supercategory': 'item',
'color': '040439',
'metadata': ''}]   
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import data
from skimage.morphology import disk
from skimage.filters.rank import mean_bilateral
import random
import os
def coordinate_segm_x(each_segm_x):
    return [each_segm_x[0][i] for i in range(0,len(each_segm_x[0]),2)]

def coordinate_segm_y(each_segm_y):
    return [each_segm_y[0][i+1] for i in range(0,len(each_segm_y[0]),2)]

def coordinate_segm_container(each_segm,random_rocation_x,random_rocation_y):
    return [((each_segm[0][i]+random_rocation_x,each_segm[0][i+1]+random_rocation_y)) for i in range(0,len(each_segm[0]),2)]
def coordinate_segm(each_segm):
    return [((each_segm[0][i],each_segm[0][i+1])) for i in range(0,len(each_segm[0]),2)]


mask_colors=[[1*255,1*255,0],[1*255,0,1*255],[0,0,1*255],[0,1*255,0],[1*255,0,0],[0,1*255,1*255],[0.5*255,0.5*255,0],[0.7*255,0.3*255,0.5*255],[1*255,0.5*255,0.1*255],[0.3*255,0.5*255,0.8*255]]
################################           Load item image ######################################################
imgimg=0


img_root='D:/KJE_Airiss/Police_data/xray/20211124_verification_data/image_reorder/'
img_RGB_root='D:/KJE_Airiss/Police_data/xray/20211124_verification_data/RGB_image/'
file_list=os.listdir(img_root)
# for ss in range(0,len(file_list),2):
#     sl=file_list[ss]
#     bg_select=sl[-25:-4]
#     bg_select1=[]
#     for bgbg in file_list:
#         if bgbg.find(bg_select)>=0:
#             bg_select1.append(bgbg)
#     bg_name_high=bg_select1[np.where(np.array([bg_select1[0].find('_H'),bg_select1[1].find('_H')])>0)[0][0].tolist()]
#     bg_name_low=bg_select1[np.where(np.array([bg_select1[0].find('_L'),bg_select1[1].find('_L')])>0)[0][0].tolist()]
#     print(bg_name_high,bg_name_low)
#     high_energy_bg_img=bg_name_high
#     low_energy_bg_img= bg_name_low
img1=cv2.imread(img_root+'1334.png',-1)/65535*255  # High energy
img2=cv2.imread(img_root+'1333.png',-1)/65535*255 # Low energy
#     plt.figure(figsize=(20,20))
#     plt.subplot(121)
#     plt.imshow(img1_bg)
#     plt.subplot(122)
#     plt.imshow(img2_bg)
indexX=img1.shape[1]
indexY=img1.shape[0]
arrayWidth=img1.shape[1]
index_1d=0
index_1d_2d=[]
for column in range((indexY)):
    for row in range((indexX)):
        index_1d_2d.append([index_1d,row,column])
        index_1d=index_1d+1
img_y=cv2.absdiff(img1, img2)
            # img_y=img1-img2
img_x=((img1+img2)/2)
def func1(x, a, b,c,d):
    return a*x**3+b*x**2+c*x+d

imgx=img_x.flatten()
imgy=img_y.flatten()
color_img=img1*0+255
color_img1=color_img.reshape(-1)
color_img2=np.reshape(color_img1,img1.shape)
color_img3=np.zeros((color_img2.shape[0],color_img2.shape[1],3),dtype='uint8')+255
for ii in range(len(imgx)):
    ix=imgx[ii]
    iy=imgy[ii]
    #### check whether metal or mixture or organic ######## 
    # check metal #
    a,b,c,d=metal_organic_params[0]
    cm=func1(ix, a, b,c,d)
    # check organic #
    a,b,c,d=metal_organic_params[1]
    co=func1(ix, a, b,c,d)
    if  0<=iy < co:
        color_img3[index_1d_2d[ii][2],index_1d_2d[ii][1],0]=materials_RGB[0][0][int(imgx[ii])]
        color_img3[index_1d_2d[ii][2],index_1d_2d[ii][1],1]=materials_RGB[0][1][int(imgx[ii])]
        color_img3[index_1d_2d[ii][2],index_1d_2d[ii][1],2]=materials_RGB[0][2][int(imgx[ii])]
    elif co<=iy<cm:
        color_img3[index_1d_2d[ii][2],index_1d_2d[ii][1],0]=materials_RGB[1][0][int(imgx[ii])]
        color_img3[index_1d_2d[ii][2],index_1d_2d[ii][1],1]=materials_RGB[1][1][int(imgx[ii])]
        color_img3[index_1d_2d[ii][2],index_1d_2d[ii][1],2]=materials_RGB[1][2][int(imgx[ii])]


    elif iy>=cm:
        color_img3[index_1d_2d[ii][2],index_1d_2d[ii][1],0]=materials_RGB[2][0][int(imgx[ii])]
        color_img3[index_1d_2d[ii][2],index_1d_2d[ii][1],1]=materials_RGB[2][1][int(imgx[ii])]
        color_img3[index_1d_2d[ii][2],index_1d_2d[ii][1],2]=materials_RGB[2][2][int(imgx[ii])]

#         plt.figure(figsize=(20,20))
#         plt.imshow(color_img3)
color_img3_RGB = cv2.cvtColor(color_img3, cv2.COLOR_BGR2RGB)


cv2.imwrite(img_RGB_root+'1333.png',color_img3_RGB )

imgimg=imgimg+1


# In[1]:


categories=[{'id': 1,
'name': 'artknife',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 2,
'name': 'fruitknife',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 3,
'name': 'chefknife',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 4,
'name': 'jackknife',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 5,
'name': 'officeutilityknife',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 6,
'name': 'scissors',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 7,
'name': 'steakknife',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 8,
'name': 'swissarmyknife',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 9,
'name': 'battery',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 10,
'name': 'laserpointer',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 11,
'name': 'gass',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 12,
'name': 'lighter',
'supercategory': 'item',
'color': '040439',
'metadata': ''},
{'id': 13,
'name': 'gun',
'supercategory': 'item',
'color': '040439',
'metadata': ''},

{'id': 14,
'name': 'container',
'supercategory': 'item',
'color': '040439',
'metadata': ''}]   
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import data
from skimage.morphology import disk
from skimage.filters.rank import mean_bilateral
import random
import os
def coordinate_segm_x(each_segm_x):
    return [each_segm_x[0][i] for i in range(0,len(each_segm_x[0]),2)]

def coordinate_segm_y(each_segm_y):
    return [each_segm_y[0][i+1] for i in range(0,len(each_segm_y[0]),2)]

def coordinate_segm_container(each_segm,random_rocation_x,random_rocation_y):
    return [((each_segm[0][i]+random_rocation_x,each_segm[0][i+1]+random_rocation_y)) for i in range(0,len(each_segm[0]),2)]
def coordinate_segm(each_segm):
    return [((each_segm[0][i],each_segm[0][i+1])) for i in range(0,len(each_segm[0]),2)]


mask_colors=[[1*255,1*255,0],[1*255,0,1*255],[0,0,1*255],[0,1*255,0],[1*255,0,0],[0,1*255,1*255],[0.5*255,0.5*255,0],[0.7*255,0.3*255,0.5*255],[1*255,0.5*255,0.1*255],[0.3*255,0.5*255,0.8*255]]
################################           Load item image ######################################################
imgimg=0


img_root='D:/KJE_Airiss/Police_data/xray/xray_real_data_20211126_High/'
img_RGB_root='D:/KJE_Airiss/Police_data/xray/xray_real_data_20211126_High_8bit/'
file_list=os.listdir(img_root)
for sl in file_list:
   
    img1=cv2.imread(img_root+sl,-1)/65535*255  # High energy
   

    cv2.imwrite(img_RGB_root+sl,img1 )
   


# In[41]:


imgimg


# In[20]:


img_root+high_energy_bg_img


# In[6]:


file_list

