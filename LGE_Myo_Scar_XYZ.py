#%% Import required libraries

import numpy
from PIL import Image
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import scipy
from skimage import morphology
from keras.models import Model, load_model
from keras.layers.core import Dropout
from keras.layers.merge import concatenate
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
import nibabel as nib
import glob
from matplotlib import pyplot as plt

path1 = r'Please provide the path where the 3D LGE CMRIs in the .nii format are located.'
LGEs = glob.glob(path1 + "/*")

path2 = r'Please provide the path where the myocardial masks created from our algorithm are located.'
MYOs = glob.glob(path2 + "/*")

path3 = r'Please provide the path where the ground truth of scar tissue in the .nii format are located.'
SCARs = glob.glob(path3 + "/*")

#%
smooth = 1.
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)
#%
scar_xy_model = load_model('segment_scar_xy.hdf5', custom_objects={'dice_coef': dice_coef,'dice_coef_loss': dice_coef_loss})
scar_xz_model = load_model('segment_scar_xz.hdf5', custom_objects={'dice_coef': dice_coef,'dice_coef_loss': dice_coef_loss}) 
scar_yz_model = load_model('segment_scar_yz.hdf5', custom_objects={'dice_coef': dice_coef,'dice_coef_loss': dice_coef_loss}) 
#%%


def myo_segment(img_test):
    myo_pred = scar_xy_model.predict(img_test, batch_size=1, verbose=1)
    myo_pred = myo_pred.reshape(x_unet, y_unet)
    myo_pred  = (myo_pred  > 0.5).astype(numpy.uint8)
    
    myo_clean = numpy.array(myo_pred, bool)
    myo_clean = morphology.remove_small_objects(myo_clean,100) 
    myo_clean = myo_clean*1 
    
    myo_clean = scipy.ndimage.morphology.binary_dilation(myo_clean,  iterations=3)
    myo_clean = scipy.ndimage.morphology.binary_erosion(myo_clean)
    return(myo_clean)
   

def model_xy_evaluate(data,mask):
    
    
    for k in range(len(data)):
        mask_sample = mask[k,:,:,:]
        mask_sample = mask_sample.reshape(x_unet, y_unet)

        img_test = data[k,:, :, :]
        img_test = img_test.reshape(1, x_unet, y_unet, 1)
        
        myo_clean = myo_segment(img_test)
        myo_clean = myo_clean[:x, :y]
        
        
        mask_clean = scipy.ndimage.morphology.binary_dilation(mask_sample,  iterations=2)
        mask_clean = scipy.ndimage.morphology.binary_erosion(mask_clean)
        mask_clean = mask_clean*1
        mask_clean = mask_clean[:x, :y]
    
        myo_xy[:,:,k] = myo_clean
        
        
        
    return()
    
     
    
def model_xz_evaluate(data,mask):
        
    for k in range(len(data)):
        mask_sample = mask[k,:,:,:]
        mask_sample = mask_sample.reshape(x_unet, y_unet)
        img_test = data[k,:, :, :]
        img_test = img_test.reshape(1, x_unet, y_unet, 1)
        img_pred = scar_xz_model.predict(img_test, batch_size=1, verbose=1)
        img_pred = img_pred.reshape(x_unet, y_unet)
        img_pred  = (img_pred  > 0.5).astype(numpy.uint8)
        
        seg_clean = numpy.array(img_pred, bool)
        seg_clean = morphology.remove_small_objects(seg_clean,100) 
        seg_clean = seg_clean*1 
        
        seg_clean = scipy.ndimage.morphology.binary_dilation(seg_clean, iterations=3)
        seg_clean = scipy.ndimage.morphology.binary_erosion(seg_clean)
        seg_clean = seg_clean*1    
        seg_clean = seg_clean[:x, :z]
               
        
        myo_xz[:,k,:] = seg_clean
        
        
    return()  
    

    
def model_yz_evaluate(data,mask):
    
    
    for k in range(len(data)):
        mask_sample = mask[k,:,:,:]
        mask_sample = mask_sample.reshape(x_unet, y_unet)
        img_test = data[k,:, :, :]
        img_test = img_test.reshape(1, x_unet, y_unet, 1)
        img_pred = scar_yz_model.predict(img_test, batch_size=1, verbose=1)
        img_pred = img_pred.reshape(x_unet, y_unet)
        img_pred  = (img_pred  > 0.5).astype(numpy.uint8)
        
        seg_clean = numpy.array(img_pred, bool)
        seg_clean = morphology.remove_small_objects(seg_clean,100) 
        seg_clean = seg_clean*1 
        
        seg_clean = scipy.ndimage.morphology.binary_dilation(seg_clean, iterations=3)
        seg_clean = scipy.ndimage.morphology.binary_erosion(seg_clean)
        seg_clean = seg_clean*1    
        seg_clean = seg_clean[:y, :z]
        

        myo_yz[k,:,:] = seg_clean
        
        
    return()    
    


    
def Create_XY_data(lge,sacr):
    
    
    lge_norm = numpy.zeros((x,y,z))
    for slice_no in range (z):
        lge_slice = lge[:, :, slice_no]
        for a in range (x):
            for b in range (y):
                if lge_slice[a,b] > 1000:
                    lge_slice[a,b] = numpy.median(lge_slice)
        if (numpy.max(lge_slice != 0)):            
            lge_slice = (lge_slice-lge_slice.min())/(lge_slice.max()-lge_slice.min())
            lge_norm[:, :, slice_no] = lge_slice


    data = numpy.zeros((1,x_unet*y_unet))
    mask_myo = numpy.zeros((1,x_unet*y_unet))
  
    
    x_pad = int(x_unet - x)
    y_pad = int(y_unet - y)
    
    for page in range(0,z):    
        lge_slice = lge_norm[:,:,page]
        myo_slice = scar[:,:,page]
        
        lge_slice = numpy.pad(lge_slice, ((0, x_pad),(0, y_pad)), 'wrap')
        myo_slice = numpy.pad(myo_slice, ((0, x_pad),(0, y_pad)), 'wrap')
        
        lge_slice = lge_slice.reshape(1,(x_unet*y_unet))
        myo_slice = myo_slice.reshape(1, (x_unet*y_unet))
        
        data = numpy.vstack((data,lge_slice ))
        mask_myo = numpy.vstack((mask_myo,myo_slice))

    data = numpy.delete(data, (0), axis=0)     
    mask_myo = numpy.delete(mask_myo, (0), axis=0) 
        
    data = data.reshape(data.shape[0], x_unet, y_unet, 1)
    mask_myo = mask_myo.reshape(mask_myo.shape[0], x_unet, y_unet, 1)
    
    model_xy_evaluate(data,mask_myo)
    
      
    return()
    
   
    
def Create_XZ_data(lge,scar):
    
    
    lge_norm = numpy.zeros((x,y,z))
    for slice_no in range (y):
        lge_slice = lge[:,slice_no,:]
        for a in range (x):
            for b in range (z):
                if lge_slice[a,b] > 1000:
                    lge_slice[a,b] = numpy.median(lge_slice)
        if (numpy.max(lge_slice != 0)):            
            lge_slice = (lge_slice-lge_slice.min())/(lge_slice.max()-lge_slice.min())
            lge_norm[:,slice_no,:] = lge_slice
      
    
    data = numpy.zeros((1,x_unet*y_unet))
    mask_myo = numpy.zeros((1,x_unet*y_unet))
    
    x_pad = int(x_unet - x)
    y_pad = int(y_unet - z)
    
    for page in range(0,y):    
        lge_slice = lge_norm[:,page,:]
        myo_slice = scar[:,page,:]
        
        lge_slice = numpy.pad(lge_slice, ((0, x_pad),(0, y_pad)), 'wrap')
        myo_slice = numpy.pad(myo_slice, ((0, x_pad),(0, y_pad)), 'wrap')
        
        lge_slice = lge_slice.reshape(1,(x_unet*y_unet))
        myo_slice = myo_slice.reshape(1, (x_unet*y_unet))
        
        data = numpy.vstack((data,lge_slice ))
        mask_myo = numpy.vstack((mask_myo,myo_slice))

    data = numpy.delete(data, (0), axis=0)     
    mask_myo = numpy.delete(mask_myo, (0), axis=0) 
        
    data = data.reshape(data.shape[0], x_unet, y_unet, 1)
    mask_myo = mask_myo.reshape(mask_myo.shape[0], x_unet, y_unet, 1)
    model_xz_evaluate(data,mask_myo)
    
    
    return()
    
     
    
def Create_YZ_data(lge,scar):
    
    lge_norm = numpy.zeros((x,y,z))
    for slice_no in range (x):
        lge_slice = lge[slice_no,:,:]
        for a in range (y):
            for b in range (z):
                if lge_slice[a,b] > 1000:
                    lge_slice[a,b] = numpy.median(lge_slice)
        if (numpy.max(lge_slice != 0)):            
            lge_slice = (lge_slice-lge_slice.min())/(lge_slice.max()-lge_slice.min())
            lge_norm[slice_no,:,:] = lge_slice
      
        
    data = numpy.zeros((1,x_unet*y_unet))
    mask = numpy.zeros((1,x_unet*y_unet))
    
    x_pad = int(x_unet - y)
    y_pad = int(y_unet - z)
    
    for page in range(0,x):    
        lge_slice = lge_norm[page,:,:]
        myo_slice = scar[page,:,:]
        
        lge_slice = numpy.pad(lge_slice, ((0, x_pad),(0, y_pad)), 'wrap')
        myo_slice = numpy.pad(myo_slice, ((0, x_pad),(0, y_pad)), 'wrap')
        
        lge_slice = lge_slice.reshape(1,(x_unet*y_unet))
        myo_slice = myo_slice.reshape(1, (x_unet*y_unet)) 
        
        data = numpy.vstack((data,lge_slice ))
        mask = numpy.vstack((mask,myo_slice))

    data = numpy.delete(data, (0), axis=0)     
    
    mask = numpy.delete(mask, (0), axis=0)     
    
    data = data.reshape(data.shape[0], x_unet, y_unet, 1)
    mask = mask.reshape(mask.shape[0], x_unet, y_unet, 1)
    model_yz_evaluate(data,mask)
       
    
    return()
    
    
#%% Create test dataset and test unseen images.
x_unet = 256
y_unet = 256
dsc_total =[]
acc_total = []
prec_total = []
rec_total = []
vol_manual = []
vol_seg = []
sec = []

for n in range(18,34): 
    print(n)
    start_time = time.time()
    data_lge = nib.load(LGEs[n]);
    lge = data_lge.get_data()
    x,y,z = lge.shape
    
    data_myo = nib.load(MYOs[n-18]);
    myo = data_myo.get_data() 
    
   
    data_scar = nib.load(SCARs[n]);
    scar = data_scar.get_data() 
    
    for i in range(x):
        for j in range(y):
            for k in range(z):
                if myo[i,j,k] == 0:
                    lge[i,j,k] = 0
    
    img_orig = numpy.zeros((x,y,z))
    myo_xy = numpy.zeros((x,y,z))
    myo_xz = numpy.zeros((x,y,z))
    myo_yz = numpy.zeros((x,y,z))  
    
    
    Create_XY_data(lge,scar)
    Create_XZ_data(lge,scar) 
    Create_YZ_data(lge,scar)
    
    
    myo_final = myo_xy + myo_xz + myo_yz
    myo_vote = numpy.zeros((myo_final.shape))
    for i in range(myo_final.shape[0]):
        for j in range(myo_final.shape[1]):
            for k in range(myo_final.shape[2]):
                if myo_final[i,j,k] >= 2:
                    myo_vote[i,j,k] = 1
    
    
    dsc = []
    acc = []
    prec = []
    rec = []
    
    
    myo_clean =  numpy.zeros((myo_vote.shape))
    gt_clean =  numpy.zeros((scar.shape))
    for page in range(myo_vote.shape[2]):
        myo_vote_slc = myo_vote[:,:,page]
        myo_slc = scar[:,:,page]
        seg_clean = numpy.array(myo_vote_slc, bool)
        seg_clean = morphology.remove_small_objects(seg_clean,100) 
        seg_clean = seg_clean*1 
    
        seg_clean = scipy.ndimage.morphology.binary_dilation(seg_clean, iterations=1)   
        seg_clean = seg_clean*1    
        myo_clean[:,:,page] = seg_clean
        
        myo_slc = scipy.ndimage.morphology.binary_dilation(myo_slc, iterations=3)   
        myo_slc = myo_slc*1    
        gt_clean[:,:,page] = myo_slc
        
        y_true = numpy.reshape(myo_slc, (x*y,1))
        y_pred = numpy.reshape(seg_clean, (x*y,1)) 
        dsc = numpy.append(dsc,f1_score(y_true, y_pred, average='macro')*100 )
        acc = numpy.append(acc,accuracy_score(y_true, y_pred)*100)
        prec = numpy.append(prec, precision_score(y_true, y_pred, average='macro')*100)
        rec = numpy.append(rec, recall_score(y_true, y_pred, average='macro')*100)
    
       
    
    dsc_total = numpy.append(dsc_total,numpy.mean(dsc))
    acc_total = numpy.append(acc_total,numpy.mean(acc))
    prec_total = numpy.append(prec_total,numpy.mean(prec))
    rec_total = numpy.append(rec_total,numpy.mean(rec))        
    vol_manual = numpy.append(vol_manual,numpy.sum(gt_clean)*1.3*0.625*0.625/1000)
    vol_seg = numpy.append(vol_seg,numpy.sum(myo_clean)*1.3*0.625*0.625/1000)     
    sec =  numpy.append(sec,(time.time() - start_time)) 
    slice_no = slice_no + myo_clean.shape[0] + myo_clean.shape[1] + myo_clean.shape[2]

   
 
#%   
print('Mean Values:')    
print('DI is :', round(numpy.mean(dsc_total),2) , '+', round(numpy.std(dsc_total),2))
print('Acc. is :', round(numpy.mean(acc_total),2), '+', round(numpy.std(acc_total),2))
print('Precision is :', round(numpy.mean(prec_total),2), '+', round(numpy.std(prec_total),2))
print('Recall is :', round(numpy.mean(rec_total),2), '+', round(numpy.std(rec_total),2))

print('Median Values:') 
print('DI is :', round(numpy.median(dsc_total),2) , '+', round(numpy.std(dsc_total),2))
print('Acc. is :', round(numpy.median(acc_total),2), '+', round(numpy.std(acc_total),2))
print('Precision is :', round(numpy.median(prec_total),2), '+', round(numpy.std(prec_total),2))
print('Recall is :', round(numpy.median(rec_total),2), '+', round(numpy.std(rec_total),2))
  
