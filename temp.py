import neuroglancer
import numpy as np
import imageio
import h5py
import matplotlib.pyplot as plt
from connectomics.utils.process import polarity2instance

ip = 'localhost' #or public IP of the machine for sharable display
port = 9999 #change to an unused port number
neuroglancer.set_server_bind_address(bind_address=ip,bind_port=port)
viewer=neuroglancer.Viewer()

#with viewer.txn() as s:
#    s.layers['image'] = neuroglancer.ImageLayer(source='precomputed://gs://neuroglancer-janelia-flyem-hemibrain/emdata/clahe_yz/jpeg/')
#    s.layers['segmentation'] = neuroglancer.SegmentationLayer(source='precomputed://gs://neuroglancer-janelia-flyem-hemibrain/v1.0/segmentation', selected_alpha=0.3)

# SNEMI (# 3d vol dim: z,y,x)
D0= './Right/test300/'
res = neuroglancer.CoordinateSpace(
        names=['z', 'y', 'x'],
        units=['nm', 'nm', 'nm'],
        scales=[30, 6, 6])
image_location = "../../datasets/nagP7/Right/300-399/"        
image_file = '0300-0399.h5'
segmentation_file = 'result.h5'

print( "image file: ",image_location+image_file, "and segmentation file: ", D0+segmentation_file)

print('load im and infernce segmentation')
with h5py.File(image_location+image_file, 'r') as f:
    #print(f.keys())
    im = np.array(f['main'])
#    im = im[0:50,:,:]
with h5py.File(D0+segmentation_file, 'r') as fl:
    gt  = np.array(fl['vol0']) #(3,100,700,700)    
    #gt  = gt[0]
    print("result.h5 shape is ", gt.shape)
    gt = gt[:,0:50,:,:]
    #gt = np.transpose(gt, (1,2,3,0)) 
    #print(gt.shape)
    #gt= gt[32,:,:,:]
    print("Hey")
    gt = polarity2instance(gt, exclusive=True) 
print(gt.shape)
#print(im.shape, gt.shape)
a= gt
hf = h5py.File(D0+'label1.h5', 'w')
hf.create_dataset('main', data=gt)
hf.close()
print("label1 file has been created and its shape is ", gt.shape)

#Second label
#with h5py.File(image_location+image_file, 'r') as f:
#    #print(f.keys())
#    im = np.array(f['main'])
#    im = im[50:100,:,:]
with h5py.File(D0+segmentation_file, 'r') as fl:
    gt  = np.array(fl['vol0']) #(3,100,700,700)    
    #gt  = gt[0]
    print("result.h5 shape is ", gt.shape)
    gt = gt[:,50:100,:,:]
    #gt = np.transpose(gt, (1,2,3,0)) 
    #print(gt.shape)
    #gt= gt[32,:,:,:]
    print("Hey")
    gt = polarity2instance(gt, exclusive=True) 
print(gt.shape)
#print(im.shape, gt.shape)
b=gt
hf = h5py.File(D0+'label2.h5', 'w')
hf.create_dataset('main', data=gt)
hf.close()
print("label2 file has been created and its shape is ", gt.shape)

#imgplot = plt.imshow(im[32,:,:])
#plt.show()
#segplt = plt.imshow (gt)
#plt.show()
gt= np.concatenate((a,b)) #concatenating both gt's
hf = h5py.File(D0+'label.h5', 'w')
hf.create_dataset('main',data=gt)
hf.close()
def ngLayer(data,res,oo=[0,0,0],tt='segmentation'):
    return neuroglancer.LocalVolume(data,dimensions=res,volume_type=tt,voxel_offset=oo)

with viewer.txn() as s:
    s.layers.append(name='im',layer=ngLayer(im,res,tt='image'))
    s.layers.append(name='gt',layer=ngLayer(gt,res, tt='segmentation'))

print(viewer)

