import numpy as np
import cv2
import h5py
import os

sn = "./Right/test300/"
images_folder ="images300-399"
path = os.path.join(sn, images_folder)
print(path)

if not os.path.isdir(path):
  os.mkdir(path)

def seg2Vast(seg):
    # convert to 24 bits
    return np.stack([seg//65536, seg//256, seg%256],axis=2).astype(np.uint8)
			
syn = h5py.File(sn +'label.h5', 'r')
print("heyyyy, label.h5 is read")
syn=np.array(syn['main'])

for z in range(0,100):
    cv2.imwrite(sn + images_folder+'/%04d.png'%z, seg2Vast(syn[z]))
    print(z)

