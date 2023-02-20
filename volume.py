import numpy as np
import os, sys
import cv2
import h5py
from matplotlib.pyplot import imread


def readVol(filename, z=None, kk=None):
    # read a folder of images
    if isinstance(filename, list) or isinstance(z, list):
        opt = 0
        if isinstance(filename, list):
            im0 = imread(filename[0])
            numZ = len(filename)
        elif isinstance(z, list):
            im0 = imread(filename%z[0])
            numZ = len(z)
            opt = 1            
        sz = list(im0.shape)
        out = np.zeros([numZ]+sz, im0.dtype)
        out[0] = im0
        for i in range(1,numZ):
            if opt ==0:
                fn = filename[i]
            elif opt ==1:
                fn = filename %z[i]
            out[i] = imread(fn)
    elif filename[-2:] == 'h5':
        tmp = h5py.File(filename,'r')
        if kk is None:
            kk = list(tmp)[0]
        if z is not None:
            out = np.array(tmp[kk][z])
        else:
            out = np.array(tmp[kk])
    elif filename[-3:] == 'zip':
        import zarr
        tmp = zarr.open_group(filename)
        if kk is None:
            kk = tmp.info_items()[-1][1]
            if ',' in kk:
                kk = kk[:kk.find(',')]
        out = np.array(tmp[kk][z])
    elif filename[-3:] in ['jpg','png']:
        import imageio
        out = imageio.imread(filename)
    elif filename[-3:] == 'txt':
        out = np.loadtxt(filename)
    elif filename[-3:] == 'npy':
        out = np.load(filename)
    else:
        raise "Can't read the file %s" % filename
    return out

def readImage(filename):
    import imageio
    image = imageio.imread(filename)
    return image

# h5 files
def readH5(filename, datasetname=None):
    import h5py
    fid = h5py.File(filename,'r')
    if datasetname is None:
        if sys.version[0]=='2': # py2
            datasetname = fid.keys()
        else: # py3
            datasetname = list(fid)
    if len(datasetname) == 1:
        datasetname = datasetname[0]
    if isinstance(datasetname, (list,)):
        out=[None]*len(datasetname)
        for di,d in enumerate(datasetname):
            out[di] = np.array(fid[d])
        return out
    else:
        return np.array(fid[datasetname])

def writeH5(filename, dtarray, datasetname='main'):
    import h5py
    fid=h5py.File(filename,'w')
    if isinstance(datasetname, (list,)):
        for i,dd in enumerate(datasetname):
            ds = fid.create_dataset(dd, dtarray[i].shape, compression="gzip", dtype=dtarray[i].dtype)
            ds[:] = dtarray[i]
    else:
        ds = fid.create_dataset(datasetname, dtarray.shape, compression="gzip", dtype=dtarray.dtype)
        ds[:] = dtarray
    fid.close()


sn = "../../datasets/nagP7/2000-2099/"

#filename_template
#1500-1599-p7_s
#1500-1599-p7_s1500.png
vol = readVol(sn + 'Alyssa_P7_fullvolume.vsvol_export_s%04d.png', [i for i in range(2000,2100)])
print("hey")
writeH5(sn+ '2000-2099.h5', vol)
print("its done!")

