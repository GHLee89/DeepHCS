import numpy as np
import skimage.io
import cv2
from scipy import ndimage

class ISBIHandler(object):
  def __init__(self, A_path, B_path, input_size):
    self.A = skimage.io.imread(A_path) # training data : x 
    self.B = skimage.io.imread(B_path) # training data : y
    self.input_size = input_size

    print 'X_shape : ', self.A.shape
    print 'y_shape : ', self.B.shape

    self.A = np.expand_dims(self.A, axis=3)
    self.B = np.expand_dims(self.B, axis=3)

    print 'X_shape : ', self.A.shape
    print 'y_shape : ', self.B.shape

    self.A = self.A.astype(np.float32)
    self.B = self.B.astype(np.float32)

    self.A = (self.A / 255.).astype(np.float32)
    self.B = (self.B / 255.).astype(np.float32)

  def sample_pair(self, batch_size, is_for_refiner=False):
    shape = self.A.shape # self.B.shape should be same.
    sample_A = np.empty([batch_size, self.input_size, self.input_size, 1])
    sample_B = np.empty([batch_size, self.input_size, self.input_size, 1])
    
    for i in range(batch_size):
      z = np.random.choice(self.A.shape[0])
      sample_A[i, :, :, :] = self.A[z, :, :, :]
      sample_B[i, :, :, 0] = self.B[z, :, :, 0]

    ############## I should add the data augmentation for input data ################
    sample_A, sample_B = self.augmentation(sample_A, sample_B, batch_size)

    return sample_A, sample_B

  def augmentation(self, sample_A, sample_B, batch_size):
    # flip, rotation ...
    for i in range(batch_size):
      # single slice
      slice_A, slice_B = sample_A[i,:,:,:], sample_B[i,:,:,0]

      # flip
      flip_flag = np.random.randint(2, size=2)
      if flip_flag[0]:
        slice_A = slice_A[::-1,:,:]
        slice_B = slice_B[::-1,:]
      if flip_flag[1]:
        slice_A = slice_A[:,::-1,:]
        slice_B = slice_B[:,::-1]

      # rotation
      rot_num = np.random.randint(4)
      slice_A = np.rot90(slice_A, rot_num)
      slice_B = np.rot90(slice_B, rot_num)

      sample_A[i,:,:,:] = slice_A
      sample_B[i,:,:,0] = slice_B

    return sample_A, sample_B
