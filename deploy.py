import tensorflow as tf
import numpy as np
import skimage.io

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from ops import mkdir
from config import get_config
from deployer import Deployer

def main():
  config = get_config()
  name = config.name
  img_name = "deploy_data/" + name + ".tif"
  data = skimage.io.imread(img_name)

  print data.shape
 
  data = np.expand_dims(data, axis=3)
  data = (data / 255.).astype(np.float32)
  
  print data.shape

  with tf.Session() as sess:
    deployer = Deployer(sess, data, config)
    deployer.deploy()

if __name__=="__main__":
  main()
