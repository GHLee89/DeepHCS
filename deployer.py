import tensorflow as tf

from models import *
from ops import save_image_with_scale_1, save_image_with_scale_2

class Deployer(object):
  def __init__(self, sess, data, config):
    self.sess = sess

    self.data = data
    height = self.data.shape[1]
    width  = self.data.shape[2]
    self.result_dir = config.result_dir
    self.name       = config.name
    self.ckpt_dir   = config.ckpt_dir
    self.result_dirs = self.result_dir + self.name

    # input data for deployment
    self.x = tf.placeholder(tf.float32, [1, height, width, 1])

    self.T   = Transform_prep("T", 1, is_train=False)
    self.R   = RefineNet("R", 1, is_train=False)

    self.y_ = self.T(self.x)

    R_input  = tf.concat([self.y_, self.x], 3)
    self.y__ = self.R(R_input)

    self.sess.run(tf.global_variables_initializer())
    self.restore()

  def restore(self):
    self.T.restore(self.sess, self.ckpt_dir+"/ckpt_T1/model.ckpt")
    self.R.restore(self.sess, self.ckpt_dir+"/ckpt_T2/model.ckpt")

  def deploy(self):
    for i in range(self.data.shape[0]):
      x = self.data[i:i+1]
      y_, y__ = self.sess.run([self.y_, self.y__], {self.x:x})

      save_image_with_scale_1(
        self.result_dirs+"/{:02d}_y_1.tif".format(i), y_[0,:,:,:])
      save_image_with_scale_1(
        self.result_dirs+"/{:02d}_y_2.tif".format(i), y__[0,:,:,:])


