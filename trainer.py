import tensorflow as tf

from models import *
from ops import mse, mse_tv, save_image_with_scale_1, save_image_with_scale_2

class Trainer(object):
  def __init__(self, sess, dh, config):
    self.sess = sess
    self.dh = dh
    self.config = config

    self.batch_size = self.config.batch_size
    self.input_size = self.config.input_size
    self.ckpt_dir = self.config.ckpt_dir
    self.tmp_dir = self.config.tmp_dir

    # raw input
    self.x = tf.placeholder(tf.float32, 
      [self.batch_size, self.input_size, self.input_size, 1])

    # label input
    self.y = tf.placeholder(tf.float32, 
      [self.batch_size, self.input_size, self.input_size, 1])
    
    self.y_f = tf.cast(self.y, tf.float32)

    # Transform Network and Refinement Network
    self.T   = Transform_prep("T", 1, is_train=True)
    self.R   = RefineNet("R", 1, is_train=True)

    self.y_  = self.T(self.x)

    R_input  = tf.concat([self.y_, self.x], 3)
    self.y__ = self.R(R_input)

    alpha = 0.8
    self.l_const_1  = mse(self.y_, self.y_f)
    self.l_const_2  = (1-alpha) * mae(self.y__, self.y_f) + alpha * tf_ms_ssim(self.y__, self.y_f)

    L_T1 = self.l_const_1   # Mean Square Error
    L_T2 = self.l_const_2  # Mean Absolute Error + SSIM

    self.optim = tf.train.AdamOptimizer(config.learning_rate)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
    	self.t1_train = self.optim.minimize(L_T1, var_list=self.T.var_list)
    	self.t2_train = self.optim.minimize(L_T2, var_list=self.R.var_list)

    self.sess.run(tf.global_variables_initializer())
      
  def save(self):
    self.T.save(self.sess, self.ckpt_dir+"/ckpt_T1/model.ckpt")
    self.R.save(self.sess, self.ckpt_dir+"/ckpt_T2/model.ckpt")

  def restore(self):
    self.T.restore(self.sess, self.ckpt_dir+"/ckpt_T1/model.ckpt")
    self.R.restore(self.sess, self.ckpt_dir+"/ckpt_T2/model.ckpt")

  def deploy(self):
    x, y = self.dh.sample_pair(self.batch_size)
    print 'X_shape : ', x.shape
    print 'y_shape : ', y.shape
    y_, y__ = self.sess.run([self.y_, self.y__], {self.x:x})
    for i in range(self.batch_size):
      save_image_with_scale_1(
        self.tmp_dir+"/{:02d}_x.tif".format(i), x[i,:,:,:])
      save_image_with_scale_1(
        self.tmp_dir+"/{:02d}_y.tif".format(i), y[i,:,:,:])
      save_image_with_scale_1(
        self.tmp_dir+"/{:02d}_y_1.tif".format(i), y_[i,:,:,:])
      save_image_with_scale_1(
        self.tmp_dir+"/{:02d}_y_2.tif".format(i), y__[i,:,:,:])

  def train(self):
    A_batch, B_batch = self.dh.sample_pair(self.batch_size)
    _, _,\
    l_const_1, l_const_2= \
      self.sess.run(
      [self.t1_train, self.t2_train,
       self.l_const_1, self.l_const_2
       ],
     {self.x:A_batch, self.y:B_batch})
    return l_const_1, l_const_2

