import tensorflow as tf
from ops import *

class RefineNet(object):
  def __init__(self, name, out, is_train=True):
    self.name = name
    self.reuse = None
    self.out = out
    self.is_train = is_train

  def __call__(self, x, act_fn=tf.nn.relu, end_act_fn=tf.nn.relu):
    input_shape = x.get_shape().as_list()

    nc = 64
    with tf.variable_scope(self.name) as vs:
      ######## Refinement Network
      x = CONV2D(x, nc, 3, 1, "d1_1", "SAME", True, act_fn, self.is_train)
      sk1 = x
      x = CONV2D(x, nc, 3, 1, "r1_1", "SAME", True, act_fn, self.is_train)
      x = CONV2D(x, nc, 3, 1, "r1_2", "SAME", True, act_fn, self.is_train)
      x = CONV2D(x, nc, 3, 1, "r1_3", "SAME", True, act_fn, self.is_train)
      x = tf.add(sk1, x, "add1")
      x = CONV2D(x, nc, 3, 1, "d1_2", "SAME", True, act_fn, self.is_train) 
      d4 = x
      x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], "VALID")
 
      x = CONV2D(x, nc*2, 3, 1, "d2_1", "SAME", True, act_fn, self.is_train)
      sk2 = x
      x = CONV2D(x, nc*2, 3, 1, "r2_1", "SAME", True, act_fn, self.is_train)
      x = CONV2D(x, nc*2, 3, 1, "r2_2", "SAME", True, act_fn, self.is_train)
      x = CONV2D(x, nc*2, 3, 1, "r2_3", "SAME", True, act_fn, self.is_train)
      x = tf.add(sk2, x, "add2")
      x = CONV2D(x, nc*2, 3, 1, "d2_2", "SAME", True, act_fn, self.is_train) 
      d5= x
      x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], "VALID")

      x = CONV2D(x, nc*4, 3, 1, "d3_1", "SAME", True, act_fn, self.is_train)
      sk3 = x
      x = CONV2D(x, nc*4, 3, 1, "r3_1", "SAME", True, act_fn, self.is_train)
      x = CONV2D(x, nc*4, 3, 1, "r3_2", "SAME", True, act_fn, self.is_train)
      x = CONV2D(x, nc*4, 3, 1, "r3_3", "SAME", True, act_fn, self.is_train)
      x = tf.add(sk3, x, "add3")
      x = CONV2D(x, nc*4, 3, 1, "d3_2", "SAME", True, act_fn, self.is_train) 
      d6 = x
      x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], "VALID")

      x = CONV2D(x, nc*8, 3, 1, "d4_1", "SAME", True, act_fn, self.is_train)
      sk4 = x
      x = CONV2D(x, nc*8, 3, 1, "r4_1", "SAME", True, act_fn, self.is_train)
      x = CONV2D(x, nc*8, 3, 1, "r4_2", "SAME", True, act_fn, self.is_train)
      x = CONV2D(x, nc*8, 3, 1, "r4_3", "SAME", True, act_fn, self.is_train)
      x = tf.add(sk4, x, "add4")
      x = CONV2D(x, nc*8, 3, 1, "d4_2", "SAME", True, act_fn, self.is_train) 
      d7 = x
      x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], "VALID")

      x = CONV2D(x, nc*16, 3, 1, "m1_1", "SAME", True, act_fn, self.is_train)
      sk5 = x
      x = CONV2D(x, nc*16, 3, 1, "r5_1", "SAME", True, act_fn, self.is_train)
      x = CONV2D(x, nc*16, 3, 1, "r5_2", "SAME", True, act_fn, self.is_train)
      x = CONV2D(x, nc*16, 3, 1, "r5_3", "SAME", True, act_fn, self.is_train)
      x = tf.add(sk5, x, "add5")
      x = CONV2D(x, nc*16, 3, 1, "m1_2", "SAME", True, act_fn, self.is_train) 

      x = upscale(x, 2)
      x = CONV2D(x, nc*8, 3, 1, "u1_1", "SAME", True, tf.nn.relu, self.is_train)
      x = tf.add(d7, x, "add6")
      x = CONV2D(x, nc*8, 3, 1, "u1_2", "SAME", True, tf.nn.relu, self.is_train)
      sk6 = x
      x = CONV2D(x, nc*8, 3, 1, "u1_3", "SAME", True, tf.nn.relu, self.is_train)
      x = CONV2D(x, nc*8, 3, 1, "u1_4", "SAME", True, tf.nn.relu, self.is_train)
      x = CONV2D(x, nc*8, 3, 1, "u1_5", "SAME", True, tf.nn.relu, self.is_train)
      x = tf.add(sk6, x, "add7")
      x = CONV2D(x, nc*8, 3, 1, "u1_6", "SAME", True, tf.nn.relu, self.is_train)

      x = upscale(x, 2)
      x = CONV2D(x, nc*4, 3, 1, "u2_1", "SAME", True, tf.nn.relu, self.is_train)
      x = tf.add(d6, x, "add8")
      x = CONV2D(x, nc*4, 3, 1, "u2_2", "SAME", True, tf.nn.relu, self.is_train)
      sk7 = x
      x = CONV2D(x, nc*4, 3, 1, "u2_3", "SAME", True, tf.nn.relu, self.is_train)
      x = CONV2D(x, nc*4, 3, 1, "u2_4", "SAME", True, tf.nn.relu, self.is_train)
      x = CONV2D(x, nc*4, 3, 1, "u2_5", "SAME", True, tf.nn.relu, self.is_train)
      x = tf.add(sk7, x, "add9")
      x = CONV2D(x, nc*4, 3, 1, "u2_6", "SAME", True, tf.nn.relu, self.is_train)

      x = upscale(x, 2)
      x = CONV2D(x, nc*2, 3, 1, "u3_1", "SAME", True, tf.nn.relu, self.is_train)
      x = tf.add(d5, x, "add10")
      x = CONV2D(x, nc*2, 3, 1, "u3_2", "SAME", True, tf.nn.relu, self.is_train)
      sk8 = x
      x = CONV2D(x, nc*2, 3, 1, "u3_3", "SAME", True, tf.nn.relu, self.is_train)
      x = CONV2D(x, nc*2, 3, 1, "u3_4", "SAME", True, tf.nn.relu, self.is_train)
      x = CONV2D(x, nc*2, 3, 1, "u3_5", "SAME", True, tf.nn.relu, self.is_train)
      x = tf.add(sk8, x, "add11")
      x = CONV2D(x, nc*2, 3, 1, "u3_6", "SAME", True, tf.nn.relu, self.is_train)

      x = upscale(x, 2)
      x = CONV2D(x, nc, 3, 1, "u4_1", "SAME", True, tf.nn.relu, self.is_train)
      x = tf.add(d4, x, "add12")
      x = CONV2D(x, nc, 3, 1, "u4_2", "SAME", True, tf.nn.relu, self.is_train)
      sk9 = x
      x = CONV2D(x, nc, 3, 1, "u4_3", "SAME", True, tf.nn.relu, self.is_train)
      x = CONV2D(x, nc, 3, 1, "u4_4", "SAME", True, tf.nn.relu, self.is_train)
      x = CONV2D(x, nc, 3, 1, "u4_5", "SAME", True, tf.nn.relu, self.is_train)
      x = tf.add(sk9, x, "add13")
      x = CONV2D(x, nc, 3, 1, "u4_6", "SAME", True, tf.nn.relu, self.is_train)

      x = CONV2D(x, self.out, 3, 1, "out", "SAME", False, end_act_fn, self.is_train)

      if self.reuse is None:
        self.reuse = True
        self.var_list = tf.contrib.framework.get_variables(vs)
        self.saver = tf.train.Saver(self.var_list)

    return x

  def save(self, sess, ckpt_path):
    self.saver.save(sess, ckpt_path)

  def restore(self, sess, ckpt_path):
    self.saver.restore(sess, ckpt_path)


class Transform_prep(object):
  def __init__(self, name, out, is_train=True):
    self.name = name
    self.reuse = None
    self.out = out
    self.is_train = is_train

  def __call__(self, x, act_fn=tf.nn.relu, end_act_fn=tf.nn.relu):
    input_shape = x.get_shape().as_list()

    p_nc = 16
    nc = 64
    with tf.variable_scope(self.name) as vs:
      ######### Pre-processor
      x = CONV2D(x, p_nc, 3, 1, "d1_1", "SAME", True, act_fn, self.is_train)
      x = CONV2D(x, p_nc, 3, 1, "d1_2", "SAME", True, act_fn, self.is_train)
      d1 = x
      x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], "VALID")

      x = CONV2D(x, p_nc*2, 3, 1, "d2_1", "SAME", True, act_fn, self.is_train)
      x = CONV2D(x, p_nc*2, 3, 1, "d2_2", "SAME", True, act_fn, self.is_train)
      d2 = x
      x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], "VALID")
      
      x = CONV2D(x, p_nc*4, 3, 1, "d3_1", "SAME", True, act_fn, self.is_train)
      x = CONV2D(x, p_nc*4, 3, 1, "d3_2", "SAME", True, act_fn, self.is_train)
      d3 = x
      x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], "VALID")

      x = CONV2D(x, p_nc*8, 3, 1, "m1_1", "SAME", True, act_fn, self.is_train)
      x = CONV2D(x, p_nc*8, 3, 1, "m1_2", "SAME", True, act_fn, self.is_train)

      x = upscale(x, 2) # u1
      x = tf.concat([d3, x], 3) # merge 1
      x = CONV2D(x, p_nc*4, 3, 1, "u1", "SAME", True, tf.nn.relu, self.is_train)
      x = CONV2D(x, p_nc*4, 3, 1, "u2", "SAME", True, tf.nn.relu, self.is_train)
      
      x = upscale(x, 2) # u2
      x = tf.concat([d2, x], 3) # merge 2
      x = CONV2D(x, p_nc*2, 3, 1, "u3", "SAME", True, tf.nn.relu, self.is_train)
      x = CONV2D(x, p_nc*2, 3, 1, "u4", "SAME", True, tf.nn.relu, self.is_train)
      
      x = upscale(x, 2) # u3
      x = tf.concat([d1, x], 3) # merge 3
      x = CONV2D(x, p_nc, 3, 1, "u5", "SAME", True, tf.nn.relu, self.is_train)
      x = CONV2D(x, p_nc, 3, 1, "u6", "SAME", True, tf.nn.relu, self.is_train)
     
      ######## Transformation Network
      x = CONV2D(x, nc, 3, 1, "d4_1", "SAME", True, act_fn, self.is_train)
      sk1 = x
      x = CONV2D(x, nc, 3, 1, "r1_1", "SAME", True, act_fn, self.is_train)
      x = CONV2D(x, nc, 3, 1, "r1_2", "SAME", True, act_fn, self.is_train)
      x = CONV2D(x, nc, 3, 1, "r1_3", "SAME", True, act_fn, self.is_train)
      x = tf.add(sk1, x, "add1")
      x = CONV2D(x, nc, 3, 1, "d4_2", "SAME", True, act_fn, self.is_train) 
      d4 = x
      x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], "VALID")
 
      x = CONV2D(x, nc*2, 3, 1, "d5_1", "SAME", True, act_fn, self.is_train)
      sk2 = x
      x = CONV2D(x, nc*2, 3, 1, "r2_1", "SAME", True, act_fn, self.is_train)
      x = CONV2D(x, nc*2, 3, 1, "r2_2", "SAME", True, act_fn, self.is_train)
      x = CONV2D(x, nc*2, 3, 1, "r2_3", "SAME", True, act_fn, self.is_train)
      x = tf.add(sk2, x, "add2")
      x = CONV2D(x, nc*2, 3, 1, "d5_2", "SAME", True, act_fn, self.is_train) 
      d5= x
      x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], "VALID")

      x = CONV2D(x, nc*4, 3, 1, "d6_1", "SAME", True, act_fn, self.is_train)
      sk3 = x
      x = CONV2D(x, nc*4, 3, 1, "r3_1", "SAME", True, act_fn, self.is_train)
      x = CONV2D(x, nc*4, 3, 1, "r3_2", "SAME", True, act_fn, self.is_train)
      x = CONV2D(x, nc*4, 3, 1, "r3_3", "SAME", True, act_fn, self.is_train)
      x = tf.add(sk3, x, "add3")
      x = CONV2D(x, nc*4, 3, 1, "d6_2", "SAME", True, act_fn, self.is_train) 
      d6 = x
      x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], "VALID")

      x = CONV2D(x, nc*8, 3, 1, "d7_1", "SAME", True, act_fn, self.is_train)
      sk4 = x
      x = CONV2D(x, nc*8, 3, 1, "r4_1", "SAME", True, act_fn, self.is_train)
      x = CONV2D(x, nc*8, 3, 1, "r4_2", "SAME", True, act_fn, self.is_train)
      x = CONV2D(x, nc*8, 3, 1, "r4_3", "SAME", True, act_fn, self.is_train)
      x = tf.add(sk4, x, "add4")
      x = CONV2D(x, nc*8, 3, 1, "d7_2", "SAME", True, act_fn, self.is_train) 
      d7 = x
      x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], "VALID")

      x = CONV2D(x, nc*16, 3, 1, "d8_1", "SAME", True, act_fn, self.is_train)
      sk5 = x
      x = CONV2D(x, nc*16, 3, 1, "r5_1", "SAME", True, act_fn, self.is_train)
      x = CONV2D(x, nc*16, 3, 1, "r5_2", "SAME", True, act_fn, self.is_train)
      x = CONV2D(x, nc*16, 3, 1, "r5_3", "SAME", True, act_fn, self.is_train)
      x = tf.add(sk5, x, "add5")
      x = CONV2D(x, nc*16, 3, 1, "d8_2", "SAME", True, act_fn, self.is_train) 
      mid_feature = x

      x = upscale(x, 2)
      x = CONV2D(x, nc*8, 3, 1, "u7_1", "SAME", True, tf.nn.relu, self.is_train)
      x = tf.add(d7, x, "add6")
      x = CONV2D(x, nc*8, 3, 1, "u7_2", "SAME", True, tf.nn.relu, self.is_train)
      sk6 = x
      x = CONV2D(x, nc*8, 3, 1, "u7_3", "SAME", True, tf.nn.relu, self.is_train)
      x = CONV2D(x, nc*8, 3, 1, "u7_4", "SAME", True, tf.nn.relu, self.is_train)
      x = CONV2D(x, nc*8, 3, 1, "u7_5", "SAME", True, tf.nn.relu, self.is_train)
      x = tf.add(sk6, x, "add7")
      x = CONV2D(x, nc*8, 3, 1, "u7_6", "SAME", True, tf.nn.relu, self.is_train)

      x = upscale(x, 2)
      x = CONV2D(x, nc*4, 3, 1, "u8_1", "SAME", True, tf.nn.relu, self.is_train)
      x = tf.add(d6, x, "add8")
      x = CONV2D(x, nc*4, 3, 1, "u8_2", "SAME", True, tf.nn.relu, self.is_train)
      sk7 = x
      x = CONV2D(x, nc*4, 3, 1, "u8_3", "SAME", True, tf.nn.relu, self.is_train)
      x = CONV2D(x, nc*4, 3, 1, "u8_4", "SAME", True, tf.nn.relu, self.is_train)
      x = CONV2D(x, nc*4, 3, 1, "u8_5", "SAME", True, tf.nn.relu, self.is_train)
      x = tf.add(sk7, x, "add9")
      x = CONV2D(x, nc*4, 3, 1, "u8_6", "SAME", True, tf.nn.relu, self.is_train)

      x = upscale(x, 2)
      x = CONV2D(x, nc*2, 3, 1, "u9_1", "SAME", True, tf.nn.relu, self.is_train)
      x = tf.add(d5, x, "add10")
      x = CONV2D(x, nc*2, 3, 1, "u9_2", "SAME", True, tf.nn.relu, self.is_train)
      sk8 = x
      x = CONV2D(x, nc*2, 3, 1, "u9_3", "SAME", True, tf.nn.relu, self.is_train)
      x = CONV2D(x, nc*2, 3, 1, "u9_4", "SAME", True, tf.nn.relu, self.is_train)
      x = CONV2D(x, nc*2, 3, 1, "u9_5", "SAME", True, tf.nn.relu, self.is_train)
      x = tf.add(sk8, x, "add11")
      x = CONV2D(x, nc*2, 3, 1, "u9_6", "SAME", True, tf.nn.relu, self.is_train)

      x = upscale(x, 2)
      x = CONV2D(x, nc, 3, 1, "u10_1", "SAME", True, tf.nn.relu, self.is_train)
      x = tf.add(d4, x, "add12")
      x = CONV2D(x, nc, 3, 1, "u10_2", "SAME", True, tf.nn.relu, self.is_train)
      sk9 = x
      x = CONV2D(x, nc, 3, 1, "u10_3", "SAME", True, tf.nn.relu, self.is_train)
      x = CONV2D(x, nc, 3, 1, "u10_4", "SAME", True, tf.nn.relu, self.is_train)
      x = CONV2D(x, nc, 3, 1, "u10_5", "SAME", True, tf.nn.relu, self.is_train)
      x = tf.add(sk9, x, "add13")
      x = CONV2D(x, nc, 3, 1, "u10_6", "SAME", True, tf.nn.relu, self.is_train)

      x = CONV2D(x, self.out, 3, 1, "out2", "SAME", False, end_act_fn, self.is_train)

      if self.reuse is None:
        self.reuse = True
        self.var_list = tf.contrib.framework.get_variables(vs)
        self.saver = tf.train.Saver(self.var_list)

    return x 

  def save(self, sess, ckpt_path):
    self.saver.save(sess, ckpt_path)

  def restore(self, sess, ckpt_path):
    self.saver.restore(sess, ckpt_path)

