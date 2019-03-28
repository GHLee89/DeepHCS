import tensorflow as tf
import numpy as np
tf.set_random_seed(2019)
np.random.seed(2019)

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from trainer import Trainer
from data import ISBIHandler

from ops import mkdir
from config import get_config

def main():
  config = get_config()

  A_path = config.A_path
  B_path = config.B_path
  dh = ISBIHandler(A_path, B_path, config.input_size)

  mkdir(config.ckpt_dir)
  mkdir(config.ckpt_dir+"/ckpt_T1")
  mkdir(config.ckpt_dir+"/ckpt_T2")
  mkdir(config.tmp_dir)

  with tf.Session() as sess:
    trainer = Trainer(sess, dh, config)
    #trainer.restore()

    iter_ = 0 
    d_stop = False
    while iter_ < config.iter_num:
      results = trainer.train()

      print(iter_+1, results)

      if (iter_+1) % 100 == 0:
        trainer.save()
        trainer.deploy()
      iter_ += 1

if __name__=="__main__":
  main()
