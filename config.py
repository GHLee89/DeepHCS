import argparse

parser = argparse.ArgumentParser()
train_arg = parser.add_argument_group("train")

train_arg.add_argument("--input_size", type=int, default=256)
train_arg.add_argument("--batch_size", type=int, default=4)
train_arg.add_argument("--iter_num", type=int, default=1000000)
train_arg.add_argument("--learning_rate", type=float, default=1e-4)
train_arg.add_argument("--ckpt_dir", type=str, default="ckpt")
train_arg.add_argument("--tmp_dir", type=str, default="tmp")
train_arg.add_argument("--result_dir", type=str, default="Results/")
#train_arg.add_argument("--result_dirTN", type=str, default="Results_TN_02")
#train_arg.add_argument("--result_dirRN", type=str, default="Results_RN_02")
train_arg.add_argument("--A_path", type=str, default="<training data (x)")
train_arg.add_argument("--B_path", type=str, default="<training data (y)")
train_arg.add_argument("--name", type=str, default="input_data")

def get_config():
  return parser.parse_args()
