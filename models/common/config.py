import argparse


parser = argparse.ArgumentParser(description="Args of this repo.")
parser.add_argument("--rate", default=0.01, type=float)
parser.add_argument("--lr", default=0.0001, type=float)
parser.add_argument("--device", default="0")
parser.add_argument("--time", default=0, type=int)
parser.add_argument("--patch_size", default=64, type=int)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--embed_dim", default=48, type=int)
parser.add_argument("--save", default=0, type=int)
parser.add_argument("--norm", default=1, type=int)
parser.add_argument("--save_path", default=f"./trained_models")
parser.add_argument("--type", default=0,type=int)
parser.add_argument("--noise", default=0, type=float)
parser.add_argument("--folder")
parser.add_argument("--my_state_dict")
parser.add_argument("--my_log")
parser.add_argument("--my_info")

para = parser.parse_args()
if para.device == "cpu":
    para.device = "cpu"
else:
    para.device = f"cuda:{para.device}"
para.folder = f"{para.save_path}/{str(int(para.rate * 100))}/"
para.my_state_dict = f"{para.folder}/step_num_state_dict.pth"
para.my_log = f"{para.folder}/step_num_log.txt"
para.my_info = f"{para.folder}/step_num_info.pth"
