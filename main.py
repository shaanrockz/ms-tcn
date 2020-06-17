
import torch

#from batch_gen import BatchGenerator
from TCNDataset import TCNDataset
from torch.utils.data import DataLoader
import os
import argparse
import random


def collate_fn_padd(batch):
    '''
    Padds batch of variable length
    '''

    max_size = batch[0][0].size()
    trailing_dims = max_size[:1]
    max_len = max([s[0].size(1) for s in batch])
    out_dims = (int(len(batch)), int(trailing_dims[0]), int(max_len))

    out_dims_target = (int(len(batch)), int(max_len))

    out_dims_mask = (int(len(batch)), batch[0][2].size()[0], int(max_len))

    out_tensor = batch[0][0].data.new(*out_dims).fill_(0)
    out_target = batch[0][1].data.new(*out_dims_target).fill_(0)
    out_mask = batch[0][2].data.new(*out_dims_mask).fill_(0)

    for i, tensor in enumerate(batch):
        out_tensor[i, :, :tensor[0].size(1)] = tensor[0]
        out_target[i, :tensor[1].size(0)] = tensor[1]
        out_mask[i, :, :tensor[2].size(1)] = tensor[2]

    return {
        "input": out_tensor,
        "target": out_target.type(torch.LongTensor),
        "mask": out_mask,
    }


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--action', default='train')
parser.add_argument('--dataset', default="breakfast")
parser.add_argument('--split', default='1')
parser.add_argument('--seed', default=786)
parser.add_argument('--algo_type', default="baas_baseline")
args = parser.parse_args()


if args.algo_type == "baas_baseline":
    from model_current import Trainer
elif args.algo_type == "multi_task":
    from model import Trainer
elif args.algo_type == "single_task":
    from model_single_stage import Trainer


seed = int(args.seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

num_stages = 4
num_layers = 10
num_f_maps = 64

if args.dataset == "cross_task":
    features_dim = 3200
    bz = 16
    num_worker = 8
elif args.dataset == "coin":
    features_dim = 512
    bz = 16
    num_worker = 8
else:
    features_dim = 2048
    bz = 16
    num_worker = 0

#bz = 16
lr = 0.0005
num_epochs = 50

# use the full temporal resolution @ 15fps
sample_rate = 1
# sample input features @ 15fps instead of 30 fps
# for 50salads, and up-sample the output to 30 fps
if args.dataset == "50salads":
    sample_rate = 2

vid_list_file = "/media/data/salam/data/"+args.dataset + \
    "/splits/train.split"+args.split+".bundle"
vid_list_file_tst = "/media/data/salam/data/" + \
    args.dataset+"/splits/test.split"+args.split+".bundle"
features_path = "/media/data/salam/data/"+args.dataset+"/features/"
gt_path = "/media/data/salam/data/"+args.dataset+"/groundTruth/"

mapping_file = "/media/data/salam/data/"+args.dataset+"/mapping.txt"


file_ptr = open(mapping_file, 'r')
actions = file_ptr.read().split('\n')[:-1]
file_ptr.close()
actions_dict = dict()
for a in actions:
    actions_dict[a.split(' ', 1)[1]] = int(a.split(' ', 1)[0])

num_classes = len(actions_dict)

trainer = Trainer(num_stages, num_layers, num_f_maps,
                  features_dim, num_classes, bz)

model_dir = "./models/"+args.dataset+"_"+trainer.type+"_"+str(seed)+"/split_"+args.split
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

results_dir = "./results/"+args.dataset+"_"+trainer.type+"_"+str(seed)+"/split_"+args.split
if not os.path.exists(results_dir):
    os.makedirs(results_dir) 

if args.action == "train":
    tcn_dataset = TCNDataset(args, num_classes, actions_dict,
                             gt_path, features_path, sample_rate, vid_list_file)
    tcn_dataloader = DataLoader(tcn_dataset, batch_size=bz, shuffle=True,
                                num_workers=num_worker, pin_memory=True, collate_fn=collate_fn_padd)
    
    trainer.train(model_dir, tcn_dataloader, num_epochs=num_epochs,
                  learning_rate=lr, device=device)

if args.action == "predict":
    trainer.predict(args, model_dir, results_dir, features_path, vid_list_file_tst, num_epochs, actions_dict, device, sample_rate)

# For Threshold Analysis
# if args.action == "predict":
#     for i in range(5):   
#         for thres in range(1, 10):
#             results_dir = "./results/"+args.dataset+"_baas_"+str(i+1)+"/thres_"+str(thres/10)
#             if not os.path.exists(results_dir):
#                 os.makedirs(results_dir) 
#             model_dir = "./models/"+args.dataset+"_baas_"+str(i+1)+"/split_"+args.split
#             if not os.path.exists(model_dir):
#                 os.makedirs(model_dir)
#             trainer.predict(args, model_dir, results_dir, features_path,
#                             vid_list_file_tst, num_epochs, actions_dict, device, sample_rate, thres/10)
