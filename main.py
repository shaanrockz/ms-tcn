import os
import sys
import argparse
from optparse import OptionParser

import random
import numpy as np

import torch

from TCNDataset import TCNDataset
from torch.utils.data import DataLoader
from eval import evaluation
from ops import collate_fn_padd
from config import config
from threshold_evaluation import threshold_eval

# Parser options
parser = OptionParser()
parser.add_option("--seed", type=int, help="seed id", default=1)
parser.add_option("--config", type=str, help="configuration")


def main(argv):
    (opts, args) = parser.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reading config
    cfg = config(external_path=opts.config)

    if cfg.algo == "separate_background":
        from model_separate_background import Trainer
    elif cfg.algo == "no_background":
        from model_no_background import Trainer
    elif cfg.algo == "baas_time":
        from model_baas_faas import Trainer
    elif cfg.algo == "baas_attention":
        from model_baas_attention import Trainer
    elif cfg.algo == "multi_stage":
        from model import Trainer
    elif cfg.algo == "single_stage":
        from model_single_stage import Trainer

    seed = int(opts.seed)

    num_stages = 4
    num_layers = 10
    num_f_maps = 64

    if cfg.dataset.dataset == "cross_task":
        features_dim = 3200
        bz = 16
        num_worker = 8
    elif cfg.dataset.dataset == "coin":
        features_dim = 512
        bz = 16
        num_worker = 8
    else:
        features_dim = 2048
        bz = 16
        num_worker = 0

    # bz = 16
    lr = 0.0005
    num_epochs = cfg.training.epoch

    # use the full temporal resolution @ 15fps
    sample_rate = 1
    # sample input features @ 15fps instead of 30 fps
    # for 50salads, and up-sample the output to 30 fps
    if cfg.dataset.dataset == "50salads":
        sample_rate = 2

    vid_list_file = "/media/data/salam/data/" + cfg.dataset.dataset + \
        "/splits/train.split" + cfg.dataset.split+".bundle"
    vid_list_file_tst = "/media/data/salam/data/" + \
        cfg.dataset.dataset+"/splits/test.split" + cfg.dataset.split+".bundle"
    features_path = "/media/data/salam/data/" + cfg.dataset.dataset+"/features/"
    gt_path = "/media/data/salam/data/" + cfg.dataset.dataset+"/groundTruth/"
    mapping_file = "/media/data/salam/data/" + cfg.dataset.dataset+"/mapping.txt"

    with open(mapping_file, 'r') as file_ptr:
        actions = file_ptr.read().split('\n')[:-1]

    actions_dict = dict()
    for a in actions:
        actions_dict[a.split(' ', 1)[1]] = int(a.split(' ', 1)[0])
    num_classes = len(actions_dict)

    trainer = Trainer(num_stages, num_layers, num_f_maps,
                      features_dim, num_classes, bz, seed)

    model_dir = "./models/" + cfg.algo+"/" + cfg.dataset.dataset + \
        "_"+trainer.type+"_"+str(seed)+"/split_" + cfg.dataset.split
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    results_dir = "./results/" + cfg.algo+"/" + cfg.dataset.dataset + \
        "_"+trainer.type+"_"+str(seed)+"/split_" + cfg.dataset.split
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    eval_dir = "./eval/" + cfg.algo+"/" + cfg.dataset.dataset+"_" + \
        cfg.algo+"_"+str(seed)+"/split_" + cfg.dataset.split
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    if cfg.training.enable:
        tcn_dataset = TCNDataset(cfg, num_classes, actions_dict,
                                 gt_path, features_path, sample_rate, vid_list_file)
        tcn_dataloader = DataLoader(tcn_dataset, batch_size=bz, shuffle=True,
                                    num_workers=num_worker, pin_memory=True,
                                    collate_fn=collate_fn_padd)
        trainer.train(model_dir, tcn_dataloader,
                      num_epochs=num_epochs, learning_rate=lr, device=device, additional_loss=cfg.additional_loss)

    if cfg.evaluation.enable:
        if cfg.evaluation.predict:
            thres_entropy = 0.25*np.log2(num_classes)
            thres_logit = 0.6
            trainer.predict(cfg, model_dir, results_dir, features_path,
                            vid_list_file_tst, num_epochs, actions_dict, device, sample_rate,
                            thres_entropy=thres_entropy, thres_logit=thres_logit, only_entropy_prediction=cfg.evaluation_option.entropy,
                            only_logit_prediction=cfg.evaluation_option.background_probability)

        if cfg.evaluation.eval:
            evaluation(cfg, results_dir, eval_dir)

        if cfg.evaluation.threshold_analysis:
            k = 0
            thres_entropy = 0
            thres_logit = 0
            while k < 20:
                k += 1
                results_dir = "./results/" + cfg.algo + "/" + cfg.dataset.dataset+"_" + \
                    trainer.type+"_"+str(seed)+"/split_" + \
                    cfg.dataset.split+"/thres_"+str(k)
                if not os.path.exists(results_dir):
                    os.makedirs(results_dir)
                trainer.predict(cfg, model_dir, results_dir, features_path,
                                vid_list_file_tst, num_epochs, actions_dict, device, sample_rate,
                                thres_entropy=thres_entropy, thres_logit=thres_logit, only_entropy_prediction=cfg.evaluation_option.entropy,
                                only_logit_prediction=cfg.evaluation_option.background_probability)
                thres_entropy += 0.05*np.log2(num_classes-1)
                thres_logit += 0.05
            
            if cfg.evaluation_option.entropy:
                thres_type = "entropy"
            elif cfg.evaluation_option.background_probability:
                thres_type = "logit"
            
            threshold_eval(cfg.dataset.dataset, cfg.dataset.split, cfg.algo, cfg.algo, thres_type=thres_type, num_thres=20, seed=str(seed))


if __name__ == "__main__":
    main(sys.argv)
