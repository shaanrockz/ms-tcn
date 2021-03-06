import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import copy
import numpy as np
import random
from ops import collate_fn_padd, logit_prediction 

class MultiStageModel(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes):
        super(MultiStageModel, self).__init__()
        self.stage1 = SingleStageModel(
            num_layers, num_f_maps, dim, num_classes)
        self.stages = nn.ModuleList([copy.deepcopy(SingleStageModel(
            num_layers, num_f_maps, num_classes, num_classes)) for s in range(num_stages-1)])

    def forward(self, x, mask):
        out = self.stage1(x, mask)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs


class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(
            2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes-1, 1)
        self.conv_bg = nn.Conv1d(num_f_maps, 1, 1)
        self.conv_time = nn.Conv1d(num_f_maps, 1, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        out1 = self.conv_out(out) * mask[:, 0:1, :]
        out2 = self.conv_bg(out) * mask[:, 0:1, :]
        out3 = self.conv_time(out) * mask[:, 0:1, :]
        return {"out_bg": out2,
                "out_class": out1,
                "out_time": out3}


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(
            in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x, mask):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]


class Trainer:
    def __init__(self, num_blocks, num_layers, num_f_maps, dim, num_classes, batch_size, seed, debugging = False):
        # self.model = MultiStageModel(
        #     num_blocks, num_layers, num_f_maps, dim, num_classes)
        self.model = SingleStageModel(num_layers, num_f_maps, dim, num_classes)
        self.ce_class = nn.CrossEntropyLoss(ignore_index=-1)
        self.ce_bg = nn.BCEWithLogitsLoss(reduction='none')
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.type = "baas_time"
        self.debugging = debugging
        self.seed = seed

    def train(self, save_dir, data_train, num_epochs, learning_rate, device):
        self.model.train()
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            print("Current Epoch : " + str(epoch))
            epoch_loss = 0
            correct = 0
            correct_bg = 0
            total = 0
            total_bg = 0
            count = 0

            # rand(self.seed)
            for idx, sample_batch in enumerate(data_train):
                batch_input = sample_batch['input']
                batch_target = sample_batch["target"]
                batch_target_fg = sample_batch["target_fg"]
                mask = sample_batch["mask"]
                batch_target_rel_time = sample_batch["target_time"]
                batch_target_rel_time_gauss = sample_batch["target_time_gauss"]

                count += 1
                batch_input = batch_input.to(device) 
                batch_target_fg = batch_target_fg.to(device)
                batch_target_rel_time = batch_target_rel_time.to(device)
                batch_target_rel_time_gauss = batch_target_rel_time_gauss.to(device)
                batch_target, mask = batch_target.to(device), mask.to(device)
                optimizer.zero_grad()

                predictions = self.model(batch_input, mask)
                pred_fg = predictions["out_class"]
                pred_bg = predictions["out_bg"]
                pred_time = predictions["out_time"]
                #print(pred_fg.shape)
                #print(pred_time.shape)

                loss = 0

                mask_background = mask[:,:1,:]
                loss_bg = self.ce_bg(pred_bg.transpose(2, 1).contiguous().view(-1, 1), (batch_target.view(-1, 1)==0).float())
                loss_bg *= mask_background.contiguous().view(-1,1)
                loss_bg = torch.mean(loss_bg)

                loss_fg = self.ce_class(pred_fg.transpose(2, 1).contiguous().view(-1, self.num_classes-1), batch_target_fg.view(-1))
                
                #print(loss_bg.item())
                #print(loss_fg.item())
                p_cat = torch.cat((pred_bg, pred_fg), 1)
                loss += 0.15*torch.mean(torch.clamp(self.mse(F.log_softmax(p_cat[:, :, 1:], dim=1), F.log_softmax(
                    p_cat.detach()[:, :, :-1], dim=1)), min=0, max=16)*mask[:, :, 1:])

                square_err_time = (torch.sigmoid(pred_time.squeeze())-batch_target_rel_time)**2
                err_time = torch.abs(torch.sigmoid(pred_time.squeeze())-batch_target_rel_time)
                gauss_amp_err = batch_target_rel_time_gauss*square_err_time

                masked_err = square_err_time*(batch_target!=0).float()
                thres = 0.1 
                loss_time = torch.max(torch.tensor(0.).cuda(), 100*((torch.sum(masked_err)/torch.sum(mask[:,0,:])) - thres)) - 100*torch.std(masked_err)
                # print(loss_time)
                # x= batch_target_rel_time.shape[0]*batch_target_rel_time.shape[1]
                # print(torch.max(batch_target_rel_time))
                
                # print(torch.sum(masked_err)/torch.sum(mask[:,0,:]))

                loss = loss + loss_bg + loss_fg + loss_time

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
                
                if self.debugging:
                    if (epoch+1)%10 == 0:
                        print(torch.sigmoid(pred_time[0,0,100:150]))
                        print(batch_target_rel_time[0,100:150]*(batch_target[0,100:150]>0).float())
                    predictions = torch.sigmoid(p_cat.data)
                    predictions = logit_prediction(predictions, 0.5)

                    _, predicted = torch.max(predictions, 1)
                    correct += ((predicted == batch_target).float() * (batch_target>0).float()
                                ).sum().item()
                    correct_bg += ((predicted == batch_target).float() * (batch_target==0)*mask[:,0,:].float()
                                ).sum().item()
                    total += torch.sum((batch_target>0).float()*mask[:,0,:]).item()
                    total_bg += (torch.sum((batch_target==0).float()*mask[:,0,:])).item()

            if not self.debugging:
                torch.save(self.model.state_dict(), save_dir +
                        "/epoch-" + str(epoch + 1) + ".model")
                torch.save(optimizer.state_dict(), save_dir +
                        "/epoch-" + str(epoch + 1) + ".opt")
            print("[epoch %d]: epoch loss = %f" % (epoch, epoch_loss / (count)))
            if self.debugging:
                print("[epoch %d]: acc_fg = %f, acc_bg = %f" % (epoch, float(correct)/total, float(correct_bg)/total_bg))

    def predict(self, args, model_dir, results_dir, features_path, vid_list_file, epoch, actions_dict, device, sample_rate, thres_logit=0.6, thres_entropy=2, only_logit_prediction=False, only_entropy_prediction=True):
        self.model.eval()
        with torch.no_grad():
            self.model.to(device)
            self.model.load_state_dict(torch.load(
                model_dir + "/epoch-" + str(epoch) + ".model"))
            file_ptr = open(vid_list_file, 'r')
            list_of_vids = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
            for vid in list_of_vids:
                vid = vid.split('/')[-1]
                features = np.load(features_path + vid.split('.')[0] + '.npy')
                if args.dataset in ["cross_task", "coin"]:
                    features = features.T
                features = features[:, ::sample_rate]
                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                predictions = self.model(
                    input_x, torch.ones(input_x.size(), device=device))

                pred_fg = torch.sigmoid(predictions["out_class"])
                pred_bg = torch.sigmoid(predictions["out_bg"])

                predictions = torch.cat((pred_bg, pred_fg), 1).data
                predictions = logit_prediction(predictions, thres_logit)

                _, predicted = torch.max(predictions.data, 1)

                predicted = predicted.squeeze()
                recognition = []
                for i in range(len(predicted)):
                    recognition = np.concatenate((recognition, [list(actions_dict.keys())[
                                                 list(actions_dict.values())[predicted[i].item()]]+"\n"]*sample_rate))
                f_name = vid.split('/')[-1].split('.')[0]
                f_ptr = open(results_dir + "/" + f_name, "w")
                f_ptr.write("### Frame level recognition: ###\n")
                f_ptr.write(''.join(recognition))
                f_ptr.close()


from torch.utils.data import DataLoader, TensorDataset
from TCNDataset import TCNDataset

def main():
    seed = 786
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    dataset = "cross_task"
    if dataset == "cross_task":
        features_dim = 3200
        bz = 16
        num_worker = 0
    elif dataset == "coin":
        features_dim = 512
        bz = 16
        num_worker = 0
    else:
        features_dim = 2048
        bz = 16
        num_worker = 0


    # use the full temporal resolution @ 15fps
    sample_rate = 1

    vid_list_file = "/media/data/salam/data/"+dataset + \
        "/splits/train.split1.bundle"
    features_path = "/media/data/salam/data/"+dataset+"/features/"
    gt_path = "/media/data/salam/data/"+dataset+"/groundTruth/"

    mapping_file = "/media/data/salam/data/"+dataset+"/mapping.txt"


    file_ptr = open(mapping_file, 'r')
    actions = file_ptr.read().split('\n')[:-1]
    file_ptr.close()
    actions_dict = dict()
    for a in actions:
        actions_dict[a.split(' ', 1)[1]] = int(a.split(' ', 1)[0])

    num_classes = len(actions_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class Args():
        def __init__(self, dataset):
            self.dataset=dataset

    args = Args(dataset)

    tcn_dataset = TCNDataset(args, num_classes, actions_dict,
                             gt_path, features_path, sample_rate, vid_list_file, debugging=True)
    
    tcn_dataloader = DataLoader(tcn_dataset, batch_size=bz, shuffle=True,
                                num_workers=num_worker, pin_memory=True, collate_fn=collate_fn_padd)

    # tcn_dataset = TensorDataset(torch.rand(256, 3200, 100))
    # tcn_dataloader = DataLoader(tcn_dataset, batch_size=16, shuffle=True, num_workers=0)

    trainer = Trainer(None, num_layers=10, num_f_maps=64, dim=features_dim, num_classes=num_classes, batch_size=bz, seed=1, debugging=True)
    trainer.train(None, tcn_dataloader, num_epochs=120, learning_rate=0.001, device=device)

if __name__ == "__main__":
    main()

