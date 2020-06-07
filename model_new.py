

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import copy
import numpy as np


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
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        self.conv_bg = nn.Conv1d(num_f_maps, 1, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        out1 = self.conv_out(out) * mask[:, 0:1, :]
        out2 = self.conv_bg(out) * mask[:, 0:1, :]
        return {"out_bg": out2,
                "out_class": out1}


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
    def __init__(self, num_blocks, num_layers, num_f_maps, dim, num_classes, batch_size):
        # self.model = MultiStageModel(
        #     num_blocks, num_layers, num_f_maps, dim, num_classes)
        self.model = SingleStageModel(num_layers, num_f_maps, dim, num_classes)
        self.ce_class = nn.CrossEntropyLoss(ignore_index=0)
        self.ce_bg = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes
        self.batch_size = batch_size

    def train(self, save_dir, data_train, num_epochs, learning_rate, device):
        self.model.train()
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            print("Current Epoch : " + str(epoch))
            epoch_loss = 0
            correct = 0
            total = 0
            count = 0

            for idx, sample_batch in enumerate(data_train):
                batch_input = sample_batch['input']
                batch_target = sample_batch["target"]
                mask = sample_batch["mask"]

                #batch_input = sample_batch[0]
                #batch_target = torch.randint(199, (32,100))
                #mask = torch.ones(32,1,100)

                count += 1
                batch_input = batch_input.to(device) 
                batch_target, mask = batch_target.to(device), mask.to(device)
                optimizer.zero_grad()
                predictions = self.model(batch_input, mask)
                p = predictions["out_class"]
                p_bg = predictions["out_bg"]

                loss = 0
                # for p in predictions:
                loss += self.ce_class(p.transpose(2, 1).contiguous().view(-1,
                                                                    self.num_classes), batch_target.view(-1))
                # print(p.size())
                # print(mask.size())
                loss += 0.15*torch.mean(torch.clamp(self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(
                    p.detach()[:, :, :-1], dim=1)), min=0, max=16)*mask[:, :, 1:])

                loss += self.ce_bg(p_bg.transpose(2, 1).contiguous().view(-1, 1), (~(batch_target.view(-1,1)>0)).float())

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
                
                #predictions = torch.cat((p_bg, p), 1)
                #print(p.size())
                #print(p_bg.size())
                predictions = p
                predictions[:, 0, :] = 1-p_bg[:, 0, :]
                _, predicted = torch.max(predictions.data, 1)
                correct += ((predicted == batch_target).float() *
                            mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()

            # data_train.reset()
            torch.save(self.model.state_dict(), save_dir +
                       "/epoch-" + str(epoch + 1) + ".model")
            torch.save(optimizer.state_dict(), save_dir +
                       "/epoch-" + str(epoch + 1) + ".opt")
            print("[epoch %d]: epoch loss = %f,   acc = %f" % (epoch + 1, epoch_loss / (count*self.batch_size),
                                                               float(correct)/total))

    def predict(self, args, model_dir, results_dir, features_path, vid_list_file, epoch, actions_dict, device, sample_rate):
        self.model.eval()
        with torch.no_grad():
            self.model.to(device)
            self.model.load_state_dict(torch.load(
                model_dir + "/epoch-" + str(epoch) + ".model"))
            file_ptr = open(vid_list_file, 'r')
            list_of_vids = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
            for vid in list_of_vids:
                print(vid)
                features = np.load(features_path + vid.split('.')[0] + '.npy')
                if args.dataset == "cross_task":
                    features = features.T
                features = features[:, ::sample_rate]
                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                predictions = self.model(
                    input_x, torch.ones(input_x.size(), device=device))

                p = predictions["out_class"]
                p_bg = predictions["out_bg"]

                predictions = p
                predictions[:,0,:] = p_bg[:,0,:]
                _, predicted = torch.max(predictions.data, 1)

                #_, predicted = torch.max(predictions[-1].data, 1)
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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tcn_dataset = TensorDataset(torch.rand(256, 3200, 100))
    tcn_dataloader = DataLoader(tcn_dataset, batch_size=32, shuffle=True, num_workers=0)

    trainer = Trainer(None, num_layers=10, num_f_maps=64, dim=3200, num_classes=200, batch_size=32)
    trainer.train(None, tcn_dataloader, num_epochs=10, learning_rate=0.0001, device=device)

if __name__ == "__main__":
    main()

