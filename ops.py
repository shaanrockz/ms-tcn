import torch
import numpy as np
import torch.distributions as dist

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
    out_target = batch[0][1].data.new(*out_dims_target).fill_(-1)
    out_mask = batch[0][2].data.new(*out_dims_mask).fill_(0)

    for i, tensor in enumerate(batch):
        out_tensor[i, :, :tensor[0].size(1)] = tensor[0]
        out_target[i, :tensor[1].size(0)] = tensor[1]
        out_mask[i, :, :tensor[2].size(1)] = tensor[2]

    out_target_fg = out_target.clone()
    out_target_fg = out_target_fg-1
    out_target_fg[out_target_fg < 0] = -1

    batch_target_rel_time = None
    batch_target_rel_time_gauss = None

    # batch_target_rel_time, batch_target_rel_time_gauss = create_rel_time(
    #    out_target)

    return {
        "input": out_tensor,
        "target": out_target.type(torch.LongTensor),
        "mask": out_mask,
        "target_fg": out_target_fg.type(torch.LongTensor),
        "target_time": batch_target_rel_time,
        "target_time_gauss": batch_target_rel_time_gauss
    }


def get_gauss(x):
    std = 0.2
    m = 1
    return np.exp(-0.5*(((x-m)/std)**2))/(std*np.sqrt(2*np.pi))


def create_rel_time(target):
    time_target = target.clone().float()
    time_target_gauss = target.clone().float()

    prev_pos = 0
    curr_pos = 0
    for i in range(target.size()[0]):
        starts = [0]
        ends = []
        for j in range(target.size()[1]-1):
            if target[i, j] != target[i, j+1]:
                starts.append(j+1)
                ends.append(j)
        ends.append(target.size()[1]-1)

        assert len(starts)==len(ends)

        for k in range(len(starts)):
            start = starts[k]
            end = ends[k]
            len_activity = end-start+1
            pos=0
            val=1
            while start+pos<=end-pos:
                time_target[i,start+pos] = val
                time_target[i,end-pos] = val
                val+=1
                pos+=1
            
            assert torch.max(time_target[i, start:end+1]) == val-1
            assert torch.min(time_target[i, start:end+1]) == 1

            time_target[i, start:end+1]/=(val-1)

    time_target_gauss = get_gauss(time_target)
    return time_target, time_target_gauss

def logit_prediction(predictions, thres_logit):
    for i_batch in range(predictions.size()[0]):
        for i_frame in range(predictions.size()[2]):
            if predictions[i_batch, 0, i_frame] >= torch.tensor(thres_logit).float():
                predictions[i_batch, 1:, i_frame] = 0
            else:
                predictions[i_batch, 0, i_frame] = 0
    return predictions


def chaos_prediction(pred_fg, predictions, thres_entropy):
    for i_batch in range(pred_fg.size()[0]):
        for i_frame in range(pred_fg.size()[2]):
            entropy = dist.Categorical(
                logits=pred_fg[i_batch, :, i_frame]).entropy()
            if entropy > torch.tensor(thres_entropy).float():
                predictions[i_batch, 1:, i_frame] = 0
            else:
                predictions[i_batch, 0, i_frame] = 0
    return predictions


def chaos_logit_prediction(pred_fg, predictions, thres_entropy, thres_logit):
    for i_batch in range(pred_fg.size()[0]):
        for i_frame in range(pred_fg.size()[2]):
            entropy = dist.Categorical(
                logits=pred_fg[i_batch, :, i_frame]).entropy()
            if entropy > torch.tensor(thres_entropy).float() and predictions[i_batch, 0, i_frame] >= torch.tensor(thres_logit).float():
                predictions[i_batch, 1:, i_frame] = 0
            else:
                predictions[i_batch, 0, i_frame] = 0
    return predictions