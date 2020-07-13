import torch
import numpy as np
import random
from torch.utils.data import Dataset

class TCNDataset(Dataset):
    def __init__(self, args, num_classes, actions_dict, gt_path, features_path, sample_rate, vid_list_file, debugging=False):
        self.list_of_examples = list()
        self.index = 0
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.gt_path = gt_path
        self.features_path = features_path
        self.sample_rate = sample_rate
        file_ptr = open(vid_list_file, 'r')
        self.list_of_examples = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        random.shuffle(self.list_of_examples)
        if debugging:
            self.list_of_examples = self.list_of_examples[:128]
        self.args= args

    def __len__(self):
        return len(self.list_of_examples)

    def __getitem__(self, idx):
        sample = self.list_of_examples[idx].split('/')[-1]
        features = np.load(self.features_path + sample.split('.')[0] + '.npy')
        if self.args.dataset.dataset in ["cross_task", "coin"]:
            features = features.T
        file_ptr = open(self.gt_path + sample, 'r')
        content = file_ptr.read().split('\n')[:-1]
        classes = np.zeros(min(np.shape(features)[1], len(content)))
        for i in range(len(classes)):
            classes[i] = self.actions_dict[content[i]]
        batch_input = features[:, ::self.sample_rate]
        batch_target = classes[::self.sample_rate]

        length_of_sequences = len(batch_target)
        #length_of_sequences = 10000
        batch_input_tensor = torch.zeros(np.shape(batch_input)[0], length_of_sequences, dtype=torch.float)
        batch_target_tensor = torch.ones(length_of_sequences, dtype=torch.float)*(-100)
        mask = torch.zeros(self.num_classes, length_of_sequences, dtype=torch.float)
        #print(np.shape(batch_input_tensor))
        #print(np.shape(batch_input))
        
        batch_input_tensor[:, :np.shape(batch_input)[1]] = torch.from_numpy(batch_input)
        batch_target_tensor[:np.shape(batch_target)[0]] = torch.from_numpy(batch_target)
        mask[:, :np.shape(batch_target)[0]] = torch.ones(self.num_classes, np.shape(batch_target)[0])

        return batch_input_tensor, batch_target_tensor, mask
        """ return {
               "input": batch_input_tensor,
               "target": batch_target_tensor.type(torch.LongTensor),
               "mask": mask,
           } """
