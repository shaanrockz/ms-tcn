import matplotlib._color_data as mcd
from matplotlib.collections import PatchCollection
import matplotlib
import matplotlib.pyplot as plt
import random

def read_file(path):
    with open(path, 'r') as f:
        content = f.read()
        f.close()
    return content

def main():
    dataset = "cross_task"
    split = "1"

    color_names = [name for name in mcd.CSS4_COLORS]
    color_names.remove('black')
    random.seed(786)
    random.shuffle(color_names)
    color_names.insert(0, 'black')
    print(color_names)

    gt_path = "/media/data/salam/data/"+dataset+"/groundTruth/"
    pred_path = "./results/"+dataset+"/split_"+split+"/"
    file_list = "/media/data/salam/data/"+dataset+"/splits/test.split"+split+".bundle"

    mapping_path = "/media/data/salam/data/"+dataset+"/mapping.txt"
    maps = read_file(mapping_path).split('\n')
    ann = {}
    for mapping in maps:
        m = mapping.split(" ", 1)
        if len(m)<2:
            break
        ann[m[1]] = int(m[0])

    list_of_videos = read_file(file_list).split('\n')[145:-1]

    for vid in list_of_videos:
        fig = plt.figure()
        
        gt_patch = []
        pred_patch = []

        gt_file = gt_path + vid
        gt_content = read_file(gt_file).split('\n')[0:-1]
        for i in range(len(gt_content)):
            curr = ann[gt_content[i]]
            if i>0:
                if not prev == curr or i == (len(gt_content)-1):
                    c= mcd.CSS4_COLORS[color_names[prev]]
                    gt_patch.append(matplotlib.patches.Rectangle((start_id/len(gt_content), 0), (i-start_id)/len(gt_content), 1, color = c))
                    start_id = i
                    prev = curr
            else:
                prev = curr
                start_id = 0
        
        ax = fig.add_subplot('211')
        ax.add_collection(PatchCollection(gt_patch, match_original=True))

        recog_file = pred_path + vid.split('.')[0]
        recog_content = read_file(recog_file).split('\n')[1:-1]
        for i in range(len(recog_content)):
            curr = ann[recog_content[i]]
            if i>0:
                if not prev == curr or i == (len(recog_content)-1):
                    c= mcd.CSS4_COLORS[color_names[prev]]
                    pred_patch.append(matplotlib.patches.Rectangle((start_id/len(recog_content), 0), (i-start_id)/len(recog_content), 1, color = c))
                    start_id = i
                    prev = curr
            else:
                prev = curr
                start_id = 0
        
        ax = fig.add_subplot('212')
        ax.add_collection(PatchCollection(pred_patch, match_original=True))
        plt.show()
        break

if __name__ == "__main__":
    main()