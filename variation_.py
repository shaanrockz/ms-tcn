
# adapted from: https://github.com/colincsl/TemporalConvolutionalNetworks/blob/master/code/metrics.py

import numpy as np
import argparse


def read_file(path):
    with open(path, 'r') as f:
        content = f.read()
        f.close()
    return content


def get_labels_start_end_time(frame_wise_labels, bg_class=["background"]):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i)
    return labels, starts, ends


def levenstein(p, y, norm=False):
    m_row = len(p)    
    n_col = len(y)
    D = np.zeros([m_row+1, n_col+1], np.float)
    for i in range(m_row+1):
        D[i, 0] = i
    for i in range(n_col+1):
        D[0, i] = i

    for j in range(1, n_col+1):
        for i in range(1, m_row+1):
            if y[j-1] == p[i-1]:
                D[i, j] = D[i-1, j-1]
            else:
                D[i, j] = min(D[i-1, j] + 1,
                              D[i, j-1] + 1,
                              D[i-1, j-1] + 1)
    
    if norm:
        score = (1 - D[-1, -1]/max(m_row, n_col)) * 100
    else:
        score = D[-1, -1]

    return score


def edit_score(recognized, ground_truth, norm=True, bg_class=["background"]):
    P, _, _ = get_labels_start_end_time(recognized, bg_class)
    Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return levenstein(P, Y, norm)


def f_score(recognized, ground_truth, overlap, bg_class=["background"]):
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)

    tp = 0
    fp = 0

    hits = np.zeros(len(y_label))

    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0*intersection / union)*([p_label[j] == y_label[x] for x in range(len(y_label))])
        # Get the best scoring segment
        idx = np.array(IoU).argmax()

        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default="cross_task")
    parser.add_argument('--split', default='1')
    with_bg = False

    bg_class = ["SIL"]

    args = parser.parse_args()

    ground_truth_path = "/media/data/salam/data/"+args.dataset+"/groundTruth/"
    
    file_list = "/media/data/salam/data/"+args.dataset+"/splits/test.split"+args.split+".bundle"

    list_of_videos = read_file(file_list).split('\n')[:-1]

    arr = np.zeros((5,9))
    for k in range(5):
        for j in range(1,10):
            recog_path = "./results/"+args.dataset+"_baas_"+str(k+1)+"/thres_"+str(j/10)+"/"
            correct = 0
            total = 0

            for vid in list_of_videos:
                gt_file = ground_truth_path + vid
                gt_content = read_file(gt_file).split('\n')[0:-1]
                
                recog_file = recog_path + vid.split('.')[0]
                recog_content = read_file(recog_file).split('\n')[1:]

                for i in range(len(gt_content)):
                    if not with_bg:
                        if gt_content[i] not in bg_class:
                            total += 1
                            if gt_content[i] == recog_content[i]:
                                correct += 1
                    else:
                        total += 1
                        if gt_content[i] == recog_content[i]:
                            correct += 1
            arr[k,j-1] = 100*float(correct)/total
            #print ("Acc: "+ str(100*float(correct)/total))
    np.save('result_without_bg', arr)


if __name__ == '__main__':
    main()