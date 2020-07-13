# Run Command : streamlit run --server.enableCORS false widget.py

import matplotlib._color_data as mcd
from matplotlib.collections import PatchCollection
import matplotlib
import matplotlib.pyplot as plt
import random
import streamlit as st
import altair as alt
import pandas as pd
from PIL import Image
from numba import njit

dataset_source = st.sidebar.selectbox(
    "Choose Dataset", ["cross_task", "breakfast"])

bg_class = ["SIL"]

pred_path = "/media/remote_home/salam/Documents/Workspace_action_segmentation/ms-tcn/results/"
plot_file = "/media/remote_home/salam/Documents/Workspace_action_segmentation/Report/plot/"

model_type = st.sidebar.selectbox("Choose Model", ["baas_chaos"])

pred_type = st.sidebar.selectbox("Prediction Type", ["entropy", 'logit'])
pred_path += "Chaos_"+pred_type+"_thres/" + \
    dataset_source+"_"+model_type+"_1/split_1/"
plot_file += dataset_source+"_"+model_type+"_"+pred_type+"_threshold.png"

split = "1"

st.text("[Dataset : "+dataset_source + "] [Model Type : " +
        model_type+"] [Threshold Param : "+pred_type+"]")

gt_path = "/media/data/salam/data/"+dataset_source+"/groundTruth/"
file_list = "/media/data/salam/data/" + \
    dataset_source+"/splits/test.split"+split+".bundle"
mapping_path = "/media/data/salam/data/"+dataset_source+"/mapping.txt"


def read_file(path):
    with open(path, 'r') as f:
        content = f.read()
        f.close()
    return content


@st.cache
def get_vid_list():
    color_names = [name for name in mcd.CSS4_COLORS]
    color_names.remove('black')
    random.seed(786)
    random.shuffle(color_names)
    color_names.insert(0, 'black')

    maps = read_file(mapping_path).split('\n')
    ann = {}
    dataset_mapping = {}
    for mapping in maps:
        m = mapping.split(" ", 1)
        if len(m) < 2:
            break
        ann[m[1]] = int(m[0])
        dataset_mapping[int(m[0])] = m[1]

    list_of_videos = read_file(file_list).split('\n')[:-1]
    return list_of_videos, ann, dataset_mapping, color_names

list_of_videos, ann, dataset_mapping, color_names = get_vid_list()

@st.cache
def get_pred(dataset_source, model_type, pred_type):
    correct = 0
    total = 0
    correct_bg = 0
    total_bg = 0
    if pred_type == "entropy":
        th = 6
    else:
        th = 13
    for vid in list_of_videos:
        gt_file = gt_path + vid
        gt_content = read_file(gt_file).split('\n')[0:-1]

        recog_file = pred_path+"thres_"+str(th)+"/" + vid.split('.')[0]
        recog_content = read_file(recog_file).split('\n')[1:-1]

        for i in range(len(gt_content)):
            if gt_content[i] not in bg_class:
                total_bg += 1
                if gt_content[i] == recog_content[i]:
                    correct_bg += 1
            total += 1
            if gt_content[i] == recog_content[i]:
                correct += 1
    return 100*float(correct)/total, 100*float(correct_bg)/total_bg

mof_, mof_bg_ = get_pred(dataset_source, model_type, pred_type)

st.text("Overall Performance : [MoF : "+"{:.2f}".format(
    mof_) + "] [MoF-Bacground : "+"{:.2f}".format(mof_bg_)+"]")

st.text("[25% and 60% threshold considered for entropy and logit respectively]")


image = Image.open(plot_file)
st.image(image, caption='Threshold Variation of ' +
         pred_type, use_column_width=True)


def create_segmentation_chart(data, mapping, action_names, color_names) -> alt.Chart:
    result = []
    for seg in data.keys():
        frame_labels = data[seg]
        labels, lengths = summarize_list(frame_labels)

        for i, (ll, ln) in enumerate(zip(labels, lengths)):
            result.append([i, mapping[ll], ln, seg])

    custom_data = pd.DataFrame(
        data=result, columns=["order", "action", "length", "segmentation"]
    )

    segmentation_chart = (
        alt.Chart(custom_data)
        .mark_bar()
        .encode(
            x="length:Q",
            y="segmentation:N",
            # color="action:N",
            color = alt.Color('action',
                    scale=alt.Scale(
                        domain=list(action_names),
                        range=color_names)),
            order=alt.Order("order", sort="ascending"),
            tooltip=["action:N"],
        )
    ).properties(height=200, width=800)

    return segmentation_chart

@njit
def summarize_list(the_list):
    """
    Given a list of items, it summarizes them in a way that no two neighboring values are the same.
    It also returns the size of each section.
    e.g. [4, 5, 5, 6] -> [4, 5, 6], [1, 2, 1]
    """
    summary = []
    lens = []
    if len(the_list) > 0:
        current = the_list[0]
        summary.append(current)
        lens.append(1)
        for item in the_list[1:]:
            if item != current:
                current = item
                summary.append(item)
                lens.append(1)
            else:
                lens[-1] += 1
    return summary, lens

def plot_data(vid, ann, color_names, thres):
    vid = vid.split('/')[-1]

    gt_patch = []
    pred_patch = []

    gt_file = gt_path + vid.split('/')[-1]
    gt_content = read_file(gt_file).split('\n')[0:-1]
    for i in range(len(gt_content)):
        curr = ann[gt_content[i]]
        if i > 0:
            if not prev == curr or i == (len(gt_content)-1):
                c = mcd.CSS4_COLORS[color_names[prev % len(color_names)]]
                gt_patch.append(matplotlib.patches.Rectangle(
                    (start_id/len(gt_content), 0), (i-start_id)/len(gt_content), 1, color=c))
                start_id = i
                prev = curr
        else:
            prev = curr
            start_id = 0

    fig = plt.figure(figsize=(8, 6))

    ax = fig.add_subplot('211')
    ax.add_collection(PatchCollection(gt_patch, match_original=True))
    ax.title.set_text('Ground Truth')
    plt.axis('off')

    recog_file = pred_path+"thres_"+str(thres)+"/" + vid.split('.')[0]
    recog_content = read_file(recog_file).split('\n')[1:-1]
    for i in range(len(recog_content)):
        curr = ann[recog_content[i]]
        if i > 0:
            if not prev == curr or i == (len(recog_content)-1):
                c = mcd.CSS4_COLORS[color_names[prev % len(color_names)]]
                pred_patch.append(matplotlib.patches.Rectangle(
                    (start_id/len(recog_content), 0), (i-start_id)/len(recog_content), 1, color=c))
                start_id = i
                prev = curr
        else:
            prev = curr
            start_id = 0

    ax = fig.add_subplot('212')
    ax.add_collection(PatchCollection(pred_patch, match_original=True))
    ax.title.set_text('Prediction')
    plt.axis('off')

    total_bg = 0
    total = 0
    correct_bg = 0
    correct = 0
    for i in range(len(gt_content)):
        if gt_content[i] not in bg_class:
            total_bg += 1
            if gt_content[i] == recog_content[i]:
                correct_bg += 1
        total += 1
        if gt_content[i] == recog_content[i]:
            correct += 1

    return fig, 100*float(correct)/total, 100*float(correct_bg)/total_bg

def plot_data_new(vid, ann, dataset_mapping, color_names, thres):
    vid = vid.split('/')[-1]

    gt_file = gt_path + vid.split('/')[-1]
    gt_content = read_file(gt_file).split('\n')[0:-1]
    gt_content_ann = []
    for i in range(len(gt_content)):
        gt_content_ann.append(ann[gt_content[i]])

    recog_file = pred_path+"thres_"+str(thres)+"/" + vid.split('.')[0]
    recog_content = read_file(recog_file).split('\n')[1:-1]
    recog_content_ann = []
    for i in range(len(recog_content)):
        recog_content_ann.append(ann[recog_content[i]])

    data = {
        "target": gt_content_ann,
        "prediction": recog_content_ann,
    }

    st.subheader("Segmentation")
    segmentation_chart = create_segmentation_chart(data=data, mapping=dataset_mapping, action_names=ann.keys(), color_names=color_names)
    st.altair_chart(segmentation_chart)

    total_bg = 0
    total = 0
    correct_bg = 0
    correct = 0
    for i in range(len(gt_content)):
        if gt_content[i] not in bg_class:
            total_bg += 1
            if gt_content[i] == recog_content[i]:
                correct_bg += 1
        total += 1
        if gt_content[i] == recog_content[i]:
            correct += 1

    return 100*float(correct)/total, 100*float(correct_bg)/total_bg

if pred_type == "entropy":
    thres = st.sidebar.slider(
        "Choose a threshold (% of Log(# of Classes))", step=5)
else:
    thres = st.sidebar.slider("Choose a threshold (% sigmoid output)", step=5)
if thres == 100:
    thres = 99
thres = (thres//5) + 1


vid = st.selectbox("Select File for Visualization", list_of_videos)

color_names = [name for name in mcd.CSS4_COLORS]
color_names.remove('black')
random.seed(786)
random.shuffle(color_names)
color_names.insert(0, 'black')

mof, mof_bg = plot_data_new(vid, ann, dataset_mapping, color_names, thres)



st.text("Video Level Performance : [MoF : "+"{:.2f}".format(
    mof) + "] [MoF-Bacground : "+"{:.2f}".format(mof_bg)+"]")

# st.pyplot(fig)
