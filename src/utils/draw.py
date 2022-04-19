import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skeletalModel import getSkeletalModelStructure, getMTCSkeletalModelStructure
import numpy as np
import os
from os.path import *
from helpers import make_dir, embed_text
from glob import glob

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier


# tj : headless mode
# https://stackoverflow.com/questions/15713279/calling-pylab-savefig-without-display-in-ipython

save_fig = True

def drawJoints(Yx, Yy, Yz, foldName=None, type=None, root_dir=None, mask=None, valid=None):
    # plt.ion()
    #plt.show()
    fig = plt.figure(figsize=(6.4, 6.4))

    save_folder = ''
    ax = fig.add_subplot(111,projection='3d')
    if type == 'openpose':
        ax.view_init(-90, -90) # tj 90, 90 front
        save_folder = join(root_dir, foldName)
        make_dir(save_folder)
        skeleton = getSkeletalModelStructure()
    else:
        raise Exception("Nothing else supported yet!")

    T, n = Yx.shape

    skeleton = np.array(skeleton)

    number = skeleton.shape[0]
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, number)]
    plt.axis('off')


    image_idx = 0
    for i in range(T):
        if valid is not None:
            if valid[i] == 0:
                continue
        # for j in range(n):
        #      #print(Yx[i,j], Yy[i,j], Yz[i,j])
        #      ax.scatter(Yx[i,j], Yy[i,j], Yz[i,j], c = 'b', marker = 'o')
        # text(x, y, s, fontsize=12)
        if mask is not None:
            if mask[i].any(): # tj : change the background color when the frame is masked
                ax.text2D(0.5, 0.9, "▨", fontsize=24, transform=ax.transAxes)
                # fig.patch.set_facecolor('black') # tj : outer background
                # ax.set_facecolor("black") # tj : inner background
            # else: # tj : otherwise, change it back
            #     fig.patch.set_facecolor('white') # tj : outer background
            #     ax.set_facecolor("white") # tj : inner background


        for j in range(number):
            if Yx[i,skeleton[j,0]] == 0 and Yy[i,skeleton[j,0]] == 0 or Yx[i,skeleton[j,1]] == 0 and Yy[i,skeleton[j,1]] == 0:
                pass
            else:
                ax.plot([Yx[i,skeleton[j,0]], Yx[i,skeleton[j,1]]], [Yy[i,skeleton[j,0]], Yy[i,skeleton[j,1]]], [0,0], color=colors[j])


        #plt.draw()
        if save_fig:
            filename = join(save_folder, "skeleton_3d_frame%d.png" % image_idx)
            image_idx += 1
            # https://stackoverflow.com/questions/4804005/matplotlib-figure-facecolor-background-color
            plt.savefig(filename, dpi=100, facecolor=fig.get_facecolor(), edgecolor='none')
        # plt.show()
        # plt.pause(0.3)
        for j in range(number):
            if Yx[i,skeleton[j,0]] == 0 and Yy[i,skeleton[j,0]] == 0 or Yx[i,skeleton[j,1]] == 0 and Yy[i,skeleton[j,1]] == 0:
                pass
            else:
                ax.lines.pop(0)

        if mask is not None:
            if mask[i].any(): # tj : change the background color when the frame is masked
                ax.texts.pop(0)

        # plt.clf()
    plt.close()
    video_name = join(save_folder, foldName + '.mp4')
    gen_video_command = 'ffmpeg -y -i ' + save_folder + '/skeleton_3d_frame%d.png -c:v libx264 ' + video_name
    print(gen_video_command)
    os.system(gen_video_command)

def draw2DJoints( Yx, Yy, foldName, root_dir, name):
    # plt.ion()
    #plt.show()
    fig = plt.figure(figsize=(6.4, 6.4))

    ax = fig.add_subplot(111,projection='3d')
    ax.view_init(-90, -90)
    save_folder = join(root_dir, foldName)
    skeleton = getSkeletalModelStructure()


    make_dir(save_folder)
    skeleton = np.array(skeleton)

    number = skeleton.shape[0]
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, number)]
    plt.axis('off')

    num_people = Yx.shape[0]

    for i in range(1):  # tj : manually choose the first person
        # for j in range(n):
        #      #print(Yx[i,j], Yy[i,j], Yz[i,j])
        #      ax.scatter(Yx[i,j], Yy[i,j], Yz[i,j], c = 'b', marker = 'o')

        for j in range(number):
            # print("%d - %d"%(skeleton[j,0], skeleton[j,1]))
            # print("%f, %f - %f, %f"%(Yx[i,skeleton[j,0]], Yx[i,skeleton[j,1]], Yy[i,skeleton[j,0]], Yy[i,skeleton[j,1]]))
            if Yx[i,skeleton[j,0]] == 0 and Yy[i,skeleton[j,0]] == 0 or Yx[i,skeleton[j,1]] == 0 and Yy[i,skeleton[j,1]] == 0:
                pass
            else:
                ax.plot([Yx[i,skeleton[j,0]], Yx[i,skeleton[j,1]]], [Yy[i,skeleton[j,0]], Yy[i,skeleton[j,1]]], [0,0], color=colors[j])
        #plt.draw()
    if save_fig:
        plt.savefig(join(save_folder, name), dpi=100)
        #plt.show()
        #plt.pause(0.3)
        # for j in range(number):
        #     ax.lines.pop(0)
        #plt.clf()
    plt.close()


def drawPlain( foldName, root_dir, name):
    # plt.ion()
    #plt.show()
    fig = plt.figure(figsize=(6.4, 6.4))

    ax = fig.add_subplot(111,projection='3d')
    save_folder = join(root_dir, foldName)
    make_dir(save_folder)
    plt.axis('off')
    if save_fig:
        plt.savefig(join(save_folder, name), dpi=100)
        #plt.show()
        #plt.pause(0.3)
        # for j in range(number):
        #     ax.lines.pop(0)
        #plt.clf()
    plt.close()

def show_ROC(fpr_thresholds, tpr_thresholds):
    roc_auc = auc(fpr_thresholds, tpr_thresholds)
    plt.figure()
    lw = 2
    plt.plot(fpr_thresholds, tpr_thresholds, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic - {}'.format(idx))
    plt.legend(loc="lower right")
    plt.show()


def draw_ROC(fpr_thresholds, tpr_thresholds, output_dir, idx):
    make_dir(output_dir)
    roc_auc = auc(fpr_thresholds, tpr_thresholds)
    plt.figure()
    lw = 2
    plt.plot(fpr_thresholds, tpr_thresholds, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic - {}'.format(idx))
    plt.legend(loc="lower right")
    #plt.show()

    filename = join(output_dir, "%d.png" % idx)
    plt.savefig(filename, dpi=100, edgecolor='none')
    plt.close()


def draw(root_dir:str, headless:bool=False ):

    if headless:
        matplotlib.use('Agg')

    valid = np.loadtxt(join(root_dir, "valid.txt"))

    origin = np.loadtxt(join(root_dir, "origin.txt"))
    origin_x = origin[0:origin.shape[0], 0:origin.shape[1]:3]
    origin_y = origin[0:origin.shape[0], 1:origin.shape[1]:3]
    origin_z = origin[0:origin.shape[0], 2:origin.shape[1]:3]
    drawJoints(origin_x, origin_y, origin_z, foldName="origin", type="openpose", root_dir=root_dir, mask=None, valid=valid)

    mask = np.loadtxt(join(root_dir, "mask.txt"))


    predict = np.loadtxt(join(root_dir, "predict.txt"))
    predict_x = predict[0:predict.shape[0], 0:predict.shape[1]:3]
    predict_y = predict[0:predict.shape[0], 1:predict.shape[1]:3]
    predict_z = predict[0:predict.shape[0], 2:predict.shape[1]:3]
    drawJoints(predict_x, predict_y, predict_z, foldName="predict", type="openpose", root_dir=root_dir, mask=mask, valid=valid)

    masked_src = np.loadtxt(join(root_dir, "masked_src.txt"))
    masked_src_x = masked_src[0:masked_src.shape[0], 0:masked_src.shape[1]:3]
    masked_src_y = masked_src[0:masked_src.shape[0], 1:masked_src.shape[1]:3]
    masked_src_z = masked_src[0:masked_src.shape[0], 2:masked_src.shape[1]:3]
    drawJoints(masked_src_x, masked_src_y, masked_src_z, foldName="masked_src", type="openpose", root_dir=root_dir, mask=None, valid=valid)


    interpolated = join(root_dir, "interpolated.txt")
    if os.path.exists(interpolated):
        interpolated = np.loadtxt(join(root_dir, "interpolated.txt"))
        interpolated_x = interpolated[0:interpolated.shape[0], 0:interpolated.shape[1]:3]
        interpolated_y = interpolated[0:interpolated.shape[0], 1:interpolated.shape[1]:3]
        interpolated_z = interpolated[0:interpolated.shape[0], 2:interpolated.shape[1]:3]
        drawJoints(interpolated_x, interpolated_y, interpolated_z, foldName="interpolated", type="openpose", root_dir=root_dir, mask=mask, valid=valid)





def merge(left_dir:str, right_dir:str):
    left_video_name = join(left_dir, 'left.mp4')
    left_video_name_text = join(left_dir, 'left_text.mp4')
    gen_video_command = 'ffmpeg -y -i ' + left_dir + '/%08d.png -c:v libx264 ' + left_video_name
    print(gen_video_command)
    os.system(gen_video_command)

    right_video_name = join(right_dir, 'right.mp4')
    right_video_name_text = join(right_dir, 'right_text.mp4')
    gen_video_command = 'ffmpeg -y -i ' + right_dir + '/skeleton_3d_frame%d.png -c:v libx264 ' + right_video_name
    print(gen_video_command)
    os.system(gen_video_command)

    embed_text(left_video_name, left_video_name_text, 'Czech')
    embed_text(right_video_name, right_video_name_text, 'Skeletor')

    merged_video_name = join(left_dir, 'merged.mp4')

    merge_command = 'ffmpeg -y -i ' + left_video_name_text + ' -i ' + right_video_name_text + \
                    ' -filter_complex "hstack,format=yuv420p" -c:v libx264  -crf 18 -preset veryfast ' + merged_video_name
    #print(merge_command)
    os.system(merge_command)


def _draw_curve(root_dir: str, cls_outputs, class_id_list, unit_frames=128):


    # cls = np.loadtxt(join(root_dir, "cls.txt"), dtype=int)
    # prob = np.loadtxt(join(root_dir, "predict.txt"))
    # predict = np.argmax(prob, axis=1)

    import pylab
    NUM_COLORS = 52
    cm = pylab.get_cmap('gist_rainbow')
    colorList = [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]




    save_folder = os.path.join(root_dir, "curve")
    make_dir(save_folder)

    # plt.ion()
    # plt.show()
    fig = plt.figure(figsize=(19.2, 6.4))
    ax = fig.add_subplot(111)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.8)
    axes = plt.gca()
    axes.set_xlim([0, unit_frames])
    axes.set_ylim([0, 1])

    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=20)

    for i in range(cls_outputs.shape[0]):

        index = i

        unit_index = int(i / unit_frames)
        unit_offset = i % unit_frames
        unit_start = unit_index * unit_frames
        if(unit_offset == 0):
            plt.clf()
            axes = plt.gca()
            axes.set_xlim([index, index + unit_frames])
            axes.set_ylim([0, 1])
            ax.tick_params(axis='both', which='major', labelsize=20)
            ax.tick_params(axis='both', which='minor', labelsize=20)
        for class_id in class_id_list:
            color = colorList[class_id]
            cls_output = cls_outputs[:, class_id]
            x_join = range(0,i+1)
            y_join = cls_output[0:i+1]
            plt.plot(x_join, y_join, c=color, linewidth=3)
        plt.draw()

        filename = join(save_folder, "%d.png" % i)
        # https://stackoverflow.com/questions/4804005/matplotlib-figure-facecolor-background-color
        plt.savefig(filename, dpi=100, facecolor=fig.get_facecolor(), edgecolor='none')
        # silence_txt.remove()

        # plt.clf()
    plt.close(fig)
    video_name = join(save_folder, 'curve.mp4')
    gen_video_command = 'ffmpeg -y -i ' + save_folder + '/%d.png -c:v libx264 ' + video_name
    # print(gen_video_command)
    os.system(gen_video_command)
    return video_name



def _draw_groundtruth(root_dir: str, cls_gt, unit_frames = 128):
    import pylab
    NUM_COLORS = 52
    cm = pylab.get_cmap('gist_rainbow')
    colorList = [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]

    save_folder = os.path.join(root_dir, "groundtruth")
    make_dir(save_folder)

    # plt.ion()
    # plt.show()
    fig = plt.figure(figsize=(19.2, 6.4))
    ax = fig.add_subplot(111)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.8)

    x_join = []
    y_join = []
    c_join = []
    for i in range(len(cls_gt)):

        index = i

        unit_index = int(i / unit_frames)
        unit_offset = i % unit_frames
        unit_start = unit_index * unit_frames
        if (unit_offset == 0):
            plt.clf()
            axes = plt.gca()
            axes.set_xlim([index, index + unit_frames])
            axes.set_ylim([0, 1])
            ax.tick_params(axis='both', which='major', labelsize=20)
            ax.tick_params(axis='both', which='minor', labelsize=20)
            x_join.clear()
            y_join.clear()

        x = index
        x_join.append(x)
        y = 0.5
        y_join.append(y)
        c = colorList[cls_gt[x]]
        c_join.append(c)

        plt.scatter(x_join, y_join, s=500, c=c_join, marker='s')

        if x == 0 or cls_gt[x] != cls_gt[x-1]:
            matplotlib.pyplot.text((float(x) / unit_frames), 0.3,  str(cls_gt[x]), fontsize=24, transform=ax.transAxes)

        # plt.plot(x_join, y_join, 'b-', linewidth=6)

        # plt.plot(x, y, color=c)
        plt.draw()


        filename = join(save_folder, "%d.png" % i)
        # https://stackoverflow.com/questions/4804005/matplotlib-figure-facecolor-background-color
        plt.savefig(filename, dpi=100, facecolor=fig.get_facecolor(), edgecolor='none')

    plt.close(fig)
    video_name = join(save_folder, 'groundtruth.mp4')
    gen_video_command = 'ffmpeg -y -i ' + save_folder + '/%d.png -c:v libx264 ' + video_name
    # print(gen_video_command)
    os.system(gen_video_command)
    return video_name





if __name__ == "__main__":


    # = tj : merge video
    merge('/vol/research/SignPose/tj/Workspace/Pheonix_Czech/results/dev/01April_2010_Thursday_heute-6698/images',
          '/home/seamanj/Workspace/BERT_skeleton/results/e_10_m_1000_lr_1e-3_w_32/eval/dev/01April_2010_Thursday_heute-6698/predict')




    # = tj : draw specified dir
    # dir = '/home/seamanj/Workspace/BERT_skeleton/results/e_1_m_100_lr_1e-3_t_2000_w_16/dev/57'
    # draw(root_dir=dir)

    # = tj : down-most dirs in one dir
    # dir_to_draw = '/home/seamanj/Workspace/BERT_skeleton/results/'
    # for dirpath, dirnames, filenames in os.walk(dir_to_draw):
    #     if not dirnames:
    #         print(dirpath)


    # = tj : specified deep-level dirs
    # dir_to_draw = '/home/seamanj/Workspace/BERT_skeleton/results/e_10_m_1000_lr_1e-3_w_32/*/*/'
    # for dir in glob(dir_to_draw):
    #     print(dir)
    #     draw(root_dir=dir)




    # origin = np.loadtxt("/home/seamanj/Workspace/BERT_skeleton/results/origin.txt")
    # origin_x = origin[0:origin.shape[0], 0:origin.shape[1]:3]
    # origin_y = origin[0:origin.shape[0], 1:origin.shape[1]:3]
    # origin_z = origin[0:origin.shape[0], 2:origin.shape[1]:3]
    # drawJoints(origin_x, origin_y, origin_z, foldName="origin", type="openpose")
    #
    #
    # predict = np.loadtxt("/home/seamanj/Workspace/BERT_skeleton/results/predict.txt")
    # predict_x = predict[0:predict.shape[0], 0:predict.shape[1]:3]
    # predict_y = predict[0:predict.shape[0], 1:predict.shape[1]:3]
    # predict_z = predict[0:predict.shape[0], 2:predict.shape[1]:3]
    # drawJoints(predict_x, predict_y, predict_z, foldName="predict", type="openpose")




