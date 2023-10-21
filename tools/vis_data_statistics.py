import pdb

import numpy as np
import matplotlib.pyplot as plt
import cv2

#plt.rc('font',family='Times New Roman')

def vis_statistics(sta_path, save_name):
    stat = np.load(sta_path)
    stat = np.sum(stat, axis = 1)
    stat = cv2.rotate(stat, cv2.ROTATE_90_COUNTERCLOCKWISE)

    fig, ax=plt.subplots()

    plt.xlabel("X axis", fontsize=12)
    plt.ylabel("Z axis", fontsize=12)
    
    ax.set_xticks(np.arange(0, 121, 20))
    ax.set_yticks(np.arange(100, -1, -20))
    ax.set_xticklabels([str(ele) for ele in np.arange(-60, 65, 20)], fontsize=12)
    ax.set_yticklabels([str(ele) for ele in np.arange(0, 105, 20)], fontsize=12)
    
    plt.imshow(stat, cmap = "afmhot")
    plt.savefig(save_name, bbox_inches='tight')

if __name__ == '__main__':
    vis_statistics(sta_path = 'ARKitScenes_statistics.npy', save_name = 'ARKitScenes_statistics.png')
