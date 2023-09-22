import pdb

import numpy as np
import matplotlib.pyplot as plt
import cv2

#plt.rc('font',family='Times New Roman')

def vis_statistics(indoor_path, outdoor_path):
    indoor_stat = np.load(indoor_path)
    indoor_stat = np.sum(indoor_stat, axis = 1)
    indoor_stat = cv2.rotate(indoor_stat, cv2.ROTATE_90_COUNTERCLOCKWISE)

    outdoor_stat = np.load(outdoor_path)
    outdoor_stat = np.sum(outdoor_stat, axis = 1)
    outdoor_stat = cv2.rotate(outdoor_stat, cv2.ROTATE_90_COUNTERCLOCKWISE)

    fig, ax=plt.subplots()

    plt.xlabel("X axis", fontsize=12)
    plt.ylabel("Z axis", fontsize=12)
    
    ax.set_xticks(np.arange(0, 121, 20))
    ax.set_yticks(np.arange(100, -1, -20))
    ax.set_xticklabels([str(ele) for ele in np.arange(-60, 65, 20)], fontsize=12)
    ax.set_yticklabels([str(ele) for ele in np.arange(0, 105, 20)], fontsize=12)
    
    plt.imshow(indoor_stat, cmap = "afmhot")
    plt.savefig('indoor_statistics.png', bbox_inches='tight')

    plt.imshow(outdoor_stat, cmap = "afmhot")
    plt.savefig('outdoor_statistics.png', bbox_inches='tight')

if __name__ == '__main__':
    vis_statistics(indoor_path = 'output/data_vis/omni3d_in_statistics.npy', outdoor_path = 'output/data_vis/omni3d_out_statistics.npy')
