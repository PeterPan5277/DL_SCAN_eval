import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib as mpl
import scipy.io as sio
import torch.nn as nn
from sys import getsizeof
def hist_3percent(pred, target, save_dir, top_percent, a, g, lossf):
    #print(np.array(pred).shape) #1 79929 20 20
    #print(np.array(target).shape) #79929 1 20 20
    pred= np.array(pred.reshape(-1))
    target = np.array(target.reshape(-1))
    #print(pred.shape) #torch.tensor 31971600 
    #print(target.shape)#torch.tensor  31971600
    true_idx=np.where(target==1)[0]
    false_idx= np.where(target==0)[0]
    pred_f = np.array(pred[false_idx])#14280
    pred_t = np.array(pred[true_idx])
    print('max of predT is', np.max(pred_t), 'and min of predT', np.min(pred_t))
    print('max of predf is', np.max(pred_f), 'and min of predf', np.min(pred_f))
    # plotting histogram and density
    plt.figure()
    sns.set(style = "ticks") # 白色網格背景
    sns.histplot(data=pred_t, log_scale=False, kde=True,bins=100, label="pred_value_of_+case", color='r')#, ax=ax2)
    plt.title('Predicting value of positive cases')
    #plt.yaxis.set_label_position("right")
    best=[]
    for i in range(len(top_percent)):
        color = ['k','b','r','w','g']
        percent_3_idx = round(len(pred_t)*top_percent[i])
        percent_3 = sorted(pred_t)[-percent_3_idx]
        best.append(percent_3)
        print(f'Top {top_percent[i]*100}% threshold of +:', percent_3)
        plt.axvline(x=percent_3, c=color[i], ls='--', lw=1, label=f'Top{top_percent[i]*100}%_threshold')
        plt.legend(loc=2)
    plt.tight_layout()
    plt.savefig(save_dir+'HIST_3%/'+f'{lossf}_{a}_{g}_(+)plot.png', dpi=300)      
    plt.close()
    #畫負的
    plt.figure()
    sns.histplot(data=pred_f, log_scale=False, kde=True,bins=10000, label="pred_value_of_-case", color='b')#, ax=ax2)
    plt.title('Predicting value of positive cases')
    for i in range(len(top_percent)):
        color = ['k','b','r','w','g']
        percent_3_idx_F = round(len(pred_f)*top_percent[i])
        percent_3_F = sorted(pred_f)[-percent_3_idx_F]
        print(f'Top {top_percent[i]*100}% threshold of -:', percent_3_F)
        plt.axvline(x=percent_3_F, c=color[i], ls='--', lw=1, label=f'Top{top_percent[i]*100}%_threshold')
        plt.legend(loc=1)
        #plt.yaxis.set_label_position("right")
    plt.tight_layout()
    plt.xlim(0, 0.15)
    plt.savefig(save_dir+'HIST_3%/'+f'{lossf}_{a}_{g}_(-)plot.png', dpi=300)
    plt.close()
    return(best)

def threshold_points(pred, target, save_dir, top_percent):
    pred = np.array(pred[0,...])
    #print(np.max(pred)) 0.9965
    target = np.array(target[:,0,...])
    #print(np.array(pred).shape) #79929 20 20
    #print(np.array(target).shape) #79929 20 20
    idx = np.where(target[:,1,8]==1)
    #print(np.array(idx[0]))
    #print(sorted(pred[idx,1,8][0]))
    point_threshold = np.zeros((20, 20), dtype=np.float32) #每個點的threshold
    N_ofpositive= np.zeros((20, 20), dtype=np.float32) #每個網格出現幾個1
    for i in range(20):
        for j in range(20):
            true_idx=np.where(target[:,i,j]==1)
            N_ofpositive[i,j] = len(true_idx[0])
            top_value = sorted(pred[true_idx,i,j][0])
            idx = (-round(len(top_value)*top_percent))
            point_threshold[i,j] = top_value[idx]
    # print(N_ofpositive)
    print('+個數最少', np.min(N_ofpositive))
    print('+個數最多', np.max(N_ofpositive))
    # print(point_threshold)
    print('threshold_min', np.min(point_threshold))
    print('threshold_max', np.max(point_threshold))
    # idx = np.where(N_ofpositive==347.0)
    # print(idx)
    # print(point_threshold[idx])
    return(point_threshold, N_ofpositive)

def burst(tar, pred, save_dir, g, a, lossf):
    print(tar.shape)
    print(pred.shape)
if __name__ == '__main__':
    print('Test codes')
