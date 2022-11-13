#to plot the output files
from tkinter import E
import conda
import os, sys
from cv2 import DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, threshold

from matplotlib import markers
conda_file_dir = conda.__file__
conda_dir = conda_file_dir.split('lib')[0]
proj_lib = os.path.join(os.path.join(conda_dir, 'share'), 'proj')
os.environ["PROJ_LIB"] = proj_lib
sys.path.append("/wk171/peterpan/SCAN/SCAN_eval/")
from os import listdir
from datetime import datetime, timedelta
import scipy.io as sio
import pickle
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits
from mpl_toolkits.basemap import Basemap
import matplotlib as mpl
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn
import torch
from PD_diagram import PFD_2D
from plot_code import hist_3percent, threshold_points, burst
class Plot_scan(nn.Module):
    def __init__(self,root, plot_save_dir, data_dir, time_dir):
        self.root = root
        self.save_dir = plot_save_dir
        #time first
        with open(self.root+time_dir,'rb') as f1:
            time_dic = pickle.load(f1)
            self.time = time_dic['initial']
            #print(self.time)
            
        #then the scan out
        with open(self.root+data_dir,'rb') as f2:
            data_dic = pickle.load(f2)
            self.pred= data_dic['out'] #1 5469 20 20 (seq/batch/x/y)
            self.inp = data_dic['inp'] #5469 6 2 120 120 (batch/seq/radar/x/y)
            self.target = data_dic['target'] #5469 1 120 120 (batch/seq/x/y)
        self.pred=nn.Sigmoid()(self.pred) #nn.sigmoid是一個class
        self.baseline = self.inp[:,-1,0,:,:]
        #print('baseline_shape', self.baseline.shape) #79929,120,120
        self.baseline = nn.MaxPool2d(6)(self.baseline)
        #print('baseline_shape', self.baseline.shape) #79929,20,20
        #print(np.array(self.inp).shape) #79929,6,2,120,120
        #print(np.array(self.pred).shape) #1 79929 20 20
        #print(np.array(self.target).shape) #79929 1 20 20
        #self.true_idx=np.where(self.target.reshape(-1)==1)[0]
        #self.false_idx= np.where(self.target.reshape(-1)==0)[0]
        #print(len(self.true_idx),len(self.false_idx)) #129287v.s.78644313 #0.16%....
    def stats(self):
        scan_location = np.where(np.array(self.inp[:,0,0,...])==1)
        total_radar = np.array(self.inp[:,0,1,...][scan_location])*35
        print('max',np.max(sorted(total_radar)))
        print('min',np.min(sorted(total_radar)))
        print('mean',np.mean(sorted(total_radar)))
        print('total#', len(total_radar))
        print('min0-100', sorted(total_radar)[:100])
        #sns.boxplot(sorted(total_radar), showfliers=True)
        sns.histplot(data=total_radar, log_scale=False, kde=True,bins=100, label="dBZ-SCAN")
        plt.savefig(self.save_dir+'/boxplot/'+'boxplot'+'.png', dpi=300)
        


    def plot_CV(self, case_idx, best, case_seq):
        grid = 120
        lon=np.linspace(120.68105,122.17745,grid) # resolution=1.3 km
        lat=np.linspace(24.0675,25.56595,grid)
        lon2d, lat2d = np.meshgrid(lon, lat)
        m = Basemap(projection='cyl', llcrnrlat=24.0675, urcrnrlat=25.56595,\
                llcrnrlon=120.68, urcrnrlon=122.17, resolution='l', lat_ts=20) 
        x, y = m(lon2d, lat2d)
        #RADAR CV color bar
        bounds_cv = [0,1,3,5,7,10,13,15,20,25,30,35,40,45,50,55,60,65]
        cmap_cv = mpl.colors.ListedColormap(['#FFFFFF', '#9CFCFF', '#03C8FF', '#059BFF', '#0363FF',
                                   '#059902', '#39FF03', '#FFFB03', '#FFC800', '#FF9500',
                                   '#FF0000', '#CC0000', '#990000', '#960099', '#C900CC',
                                   '#FB00FF', '#FDC9FF'])
        norm_cv = mpl.colors.BoundaryNorm(bounds_cv, cmap_cv.N)    
        #Scan
        cmap = mpl.colors.ListedColormap(['w','r'])
        bounds = [-0.1,0.5,1.1]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        time_step=1
        for i in range(0, 6*(case_seq-1)+1 ,6):
            ax1=plt.subplot(2,case_seq,time_step)
            #m.fillcontinents()
            m.drawcoastlines()
            m.drawparallels(np.arange(24,26,0.25),labels=[1,1,0,1], fontsize=8)
            m.drawmeridians(np.arange(121,122,0.25),labels=[1,1,0,1], rotation=45, fontsize=8)    
            m.pcolormesh(x, y, self.inp[case_idx+i,0,1,...]*35, zorder=0, edgecolors='none',
                                    shading='auto',norm=norm_cv, cmap=cmap_cv)
            #m.colorbar(location='right',pad=0.5)
      
            
            ax1.set_title(self.time[case_idx+i],y=1.05,fontsize=15)
            ax1.set_ylabel('RadarCV',rotation=0,fontsize=10) #設定每個小圖的y
            ax1.yaxis.set_label_coords(-0.3,0.8) #設定ylabel位置
            #
            ax2=plt.subplot(2,case_seq,time_step+case_seq)
            #m.fillcontinents()
            m.drawcoastlines()
            m.drawparallels(np.arange(24,26,0.25),labels=[1,1,0,1], fontsize=8)
            m.drawmeridians(np.arange(121,122,0.25),labels=[1,1,0,1], rotation=45, fontsize=8)       
            m.pcolormesh(x, y, self.inp[case_idx+i,0,0,...],norm=norm,cmap=cmap ,zorder=0)
            #m.colorbar(location='right',pad=0.4)
            ax2.set_ylabel('Scan',rotation=0,fontsize=10)
            ax2.yaxis.set_label_coords(-0.3,0.8)
            time_step+=1

            plt.tight_layout()
            plt.savefig(self.save_dir+f'/CV/'+'Focal_'+str(self.time[case_idx])+'.png', dpi=300)
    def plot_space_correlation(self, threshold_ofeachpoint, Nofcase):
        #RADAR CV color bar
        bounds1 = [0.15,0.2,0.22,0.25,0.27,0.3,0.33,0.37,0.4]
        #cmap1 = mpl.colors.ListedColormap(['#FFFFFF', '#9CFCFF', '#03C8FF', '#059BFF', '#0363FF',
        #                           '#059902', '#39FF03', '#FFFB03', '#FFC800', '#FF9500',
        #                           '#FF0000', '#CC0000', '#990000', '#960099', '#C900CC',
        #                           '#FB00FF', '#FDC9FF'])

        norm1 = mpl.colors.BoundaryNorm(bounds1, ncolors=256)   
        bounds2 = [400,500,800,1000,1200,1500,1800,2000]
        cmap2 = mpl.colors.ListedColormap(['white','silver','green','blue','orange','red','purple'])
        norm2 = mpl.colors.BoundaryNorm(bounds2, ncolors=256)
        mat = sio.loadmat('city_lonlat_region5.mat')
        citylon = mat['citylon']
        citylat = mat['citylat']
        del mat
        grid = 20
        lon=np.linspace(120.68105,122.17745,grid) # resolution=1.3 km
        lat=np.linspace(24.0675,25.56595,grid)
        lon2d, lat2d = np.meshgrid(lon, lat)
        fig, ax = plt.subplots(2,1, figsize=(10, 8.5), dpi=200, facecolor='w', squeeze=False)#最後這個防止(原本想要回傳2,1結果回傳大小只有2,)
        for time_step in range(1):
            for y in range(2):   #y等於幾個inps, 目前試thresholds each point and N_of_case 
                if y ==0:
                    ax[y,time_step].set_title('Thresholds of each point',y=1.05,fontsize=15)
                else:
                    ax[y,time_step].set_title('Cases of each point',y=1.05,fontsize=15)
                if y ==0:
                    im=ax[y,time_step].pcolormesh(lon2d, lat2d, threshold_ofeachpoint, edgecolors='none',
                                        shading='auto',norm=norm1, cmap='YlOrRd')
                    #im=ax[y,time_step].imshow(threshold_ofeachpoint, cmap="YlOrRd", alpha=1)
                    #im = sns.heatmap(threshold_ofeachpoint, ax = ax[y,time_step], alpha=1, cmap="YlOrRd", zorder=1)
                    #ax[y,time_step].invert_yaxis() #zorder越大代表越上層
                    cbar = fig.colorbar(im, ax=ax[y,time_step], orientation='vertical')
                else:
                    im=ax[y,time_step].pcolormesh(lon2d, lat2d, Nofcase, norm=norm2, cmap='YlGnBu' ,zorder=0)
                    #im=ax[y,time_step].imshow(Nofcase, cmap="YlGnBu", alpha=1)
                    #im =  sns.heatmap(Nofcase, ax = ax[y,time_step], alpha=1, cmap="YlGnBu", zorder=1)
                    #ax[y,time_step].invert_yaxis()
                    cbar = fig.colorbar(im, ax=ax[y,time_step], orientation='vertical')
                ax[y,time_step].plot(citylon,citylat,'k',linewidth=0.6,alpha=0.4)
                ax[y,time_step].axis([120.6875, 122.1875, 24.0625, 25.5625])# whole area [119, 123, 21, 26]
                ax[y,time_step].set_aspect('equal')
                #ax[y,time_step].set_xticks([])
                #ax[y,time_step].set_yticks([])

        plt.tight_layout()
        plt.savefig(self.save_dir+f'/space_relation/'+f'{lossf}_{a}_{g}_space_correlation_imshow'+'.png', dpi=300)
        plt.close()

    def plot_new(self, case_idx, best, case_seq):
        #RADAR CV color bar
        bounds_cv = [0,1,3,5,10,15,20,25,30,35,40,45,50,55,60,65]
        cmap_cv = mpl.colors.ListedColormap(['#FFFFFF', '#9CFCFF', '#03C8FF', '#059BFF', '#0363FF',
                                   '#059902', '#39FF03', '#FFFB03', '#FFC800', '#FF9500',
                                   '#FF0000', '#CC0000', '#990000', '#960099', '#C900CC',
                                   '#FB00FF', '#FDC9FF'])
        norm_cv = mpl.colors.BoundaryNorm(bounds_cv, cmap_cv.N)    
        #Scan
        cmap = mpl.colors.ListedColormap(['w','r'])
        bounds = [-0.1,0.5,1.1]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        
        mat = sio.loadmat('city_lonlat_region5.mat')
        citylon = mat['citylon']
        citylat = mat['citylat']
        del mat
        grid = 120
        lon=np.linspace(120.68105,122.17745,grid) # resolution=1.3 km
        lat=np.linspace(24.0675,25.56595,grid)
        lon2d, lat2d = np.meshgrid(lon, lat)
        fig, ax = plt.subplots(2, case_seq, figsize=(10, 8.5), dpi=200, facecolor='w')
        for time_step in range(case_seq):
            for y in range(2):   #y等於幾個inps, 假如有SCAN adn radar y=2
                if y ==0:
                    ax[y,time_step].set_title('Inp time:\n'+str(self.time[case_idx+time_step]),y=1.05,fontsize=15)
                ax[y,time_step].plot(citylon,citylat,'k',linewidth=0.6)
                ax[y,time_step].axis([120.6875, 122.1875, 24.0625, 25.5625])# whole area [119, 123, 21, 26]
                ax[y,time_step].set_aspect('equal')
            
                if y ==0:
                    im=ax[y,time_step].pcolormesh(lon2d, lat2d, self.inp[case_idx+time_step,0,1,...]*35, edgecolors='none',
                                        shading='auto', norm=norm_cv, cmap=cmap_cv)
                    #if time_step == case_seq-1:
                    #    cbar = fig.colorbar(im, ax=ax[y,time_step], orientation='vertical')
                else:
                    im=ax[y,time_step].pcolormesh(lon2d, lat2d, (self.inp[case_idx+time_step,0,0,...]),norm=norm,cmap=cmap ,zorder=0)
                    #if time_step == case_seq-1:
                    #    cbar = fig.colorbar(im, ax=ax[y,time_step], orientation='vertical')
                ax[y,time_step].set_xticks([])
                ax[y,time_step].set_yticks([])
        plt.tight_layout()
        plt.savefig(self.save_dir+f'/CV_new/'+f'Focal_{inp_d}'+str(self.time[case_idx])+'.png', dpi=300)
        plt.close()
    def plot_pcolor(self, case_idx, best, case_seq, top, grid_thresh):
        if mp=='YES':
            grid = 20
        else:
            grid = 120
        mat = sio.loadmat('city_lonlat_region5.mat')
        citylon = mat['citylon']
        citylat = mat['citylat']
        del mat
        lon=np.linspace(120.68105,122.17745,grid) # resolution=1.3 km
        lat=np.linspace(24.0675,25.56595,grid)
        lon2d, lat2d = np.meshgrid(lon, lat)
        m = Basemap(projection='cyl', llcrnrlat=24.0675, urcrnrlat=25.56595,\
                llcrnrlon=120.68, urcrnrlon=122.17, resolution='l', lat_ts=20) 
        x, y = m(lon2d, lat2d)
        #若沒有門檻的話
        # cmap = plt.cm.get_cmap('Reds')
        # bounds = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7]
        #自訂color
        cmap = mpl.colors.ListedColormap(['w','r'])
        bounds = [-0.1,0.5,1.1]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
  
        bounds_cv = [0,0.2,0.4,0.6,0.7,0.75,0.8,0.85,0.9,0.95]
        cmap_cv = mpl.colors.ListedColormap(['#FFFFFF', '#9CFCFF', '#03C8FF', '#059BFF', '#0363FF',
                                   '#059902', '#39FF03', '#FFFB03', '#FFC800', '#FF9500',
                                   '#FF0000', '#CC0000', '#990000', '#960099', '#C900CC',
                                   '#FB00FF', '#FDC9FF'])
        norm_cv = mpl.colors.BoundaryNorm(bounds_cv, cmap_cv.N)   

        fig=plt.figure(figsize=(10,8))
        
        time_step=1
        
        for i in range(0, 6*(case_seq-1)+1 ,6):
            pred = np.array(self.pred[0,case_idx+i,:,:])
            if grid_thresh:
                for idx in range(20):
                    for jdx in range(20):
                        #pred 1 79929 20 20
                        if pred[idx,jdx] >= best[idx,jdx]:
                            pred[idx,jdx]=1
                        else:
                            pred[idx,jdx]=0
            else:
                thresh = best[0]
                print(thresh)
                pred = np.where(pred>thresh, 1, 0)
            ax1=plt.subplot(2,case_seq,time_step)
            #m.fillcontinents()
            m.drawcoastlines()
            m.drawparallels(np.arange(20,27,1),labels=[1,1,0,1], fontsize=8)
            m.drawmeridians(np.arange(118,124,1),labels=[1,1,0,1], rotation=45, fontsize=8)    
            m.pcolormesh(x, y, (self.target[case_idx+i,0,:,:]), zorder=0, norm=norm, cmap=cmap)
            #m.colorbar(location='right',pad=0.4)
            ax1.set_title(self.time[case_idx+i],y=1.05,fontsize=15)
            ax1.set_ylabel('GT',rotation=0,fontsize=10) #設定每個小圖的y
            ax1.yaxis.set_label_coords(-0.2,0.9) #設定ylabel位置
            #
            ax2=plt.subplot(2,case_seq,time_step+case_seq)
            #m.fillcontinents()
            m.drawcoastlines()
            m.drawparallels(np.arange(20,27,1),labels=[1,1,0,1], fontsize=8)
            m.drawmeridians(np.arange(118,124,1),labels=[1,1,0,1], rotation=45, fontsize=8) 
            m.pcolormesh(x, y, pred ,norm=norm, cmap=cmap ,zorder=0) 
            #plt.colorbar()       
            ax2.set_ylabel('Pred\n'+str({inp_d}),rotation=0,fontsize=10)
            ax2.yaxis.set_label_coords(-0.2,0.9)
            time_step+=1
            plt.tight_layout()
            #plt.suptitle('Predicts 0-1 hr SCAN dots(With Threshold):',fontsize=20)
            plt.savefig(self.save_dir+'/contour/'+f'grid_{grid_thresh}_{lossf}_{top}_{inp_d}_a{a}g{g}_Maxpool_{mp}_'+str(self.time[case_idx])+'_Thresholds'+'.png', dpi=300)
        plt.close()
    def Find_Optimal_Cutoff(self, fpr,tpr,threshold):
        i = np.arange(len(tpr)) #i=[0,1,2,..240] total 241    
        roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 
                            'threshold' : pd.Series(threshold, index=i),
                            'fpr' : pd.Series(fpr,index=i),
                            'tpr' : pd.Series(tpr,index=i)}) 
        #print(roc.fpr[100])#兩個pd.series都有給定index(也就是0~240)
        roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]
 
        return list(roc_t['threshold']) 

    def plot_ROC(self, case_idx, is_case, case_seq):
        best=[]
        is_case=False
        if is_case:
            for i in range(0,(case_seq-1)*6+1,6):
                target = self.target[case_idx+i,0,:,:].reshape(-1)
                pred = self.pred[0,case_idx+i,:,:].reshape(-1)
        else:
            target = self.target[:,0,:,:].reshape(-1)
            pred = self.pred[0,:,:,:].reshape(-1)

            fpr, tpr, threshold = metrics.roc_curve(target, pred) #input is 1D-array
            Best_threshold = self.Find_Optimal_Cutoff(fpr,tpr,threshold)[0]
            best.append(Best_threshold)
            auc = round(metrics.auc(fpr, tpr),3)
            print('AUC value is ', auc)
            plt.plot(fpr, tpr, color='red', label='ROC')
            plt.plot([0, 1], [0, 1], color='green', linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve with AUC={auc}')

            print('Best threshold is', Best_threshold)
            plt.legend()
        if is_case:
            plt.savefig(self.save_dir+'ROC/'+f'Focal_{inp_d}_Maxpool_{mp}_alpha{a}g{g}_ROC'+str(self.time[case_idx+i])+'.png', dpi=300)
        else:
            plt.savefig(self.save_dir+'ROC/'+'ALL_Focal_Maxpool_{mp}_alpha{a}g{g}_ROC'+str(self.time[case_idx])+'.png', dpi=300)
        plt.close()
        return(best)
    def plot_hitogram(self, case_idx, is_case):
        if is_case:
            pred_dt= self.pred[:,case_idx,:,:].reshape(-1)
            target_dt = self.target[case_idx,:,...].reshape(-1)
            #print(pred_dt)
            #print(target_dt)
            true_idx=np.where(target_dt.reshape(-1)==1)[0]
            false_idx= np.where(target_dt.reshape(-1)==0)[0]
            pred_dt_f = np.array(pred_dt[false_idx])#14280
            pred_dt_t = np.array(pred_dt[true_idx])#120
            #print('max of predF is', np.max(pred_dt_f), 'and min of predF', np.min(pred_dt_f))
            #print('max of predT is', np.max(pred_dt_t), 'and min of predT', np.min(pred_dt_t))
            # plotting histogram and density
            sns.set(style = "ticks") # 白色網格背景
            sns.histplot(data=pred_dt_f, log_scale=False, kde=True,bins=100, label="pred_of_F")
            plt.xlim(10e-2,10e-1)
            plt.legend(loc=9) #loc=1~9有各自的位置 我這邊是利用label來設定(所以要plt.legent())
            #ax2 = plt.twinx()
            sns.histplot(data=pred_dt_t, log_scale=False, kde=True,bins=20, label="pred_of_T", color='r')#, ax=ax2)
            #plt.axvline(x=self.best, c='k', ls='--', lw=1, label='best_threshold')
            plt.legend(loc=2)
            plt.savefig(self.save_dir+'HIST/'+f'{lossf}_{inp_d}_Maxpool_{mp}_alpha{a}g{g}'+str(self.time[case_idx])+'.png', dpi=300)
            plt.close()
            #sns.displot(data=pred_dt_f)
            #sns.displot(data=pred_dt_t,x='pred_probability', hue="species", kind="kde")
            #sns.kdeplot(data=pred_dt_f, label="pred_of_F", shade=True, log_scale=True)
            #sns.kdeplot(data=pred_dt_t, label="pred_of_T", shade=True, log_scale=True)
            #劃出best那條線
            #plt.axvline(x=self.best, c='k', ls='--', lw=1, label='best_threshold')
            #plt.xlim(0,0.5)
            #plt.ylim(0,50)
            #plt.xlabel('pred_value')
            #plt.ylabel('KDE')
            #plt.title('Kernel density estimation')
            #plt.savefig(self.save_dir+'HIST/'+'Separate_KDE'+str(self.time[case_idx])+'.png', dpi=300)
       
        else:
            pred_dt= self.pred[:,:,:,:].reshape(-1)
            target_dt = self.target[:,:,...].reshape(-1)
            true_idx=np.where(target_dt.reshape(-1)==1)[0]
            false_idx= np.where(target_dt.reshape(-1)==0)[0]
            pred_dt_f = np.array(pred_dt[false_idx])#14280
            pred_dt_t = np.array(pred_dt[true_idx])#120

            # sns.set(style = "ticks")
            # sns.kdeplot(data=pred_dt_t, label="pred_T", shade=True, log_scale=True, color='r')
            # sns.kdeplot(data=pred_dt_f, label="pred_F", shade=True, log_scale=True)
            # plt.xlabel('pred_value(log_scale)')
            # plt.ylabel('KDE')
            # plt.title('Kernel density estimation')
            # plt.legend(loc=2)

            # plt.savefig(self.save_dir+'/KDE/'+'All_Focal_alpha0.01g6_KDE'+'.png', dpi=300)
            # plt.close()
            #################################################################
            sns.set(style = "ticks")
            sns.histplot(data=pred_dt_f, log_scale=True, kde=True,bins=30, label="pred_F")
            plt.legend(loc=3)
            ax2 = plt.twinx()
            sns.histplot(data=pred_dt_t, log_scale=True, kde=True,bins=30, label="pred_T", color='r',ax=ax2)
            plt.xlabel('pred_value')
            plt.ylabel('count')
            plt.legend(loc=2)
            plt.title('Inbalanced True and False')
            plt.xlim(10e-2,10e-1)

            plt.savefig(self.save_dir+'HIST/'+f'All_Focal_{inp_d}_Maxpool_{mp}_alpha{a}g{g}_hist'+'.png', dpi=300)
            plt.close()

    def plot_boxplot(self, case_idx):
        pred_dt= np.array(self.pred[0,case_idx,:,:].reshape(-1))
        # ax=sns.boxplot(x=pred_dt)
        # ax=sns.swarmplot(x=pred_dt)

        plt.boxplot(pred_dt, showfliers=True)
        plt.savefig(self.save_dir+'plotbox'+str(self.time[case_idx])+'.png', dpi=300)

    def cal_accuracy(self, case_idx, is_case, best, case_seq):
        for i in range(0,(case_seq-1)*6+1,6):
            best_is = best[int(i/6)]
            if is_case:
                pred= self.pred[0, case_idx+i, :,:].reshape(-1) #1 2133 120 120 (seq/batch/x/y)
                tar= self.target[case_idx+i, 0, :,:].reshape(-1)#
            else:
                pred= self.pred[0, :, :,:].reshape(-1) #1 2133 120 120 (seq/batch/x/y)
                tar= self.target[:, 0, :,:].reshape(-1)#
            pred=np.where(pred>best_is,1,0)
            accu = metrics.accuracy_score(tar, pred)
            print(f'Thereshold:{round(best_is,5)}, Accuracy:',round(accu,3))
            TP=0
            TN=0
            FP=0
            FN=0
            for i in range(len(pred)):
                if tar[i]==1:
                    if pred[i]==1:
                        TP+=1
                    elif pred[i]==0:
                        FN+=1
                elif tar[i]==0:
                    if pred[i]==1:
                        FP+=1
                    elif pred[i]==0:
                        TN+=1
            FPR= FP/(FP+TP)
            TPR= TP/(TP+FN)
         
            print('TPR:',round(TPR,5),'FPR:',round(FPR,5))
            print('TP:',TP,'TN:',TN,'FP:',FP,'FN:',FN) #幾乎都是FP(也就是明明沒有卻預報有)


    def plot_PFD(self, is_case, best, case_idx, grid_thresh, top):
        # self.pred= data_dic['out'] #1 2133 120 120 (seq/batch/x/y)
        # self.inp = data_dic['inp'] #2133 6 1 120 120 (batch/seq/radar/x/y)
        # self.target = data_dic['target'] #2133 1 120 120 (batch/seq/x/y)
        if is_case:
            last = 72 #意思是要看個案從初始往後推幾個10分鐘 72=6*12 看初始~12hrs after
            label = np.array(self.target)[case_idx:case_idx+last,0,...]
            pred = np.array(self.pred).transpose(1,0,2,3)[case_idx:case_idx+last,0,...]
            # inp(batch/seq/radarSCAN/x/y)
            self.baseline = self.inp[case_idx:case_idx+last,-1,0,:,:]
            #print('baseline_shape', self.baseline.shape) #79929,120,120
            self.baseline = nn.MaxPool2d(6)(self.baseline)
            if grid_thresh: 
                #若best是各網格點不一:
                #print(pred.shape)#79929 20 20
                best_new = np.zeros((pred.shape[0],20,20))
                best_new[:,...] = best #79929,20,20
                condition = pred>best_new
                pred_final = np.where(condition==True,1,0)
                preds=[pred_final, np.array(self.baseline)]
                threshold=[0,1.0] #分成0和1兩個標準
                PFD_2D(preds=preds, label=label, thresholds=threshold, names=['pred','BaseLine'])
                plt.tight_layout()
                plt.savefig(self.save_dir+'PFD/'+f'Case_top{top}_{lossf}_{inp_d}__Maxpool_{mp}_alpha{a}g{g}'+str(self.time[case_idx])+'.png', dpi=300)
            else:
                #若best為一個統一值
                pred1 = np.where(pred>=best[0],1,0)
                # pred2 = np.where(pred>=best[1],1,0)
                # pred3 = np.where(pred>=best[2],1,0)
                # pred4 = np.where(pred>=best[3],1,0)
                # pred5 = np.where(pred>=best[4],1,0)
                preds=[pred1, np.array(self.baseline)]
                #preds=[pred1, pred2, pred3, pred4, pred5, np.array(self.baseline)]
                threshold=[0,1.0] #分成0和1兩個標準
                PFD_2D(preds=preds, label=label, thresholds=threshold, names=['TOP3%','10%','30%','40%','50%','BaseLine'])
                plt.tight_layout()
                plt.savefig(self.save_dir+'PFD/'+f'Casenongrid_{lossf}_{inp_d}__Maxpool_{mp}_alpha{a}g{g}'+str(self.time[case_idx])+'.png', dpi=300)
        else:
            label = np.array(self.target)[:,0,...]
            pred = np.array(self.pred).transpose(1,0,2,3)[:,:,...]
            #若best為一個統一值
            # pred1 = np.where(pred>=best[0],1,0)
            # pred2 = np.where(pred>=best[1],1,0)
            # pred3 = np.where(pred>=best[2],1,0)
            # preds=[pred1, pred2, pred3, np.array(self.baseline)]

            #若best是各網格點不一:
            ##print(pred.shape)#79929 20 20
            best_new = np.zeros((pred.shape[0],20,20))
            best_new[:,...] = best #79929,20,20
            condition = pred>best_new
            pred_final = np.where(condition==True,1,0)
            preds=[pred_final, np.array(self.baseline)]

            
            #threshold=list(np.linspace(1.0))
            threshold=[0,1.0] #分成0和1兩個標準
            #PFD_2D(preds=preds, label=label, thresholds=threshold, names=['TOP3%','10%','15%','BaseLine'])
            PFD_2D(preds=preds, label=label, thresholds=threshold, names=['pred','BaseLine'])
            plt.tight_layout()
            plt.savefig(self.save_dir+'PFD/'+f'All_{lossf}_{inp_d}__Maxpool_{mp}_alpha{a}g{g}'+'.png', dpi=300)
    def last10m_Base(self, case_idx, case_seq): 
        #Scan baseliney
        cmap = mpl.colors.ListedColormap(['w','r'])
        bounds = [-0.1,0.5,1.1]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        
        mat = sio.loadmat('city_lonlat_region5.mat')
        citylon = mat['citylon']
        citylat = mat['citylat']
        del mat
        grid = 20
        lon=np.linspace(120.68105,122.17745,grid) # resolution=1.3 km
        lat=np.linspace(24.0675,25.56595,grid)
        lon2d, lat2d = np.meshgrid(lon, lat)
        fig, ax = plt.subplots(2, case_seq, figsize=(10, 8.5), dpi=200, facecolor='w')
        for time_step in range(case_seq):
            for y in range(2):   #y等於幾個inps, 假如有SCAN10m adn SCANtarget y=2
                if y ==0:
                    ax[y,time_step].set_title('Inp time:\n'+str(self.time[case_idx+time_step*6]),y=1.05,fontsize=15)
                ax[y,time_step].plot(citylon,citylat,'k',linewidth=0.6)
                ax[y,time_step].axis([120.6875, 122.1875, 24.0625, 25.5625])# whole area [119, 123, 21, 26]
                ax[y,time_step].set_aspect('equal')
            
                if y ==0:
                    # target = torch.from_numpy(self.inp) #轉torch是為了+maxpool
                    inp_last10m_max = (nn.MaxPool2d(6)(self.inp[:,:,0,...]))   #只取0(inp中的scan)做MAXPOOL
                    #print(np.array(inp_last10m_max).shape)  5469,6,20,20      
                    im=ax[y,time_step].pcolormesh(lon2d, lat2d, (inp_last10m_max[case_idx+time_step*6,-1,...]), edgecolors='none',
                                        shading='auto', norm=norm, cmap=cmap)
                    ax[y,time_step].set_ylabel('Inp-10m',rotation=0,fontsize=10) #設定每個小圖的y
                    ax[y,time_step].yaxis.set_label_coords(-0.2,0.9) #設定ylabel位置
                    #if time_step == case_seq-1:
                    #    cbar = fig.colorbar(im, ax=ax[y,time_step], orientation='vertical')
                else:
                    im=ax[y,time_step].pcolormesh(lon2d, lat2d, self.target[case_idx+time_step*6,0,:,:],norm=norm,cmap=cmap ,zorder=0)
                    ax[y,time_step].set_ylabel('GT',rotation=0,fontsize=10) #設定每個小圖的y
                    ax[y,time_step].yaxis.set_label_coords(-0.2,0.9) #設定ylabel位置
                    #if time_step == case_seq-1:
                    #    cbar = fig.colorbar(im, ax=ax[y,time_step], orientation='vertical')
                ax[y,time_step].set_xticks([])
                ax[y,time_step].set_yticks([])
        plt.tight_layout()
        plt.savefig(self.save_dir+f'/Last10mSCAN/'+f'Focal_{inp_d}'+str(self.time[case_idx])+'.png', dpi=300)
        plt.close()
        
    def main(self, is_case, case):
        case_idx = (self.time).index(case)#4773 這是對於20210604-0000此筆radar當最初始(-5)然後用-5~0(6筆)去預報+1~+6(0-1hr)
        if is_case:
            case_seq=3 #幾個時間序列
            top= 0.4#取前幾%
            grid_thresh = False
            AUC = False
            if AUC:
                best=self.plot_ROC(case_idx, is_case, case_seq)#先找出thereshold在把這個值丟入下一行去決定閥值        
            elif not AUC:
                if grid_thresh:
                    if testing =='201920':
                        best, Nofcase = threshold_points(self.pred, self.target, self.save_dir, top)
                        tmp= {'best':best, 'Nofcase':Nofcase}
                        with open(f'/wk171/peterpan/SCAN/SCAN_eval/grid_saver/grid_threshold_{lossf}_a={a}_{top}%.pkl', 'wb') as f:
                            pickle.dump(tmp, f)
                    if testing =='2021':
                        with open (f'/wk171/peterpan/SCAN/SCAN_eval/grid_saver/grid_threshold_{lossf}_a={a}_{top}%.pkl', 'rb') as f:               
                            tmp = pickle.load(f) #
                            best = tmp['best']
                            Nofcase = tmp['Nofcase']
                else:
                    top_percent = [0.03, 0.1, 0.3, 0.4, 0.5] #3 10 30 40 50%
                    #top_percent = [0.4]
                    best=hist_3percent(self.pred, self.target, self.save_dir, top_percent, a, g, lossf) 
                #
            #self.plot_space_correlation(best, Nofcase)
            #self.plot_pcolor(case_idx, best, case_seq,top, grid_thresh)    
            #self.plot_hitogram(case_idx, is_case)
            #self.cal_accuracy(case_idx, is_case, best, case_seq)
            #self.plot_boxplot(case_idx)
            #self.plot_PFD(is_case, best, case_idx, grid_thresh, top)
            #burst(self.tar, self.pred)
            #self.plot_CV(case_idx, best, case_seq)
            #self.plot_new(case_idx, best, case_seq)
            #self.last10m_Base(case_idx, case_seq)
            #self.stats()   

        
        else:
            case_seq=1
            #best=self.plot_ROC(case_idx, is_case, case_seq) 
            #self.plot_hitogram(case_idx, is_case)
            #self.cal_accuracy(case_idx, is_case, best, case_seq)
            #self.plot_boxplot(case_idx)
            best=[0.85]
            self.plot_PFD(is_case, best, case_idx)
            

if __name__ == '__main__' :
    os.system('export DISPLAY=:1')
    root  = '/wk171/peterpan/SCAN/output_SCAN/'
    plot_save_dir = '/wk171/peterpan/SCAN/SCAN_eval/SCAN_plot_out/'
    #TODO list->之後改成argsparser
    a=0.25
    g=2
    lossf='BCElogits' #Focal or BCElogits
    mp='YES' #YES or other
    inp_d = 'SR' #SCAN OR RADAR or SR
    testing = '201920'
    if lossf =='Focal':
        if mp=='YES':
            data_dir = f'{lossf}_{inp_d}_Maxpool_{mp}_{testing}_alpha{a}g{g}_data.pkl'
            time_dir = f'{lossf}_{inp_d}_Maxpool_{mp}_{testing}_alpha{a}g{g}_timeinfo.pkl'
        else:
            data_dir = f'{lossf}_{inp_d}_Maxpool_{mp}_{testing}_alpha{a}g{g}_data.pkl'
            time_dir = f'{lossf}_{inp_d}_Maxpool_{mp}_{testing}_alpha{a}g{g}_timeinfo.pkl'
    else:
        if mp=='YES':
            data_dir = f'{lossf}_{inp_d}_Maxpool_{mp}_{testing}_data.pkl'
            time_dir = f'{lossf}_{inp_d}_Maxpool_{mp}_{testing}_timeinfo.pkl'
        else:
            data_dir = f'{lossf}_{inp_d}_Maxpool_{mp}_{testing}_data.pkl'
            time_dir = f'{lossf}_{inp_d}_Maxpool_{mp}_{testing}_timeinfo.pkl'
    p = Plot_scan(root, plot_save_dir, data_dir, time_dir)

    for i in range(1): #2021/6/4/0400 TS 2018/5/7/1400鋒面 2021/6/22/0400 滯留鋒面
        case = datetime(2021,6,4,4+i*3,0)#choose the interesting case time
        p.main(is_case=True, case=case)
    