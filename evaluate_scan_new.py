# conda install -c conda-forge basemap-data-hires
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
import sys
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
os.environ['CUDA_LAUNCH_BLOCKING'] ='1'
os.environ['CUDA_VISIBLE_DEVICES']="0"

os.environ['ROOT_DATA_DIR']='/wk171/peterpan/SCAN/'
sys.path.append("/wk171/peterpan/SCAN/SCAN_train/")
from data_loaders.data_loader_all_loaded import DataLoaderAllLoaded
from datetime import datetime
from utils.run_utils import checkpoint_parser, get_model
from core.enum import DataType
from models.loss import Focalloss
#這邊要改時間, sampling rate 改成1
s = datetime(2019,1,1,0,0)
e = datetime(2021,12,31,23,50)   #if for 2021/06/01->2021/06/30原先個案
input_shape=(120,120) #120.120
num_workers = 8
sampling_rate= 1
is_test = True
is_validation=False
is_train=False
Root = '/wk171/peterpan/'

#輸入資料Radar+ maxpool
#ADV+BCEwithlogits
#checkpoint_test = Root+'SCAN_checkpoints/RF_10101_41231_mt-19_dt-16_lt-16_tlen-1_la-0_lsz-5_res-0_ilen-6_scan-1_maxpool_atlast-1_AdvW-0.01_DisD-3_v-97_epoch=8_val_loss=0.039656.ckpt'
#ADV+FOCAL(a=0.999,g=2)(OK)
#checkpoint_test = Root+'SCAN_checkpoints/RF_10101_41231_mt-19_dt-16_lt-15_tlen-1_la-0_lsz-5_res-0_ilen-6_scan-1_maxpool_atlast-1_AdvW-0.01_DisD-3_v-117_epoch=2_val_loss=0.000347.ckpt'


#輸入資料SCAN+ maxpool
#ADV+FOCAL(a=0.999,g=2)(OK)
#checkpoint_test = Root+'SCAN_checkpoints/RF_10101_41231_mt-19_dt-256_lt-15_tlen-1_la-0_lsz-5_res-0_ilen-6_scan-1_maxpool_atlast-1_AdvW-0.01_DisD-3_v-0_epoch=5_val_loss=0.000433.ckpt'



# 輸入資料SR+ maxpool
#ADV+FOCAL(a=0.999,g=2)(OK)
checkpoint_test = Root+'SCAN_checkpoints/RF_10101_41231_mt-19_dt-272_lt-15_tlen-1_la-0_lsz-5_res-0_ilen-6_scan-1_maxpool_atlast-1_AdvW-0.01_DisD-3_v-0_epoch=5_val_loss=0.000353.ckpt'
#ADV+FOCAL(a=0.99,g=2) (OK)
#checkpoint_test = Root+'SCAN_checkpoints/RF_10101_41231_mt-19_dt-272_lt-15_tlen-1_la-0_lsz-5_res-0_ilen-6_scan-1_maxpool_atlast-1_AdvW-0.01_DisD-3_v-99_epoch=3_val_loss=0.001318.ckpt'

##TODO
mp='YES' ##maxpool or not /YES or NO
losstype = 'FOCAL' # FOCAL or BCE
a=0.999
g=2
input_d = 'SR' #SCAN OR RADAR OR SR
maxpool_atlast = int(1) # 1(YES) or 0(NO)

kwargs = checkpoint_parser(checkpoint_test)
for k in kwargs.keys():
    if k in ['lw','adv_w','AdvW','auc','D_pos_acc','D_neg_acc','D_auc','lssim','lmae']:
        kwargs[k] = float(kwargs[k])
    elif k not in ['start','end','val_loss','pdsr']:
        kwargs[k] = int(kwargs[k])


model_type = int(kwargs['mt'])
target_len = int(kwargs['tlen'])
just_rain = bool(kwargs.get('jst_rain',0))
just_radar = bool(kwargs.get('jst_radar',0))
residual = bool(kwargs.get('res',0))
hourly_data = bool(kwargs.get('hrly',0))
input_len = int(kwargs.get('ilen',5))
target_offset = int(kwargs.get('toff', 0))
random_std = int(kwargs.get('r_std', 0))

if 'dt' not in kwargs:
    if just_rain:
        data_type = DataType.Rain
    elif just_radar:
        data_type = DataType.Radar
    else:
        data_type = DataType.Rain + DataType.Radar + DataType.Altitude +DataType.Scan
else:
    data_type=int(kwargs['dt'])


data_kwargs = {
    'data_type': data_type,
    'residual': residual,
    'target_offset': target_offset,
    'target_len': target_len,
    'input_len': input_len,
    'hourly_data': hourly_data,
    'hetero_data': bool(kwargs.get('hetr',0)),
    'SCAN_data' : bool(int(kwargs.get('SCAN_data', 1))),
    'maxpool_atlast' : bool(int(kwargs.get('maxpool_atlast', maxpool_atlast))),  #maxpool at output or not
    'sampling_rate': sampling_rate,
    'prior_dtype': DataType.NoneAtAll,
    'random_std': random_std,
    'threshold': 0.5,
}
model_kwargs = {
        'adv_w': float(kwargs.get('adv_w', 0.1)),
        'model_type': model_type,
        'dis_d': int(kwargs.get('dis_d', 3)),
        'teach_force': int(-1) # for testing
    }

loss_kwargs = {'type': kwargs['lt'], 
               'aggregation_mode': kwargs.get('la'), 
               'kernel_size': kwargs.get('lsz'),
               'w': float(kwargs.get('lw', 1)),
               'residual_loss':None,
               'mae_w':0.1,
               'ssim_w':0.02,
              }

dataset = DataLoaderAllLoaded(s,e,input_len,target_len, 
                              workers=8,                               
                              target_offset=int(kwargs.get('toff',0)),
                              data_type=data_type,
                              is_validation=not is_test,
                              is_test = is_test,
                              img_size=input_shape,
                              residual=residual,
                              hourly_data=hourly_data,
                              hetero_data=data_kwargs['hetero_data'],
                              SCAN_data = data_kwargs['SCAN_data'],
                              maxpool_atlast = data_kwargs['maxpool_atlast'],
                              sampling_rate=sampling_rate,
                              threshold=data_kwargs['threshold']
                             )

batch_size=dataset.__len__()
loader = DataLoader(dataset, batch_size=16, num_workers=num_workers, shuffle=False)


model = get_model(
        s,
        e,
        model_kwargs,
        loss_kwargs,
        data_kwargs,
        '',
        '',
        data_loader_info=dataset.get_info_for_model(),
    )

initial_saver = []
for i in range(len(dataset)):#5469
    tmp = dataset._get_internal_index(i)
    initial_saver.append(dataset._time[tmp])
    del tmp

total_time = {}
total_time["initial"]=initial_saver
if losstype=='FOCAL':
    with open(Root+f'SCAN/output_SCAN/Focal_{input_d}_Maxpool_{mp}_alpha{a}g{g}_timeinfo.pkl', 'wb') as f:
        pickle.dump(total_time, f)
else:
    with open(Root+f'SCAN/output_SCAN/BCElogits_{input_d}_Maxpool_{mp}_timeinfo.pkl', 'wb') as f:
        pickle.dump(total_time, f)


#GPU
checkpoint = torch.load(checkpoint_test)
_ = model.load_state_dict(checkpoint['state_dict'])
model = torch.nn.DataParallel(model).cuda()

# CPU
# checkpoint = torch.load(checkpoint_test, map_location=torch.device('cpu'))
# _ = model.load_state_dict(checkpoint['state_dict']) #

#print(checkpoint['state_dict'].keys())
#checkpoints裡面有很多訓練好的model的資訊
#其中state_dict是所有model(encoder/forecaster/discriminator/attention)的權重(W)與偏值(B)




#criterion = nn.BCEWithLogitsLoss()#get_criterion(loss_kwargs)
criterion = Focalloss()
loss_dict = {i:[] for i in range(target_len)} 
#print(loss_dict) {0:[]}

model.eval()
loss_sum= 0
loss_p_sum=0
loss_n_sum=0
loss_BCE_sum=0
N=0
n_p=0
p_num=0
n_num=0
with torch.no_grad():
    tmp={}
    in_list =[]
    out_list = []
    target_list=[]
    for batch in tqdm(loader):
        #print(np.array(batch[1]).shape)
        inp,target = batch
        # if torch.max(inp).item() > 5:
        #     print(torch.max(inp).item())
        batch_size_in= inp.shape[0]
        inp_size = inp.shape
        tar_size = target.shape
        #print(batch_size_in)
        #print(inp.shape)   #16 6 2 120 120(batchsize, seq, radar, x,y)
        #print(target.shape) #16 1 120 120(batchsize, seq, x,y)
        inp= inp.cuda()
        target = target.cuda()
        output= model(inp)
        #print(output.shape) #1 5469 120 120
        inp = inp.cpu()
        target = target.cpu()
        output = output.cpu()
        in_list = np.append(in_list , np.array(inp).reshape(-1))
        out_list = np.append(out_list , np.array(output).reshape(-1))
        target_list = np.append(target_list , np.array(target).reshape(-1))
        # tmp['inp'] =in_list
        # tmp['out']= out_list
        # tmp['target'] = target_list
        # if losstype=='FOCAL':
        #     with open(Root+f'SCAN/output_SCAN/input_onlyscan/Focal_{input_d}_Maxpool_{mp}_alpha{a}g{g}_data.pkl', 'wb') as f:
        #         pickle.dump(tmp, f)
        # else:
        #     with open(Root+f'SCAN/output_SCAN/input_onlyscan/BCElogits_{input_d}_Maxpool_{mp}_data.pkl', 'wb') as f:
        #         pickle.dump(tmp, f)
        for t in range(target_len): 
                     
            output_t = output[t:t+1,...]   
            target_t = target[:,t:t+1,:,:]        
            output_t = output_t.permute(1,0,2,3)
            #torch轉成純量(.item())
            #For focal
            if losstype == 'FOCAL':
                loss_dict[t].append(criterion(output_t, target_t, alpha=a,gamma=g, reduction='mean').item()) 
                loss_sum+= criterion(output_t, target_t, alpha=a,gamma=g, reduction='mean').item()*batch_size_in  
            else:
            #For BCE
                loss_dict[t].append(nn.BCEWithLogitsLoss()(output_t, target_t).item())
                loss_sum+= nn.BCEWithLogitsLoss()(output_t, target_t).item()*batch_size_in
        #選出答案為+，預報的分數，以及答案為-，預報的分數
        p_idx=torch.where(target==1)
        output=output.permute(1,0,2,3)
        n_idx=torch.where(target==0)
        predp =  output[p_idx]
        predn =  output[n_idx]
        tarp = target[p_idx]
        tarn = target[n_idx]
        #非空才計算
        if np.isnan(nn.BCEWithLogitsLoss()(predp, tarp).item()):
            pass
        else:
            loss_p_sum+= nn.BCEWithLogitsLoss()(predp, tarp).item()*batch_size_in   
            n_p+=batch_size_in
        loss_n_sum+= nn.BCEWithLogitsLoss()(predn, tarn).item()*batch_size_in   
        loss_BCE_sum+= nn.BCEWithLogitsLoss()(output, target).item()*batch_size_in
        p_num+=len(predp)
        n_num+=len(predn)
        N+=batch_size_in
    #final reshape them
    in_list = np.array(in_list).reshape(-1,6,inp_size[2],120,120) #(batchsize, seq, radar, x,y)
    out_list = np.array(out_list).reshape(1,-1,tar_size[2],tar_size[-1]) #(seq, batchsize, x,y)
    target_list = np.array(target_list).reshape(-1,1,tar_size[2],tar_size[-1]) #(batchsize, seq, x,y)
    in_list = torch.Tensor(in_list)
    out_list = torch.Tensor(out_list)
    target_list = torch.Tensor(target_list)
    tmp['inp'] =in_list
    tmp['out']= out_list
    tmp['target'] = target_list
    if losstype=='FOCAL':
        with open(Root+f'SCAN/output_SCAN/Focal_{input_d}_Maxpool_{mp}_alpha{a}g{g}_data.pkl', 'wb') as f:
            pickle.dump(tmp, f)
    else:
        with open(Root+f'SCAN/output_SCAN/BCElogits_{input_d}_Maxpool_{mp}_data.pkl', 'wb') as f:
            pickle.dump(tmp, f)
print('Total batchs is', N) 
print(f'Loss {losstype}_sum is', loss_sum)
print('Loss BCE_sum of p+', loss_p_sum)
print('Loss BCE_sum of n-', loss_n_sum)
print('Total BCE_sum loss', loss_BCE_sum)

for t in range(target_len):
    print(f'Avg Loss for {t} to {t+1} Hour is: ', round(loss_sum/N,4) )
print('Total Avg Focal_Loss is : ', round(loss_sum/N,4))
print('Total Ave + BCE_loss is:', round(loss_p_sum/n_p,4), 'Pt is', round(np.exp(round(-loss_p_sum/n_p,4)),2))
print('Total Ave - BCE_loss is:', round(loss_n_sum/N,4), 'Pt is', round(np.exp(round(-loss_n_sum/N,4)),2))
print('Total Ave BCE_loss is:', round(loss_BCE_sum/N,4), 'Pt is', round(np.exp(round(-loss_BCE_sum/N,4)),2))
print(f'負樣本為正樣本的{round(n_num/p_num,0)}倍')
# for t in range(1,target_len+1):
#     print(f'Avg Loss for {t-1} to {t} Hour is: ', np.mean(loss_dict[t-1]) )
    
# print('Total Avg Loss is : ', np.mean([np.mean(v) for _,v in loss_dict.items()]))



