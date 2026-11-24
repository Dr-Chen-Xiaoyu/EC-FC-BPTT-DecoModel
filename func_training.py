import jax
import brainpy as bp
import brainpy.math as bm
import numpy as np
from sklearn.model_selection import KFold

from func_brainpy_metrics import correlation_similarity_flatt,correlation_similarity_batch,cosine_similarity_flatt,cosine_similarity_batch

#%% tool functions:
def split_dict_by_key(dictionary):
    sub_dict_list=[]
    for key in dictionary.keys():
        sub_dict_list.append({key: dictionary[key]})
    return sub_dict_list


def KFold_connectome(input_conn,fold_i,fold_N,random_state=123,type='EC'):
    conn=np.copy(input_conn) # 复制一份
    kf = KFold(n_splits=fold_N, shuffle=True, random_state=random_state) 

    if type=='EC' : # 随机将不是NaN的元素进行Kfold，进一步将test的元素设为NaN
        IaN_rc = np.argwhere(~np.isnan(conn)) # rc[row, col] of IaN(is a number) elements
        folds=[(tra,tes) for tra,tes in kf.split(IaN_rc)] # 分Kfold

        tra, tes = folds[fold_i] # 选出这次训练所用的fold
        IaN_rc_test=IaN_rc[tes,:] # 及其test的[row,col]
        conn[IaN_rc_test[:,0],IaN_rc_test[:,1]] = np.nan # mask掉test的部分

    elif type=='FC' : # 随机将上三角的元素Kfold，将test的元素和对角设为NaN
        triu = np.triu(np.ones_like(conn), k=1)>0 # Extract the upper triangle of the matrix
        triu_rc = np.argwhere(triu) # [row,col]

        folds=[(tra,tes) for tra,tes in kf.split(triu_rc)] #分fold
        tra, tes = folds[fold_i] #选出这次训练所用的fold
        
        triu_rc_tra=triu_rc[tra,:] #及其train的[row,col]
        cvmask_tra=np.zeros_like(conn)
        cvmask_tra[triu_rc_tra[:,0],triu_rc_tra[:,1]] = 1 # mask掉
        cvmask_tra=cvmask_tra+cvmask_tra.T

        triu_rc_tes=triu_rc[tes,:] #及其test的[row,col]
        cvmask_tes=np.zeros_like(conn)
        cvmask_tes[triu_rc_tes[:,0],triu_rc_tes[:,1]] = 1 # mask掉
        cvmask_tes=cvmask_tes+cvmask_tes.T

        conn[cvmask_tra==0]=np.nan

    return conn


#%% EC training data get:
def get_stim_timing(
        dt, # 2000 Hz
        pre_time=1,# 1 seconds, 1000 ms
        pos_time=1,# 1 seconds, 1000 ms
        ):

    num_timepoint=int(1+(pre_time+pos_time)/dt)
    ts = np.linspace(-pre_time, pos_time, num=num_timepoint, endpoint=True)
    
    stim_onset = int(np.where(ts==0)[0])# 0 time point as stim-onset 
    pre_period = (ts>=(-200/1000))&(ts<(-10/1000)) # [-200, -10)ms 和 ftract 一致
    pos_period = (ts>(10/1000))&(ts<=(200/1000))   # (10, 200]ms

    return ts,stim_onset,pre_period,pos_period


def training_data_synthesis(target,EC,N,ts,stim_onset):
    p = EC[:,target] #每一个column代表刺激点以及其他节点的响应的概率,基于ftract_ctx
    
    #基于nan来产生mask
    mas = ~np.isnan(p)
    
    #强度等于probability
    res = np.copy(p)
    res[~mas]=0
    
    #输入刺激
    stiminput = np.zeros((ts.size,N))
    stiminput[stim_onset,target]=1.0

    #对应的mask
    mask_node = np.zeros((ts.size,N))
    mask_node[:,:]=mas

    if_Not_all_masked = np.sum(mas)>0

    return stiminput, res, mask_node,if_Not_all_masked


def get_training_data_EC(N, EC, stim_timing, if_delete_ALLmaskedNodes=False, if_add_epsi_to_zeros=False, epsi=1e-5):
    
    ts,stim_onset,pre_period,pos_period = stim_timing

    Batch_size = N
    tr_EC        = np.zeros((Batch_size,N)) #(batch,node) 
    tr_stiminput = np.zeros((Batch_size,ts.size,N)) #(batch,time,node) 
    tr_mask_node = np.zeros((Batch_size,ts.size,N)) # mask掉每次刺激所没有记录的节点
    tr_mask_diag = np.ones ((Batch_size,ts.size,N))
    tr_mask_pre  = np.zeros((Batch_size,ts.size,N))
    tr_mask_pos  = np.zeros((Batch_size,ts.size,N))
    
    tr_mask_pre[:,pre_period,:] = 1.0
    tr_mask_pos[:,pos_period,:] = 1.0
    tr_mask_time = tr_mask_pos-tr_mask_pre

    used_ids = []
    for i in range(Batch_size): 
        stiminput, res, mask_node, if_Not_all_masked = training_data_synthesis(target=i,EC=EC,N=N,ts=ts,stim_onset=stim_onset)
        tr_stiminput[i,:,:] = stiminput
        tr_EC[i,:] = res
        tr_mask_node[i,:,:] = mask_node
        tr_mask_diag[i,:,i] = 0.0 #刺激的靶点，其接受的靶点mask掉

        if if_Not_all_masked:
            used_ids.append(i)

    # further preprocessing
    if if_delete_ALLmaskedNodes:
        print("len of used_ids = ", len(used_ids))
        tr_EC           = tr_EC[used_ids,:]
        tr_stiminput    = tr_stiminput[used_ids,:,:]#全0有的时候会出NaN
        tr_mask_node    = tr_mask_node[used_ids,:,:]
        tr_mask_diag    = tr_mask_diag[used_ids,:,:]
        tr_mask_pre     = tr_mask_pre [used_ids,:,:]
        tr_mask_pos     = tr_mask_pos [used_ids,:,:]
        tr_mask_time    = tr_mask_time[used_ids,:,:]

    if if_add_epsi_to_zeros:
        tr_EC[tr_EC==0]=epsi
        tr_stiminput[tr_stiminput==0]=epsi
        tr_mask_node[tr_mask_node==0]=epsi
        tr_mask_diag[tr_mask_diag==0]=epsi
        tr_mask_pre[tr_mask_pre==0]=epsi
        tr_mask_pos[tr_mask_pos==0]=epsi
        tr_mask_time[tr_mask_time==0]=epsi

    return tr_EC,tr_stiminput,tr_mask_node,tr_mask_diag,tr_mask_pre,tr_mask_pos,tr_mask_time


#%% EC's loss function:
def get_loss_fun_EC(model,architecture):

    if architecture == 'outLinear':
        def loss_fun_EC(inputs, EC_target, masks):
            mask_time, mask_node, mask_diag = masks

            runner = bp.DSTrainer(model, progress_bar=False, numpy_mon_after_run=False)
            outputs = runner.predict(inputs, reset_state=False)

            EC_predict = bm.mean(outputs*mask_time*mask_node ,axis=1) # 用mean的话，开始scale比较小，a从小达大

            return bp.losses.mean_squared_error(EC_predict,EC_target) # MSE loss

    if architecture == 'corrLoss':
        def loss_fun_EC(inputs, EC_target, masks):
            mask_time, mask_node, mask_diag = masks

            runner = bp.DSTrainer(model, progress_bar=False, numpy_mon_after_run=False)
            outputs = runner.predict(inputs, reset_state=False)

            EC_predict = bm.sum(outputs*mask_time*mask_node ,axis=1) # 后面直接计算correlation

            return 1-correlation_similarity_flatt(EC_predict, EC_target) # correlation loss
        
    if architecture == 'maxNorm':
        def loss_fun_EC(inputs, EC_target, masks):
            mask_time, mask_node, mask_diag = masks

            runner = bp.DSTrainer(model, progress_bar=False, numpy_mon_after_run=False)
            outputs = runner.predict(inputs, reset_state=False)

            EC_predict = bm.sum(outputs*mask_time*mask_diag ,axis=1)
            EC_predict = EC_predict/bm.max(EC_predict) # EC直接强制rescale到[0 1]之间，使用除对角线外的原始完整矩阵

            EC_predict = EC_predict*(mask_node.mean(axis=1)) # 只比较测量的边
            
            return  bp.losses.mean_squared_error(EC_predict,EC_target) # MSE loss
        
    if architecture == 'maxNorm(node strength)':
        def loss_fun_EC(inputs, EC_target, masks):
            mask_time, mask_node, mask_diag = masks

            runner = bp.DSTrainer(model, progress_bar=False, numpy_mon_after_run=False)
            outputs = runner.predict(inputs, reset_state=False)

            EC_predict = bm.sum(outputs*mask_time*mask_diag ,axis=1)
            EC_predict = EC_predict/bm.max(EC_predict) # EC直接强制rescale到[0 1]之间，使用除对角线外的原始完整矩阵

            EC_predict = EC_predict*(mask_node.mean(axis=1)) # 只比较测量的边

            #某节点的在刺激所有其他节点时的平均响应（入度afferent,nan已经mask掉
            NS_aff_predict = bm.mean(EC_predict,axis=0) 
            NS_aff_target  = bm.mean(EC_target,axis=0) #tr_EC做过transpose,因此batch_axis(axis=0)为刺激维度并沿着此维度求平均
            
            return  bp.losses.mean_squared_error(NS_aff_predict,NS_aff_target) # MSE loss

        
    return loss_fun_EC


#%% EC's training function:
def get_train_fun(train_dict,loss_fun,optim_func,lr,if_largerLRforW=False,scale_largerLRforW=1,list_scale_vars=['DecoModel0.w',], clip_epsilon=1e10):

    grad_fun = bm.grad(loss_fun, grad_vars=train_dict, has_aux=False, return_value=True)

    # define training function
    if if_largerLRforW:#w的LR再乘以一个系数
        
        train_dict1=train_dict.copy()
        train_dict2=train_dict.copy()

        for i in list_scale_vars: train_dict1.pop(i) 
        list_other_vars = list(train_dict1.keys())
        for i in list_other_vars: train_dict2.pop(i)

        print('\ntrain_dict1:',train_dict1)
        print('\ntrain_dict2:',train_dict2)

        opt1= optim_func(train_vars=train_dict1 , lr=lr, ) 
        opt2= optim_func(train_vars=train_dict2 , lr=lr*scale_largerLRforW, ) 

        print('\nopt1:',opt1)
        print('\nopt2:',opt2)

        # training function
        @bm.jit
        def train_fun(xs, ys, masks, *args):
            grads, loss = grad_fun(xs, ys, masks, *args)
            grads = jax.tree_util.tree_map(lambda g:bm.clip_by_value(g,-clip_epsilon,clip_epsilon) ,grads) # clipping gradient by value

            grads1=grads.copy()
            grads2=grads.copy()

            for i in list_scale_vars: grads1.pop(i)
            for i in list_other_vars: grads2.pop(i)
    
            opt1.update(grads1)
            opt2.update(grads2)
            return grads, loss
        
    else:
        opt = optim_func( lr=lr, train_vars=train_dict )# define the optimizer we need
        print('opt:',opt)

        @bm.jit
        def train_fun(xs, ys, masks, *args):
            grads, loss = grad_fun(xs, ys , masks, *args)
            grads = jax.tree_util.tree_map(lambda g:bm.clip_by_value(g,-clip_epsilon,clip_epsilon) ,grads) # clipping gradient by value
            opt.update(grads)
            return grads, loss
    
    return grad_fun,train_fun


def get_train_EC(model,loss_fun_EC,optimize_method,lr,if_largerLRforW=False,scale_largerLRforW=1,list_scale_vars=['DecoModel0.w',],clip_epsilon=1e10):
    
    # 确定训练参数
    train_dict = model.train_vars().unique()
    print('train_dict:',train_dict)

    if optimize_method == 'Adam':
        optim_func = bp.optim.Adam

    grad_fun_EC,train_EC = get_train_fun(train_dict,loss_fun_EC,optim_func,lr,if_largerLRforW,scale_largerLRforW,list_scale_vars, clip_epsilon)
    
    return grad_fun_EC,train_EC


#%% FC training data get: 
def get_training_data_FC(FC,Batch_size,duration,N,dt,epsi=1e-10,seed=123):
    
    tr_FC = np.expand_dims(FC,0).repeat(Batch_size,axis=0) # batch-wise广播FC
    tr_FC[np.isnan(tr_FC)]=epsi # NaN全部置为epsilon
    
    # mask掉NaN和对角元素,mask里面已经除掉了n(即duration) for Cov calculation (1/n)X*XT
    mask=np.ones_like(FC)
    mask[np.isnan(FC)]=epsi
    mask[np.diag_indices_from(mask)]=epsi
    tr_mask = np.expand_dims(mask,0).repeat(Batch_size,axis=0)/duration
    
    tr_noiseinput = np.random.RandomState(seed).randn(Batch_size,duration,N)*np.sqrt(dt) # (batch,time,node) 生成随机时间序列
    
    return tr_FC,tr_noiseinput,tr_mask


def def_noiseinput_generator(rng,Batch_size,duration,N,dt):
    @bm.jit
    def noiseinput_generator():
        tr_noiseinput = rng.randn(Batch_size,duration,N)*np.sqrt(dt) # (batch,time,node) 生成随机时间序列
        return tr_noiseinput
    return noiseinput_generator


#%% FC's loss function:
def get_loss_fun_FC(model, N, sigma=1e-5, losstype='mse', if_train_sigma = True,):
    class loss_function_FC(bp.BrainPyObject):
        def __init__(self, N, sigma, losstype, if_train_sigma):

            super(loss_function_FC, self).__init__()

            if if_train_sigma:
                if isinstance(sigma, float):
                    self.sigma = bm.TrainVar(sigma*bm.ones(N)) # (num,) with same initialization, last axis-based broadcatsing operation, N is the node   
                else: # bm.array:
                    self.sigma = bm.TrainVar(sigma) # (1,) or (num,)
            else:
                self.sigma = sigma # float or (1,) (num,) bm.array

            if losstype == 'mse':
                self.loss_fun = bp.losses.mean_squared_error # bm.mean(bp.losses.mean_squared_error(FC_predict, FC_target, axis=[1,2])) # 第一个维度为batch维，先MSE再平均,一样
            if losstype == 'cor':
                self.loss_fun = lambda x,y: 1-bm.mean(correlation_similarity_batch(x,y))
            if losstype == 'cos':
                self.loss_fun = lambda x,y: 1-bm.mean(cosine_similarity_batch(x,y))
            if losstype == 'mse(node strength)':
                self.loss_fun = lambda x,y: bp.losses.mean_squared_error(bm.mean(x,axis=-1), bm.mean(y,axis=-1))


        def __call__(self, inputs, FC_target, masks):
            mask = masks #  mask掉NaN和对角元素,mask里面已经除掉了n(即duration) for Cov calculation (1/n)X*XT

            runner = bp.DSTrainer(model, progress_bar=False, numpy_mon_after_run=False)
            output = runner.predict(inputs*bm.abs(self.sigma), reset_state=False)

            X = bm.normalize(output,axis=1)
            FC_predict = bm.matmul(X.transpose(0,2,1),X) * mask

            return self.loss_fun(FC_predict, FC_target)

    return loss_function_FC(N=N,sigma=sigma,losstype=losstype, if_train_sigma=if_train_sigma)


#%% FC's training function:
def get_train_FC(model,loss_fun_FC,optimize_method,lr, if_largerLRforW=False,scale_largerLRforW=1,list_scale_vars=['DecoModel0.w',], clip_epsilon=1e10):

    # 确定训练参数
    train_dict = model.train_vars().unique()
    train_dict.update(loss_fun_FC.train_vars().unique())
    print('train_dict:',train_dict)

    if optimize_method == 'Adam':
        optim_func = bp.optim.Adam

    grad_fun_FC,train_FC = get_train_fun(train_dict,loss_fun_FC,optim_func,lr,if_largerLRforW,scale_largerLRforW,list_scale_vars,clip_epsilon)
    
    return grad_fun_FC,train_FC
