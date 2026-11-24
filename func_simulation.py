import numpy as np  
import brainpy as bp
import brainpy.math as bm

import os,sys
sys.path.append("/data/xyc/codes/iEEG_network/4_model_BrainPy")
from func_brainpy_model import get_AbbottChance,DecoModel,outBalloon


#%% tool functions
def get_parallel_parameters_from_bppaths(bppaths):

    paraN = bppaths.size

    #构建pmap并行计算的输入参数
    states = bp.checkpoints.load_pytree(bppaths[0])
    try:
        G = states['model']['DecoModel0']['DecoModel0.G']
        w = states['model']['DecoModel0']['DecoModel0.w']
        I = states['model']['DecoModel0']['DecoModel0.I']
    except:
        G = states['epoch_G'][-1]
        w = states['epoch_w'][-1]
        I = states['epoch_I'][-1]
    S = states['model']['DecoModel0']['DecoModel0.S']

    parallel_G=np.empty((paraN,))
    parallel_w=np.empty((paraN,w.shape[0]))
    parallel_I=np.empty((paraN,I.shape[0]))
    parallel_S=np.empty((paraN,S.shape[0],S.shape[1]))

    try:
        sigma = states['loss_fun_FC']['loss_function_FC0']['loss_function_FC0.sigma']
        parallel_sigma=np.empty((paraN,sigma.shape[0]))
    except:
        parallel_sigma=np.zeros((paraN,1))

    for i in range(paraN):

        print("{}/{}".format(i,paraN))

        states = bp.checkpoints.load_pytree(bppaths[i])

        try:
            G = states['model']['DecoModel0']['DecoModel0.G']
            w = states['model']['DecoModel0']['DecoModel0.w']
            I = states['model']['DecoModel0']['DecoModel0.I']
        except:
            G = states['epoch_G'][-1]
            w = states['epoch_w'][-1]
            I = states['epoch_I'][-1]
        S = states['model']['DecoModel0']['DecoModel0.S']

        parallel_G[i]=G
        parallel_w[i,:]=w
        parallel_I[i,:]=I
        parallel_S[i,:,:]=S

        try:
            sigma = states['loss_fun_FC']['loss_function_FC0']['loss_function_FC0.sigma']
            parallel_sigma[i,:]=sigma
        except:
            pass

    return parallel_G,parallel_w,parallel_I,parallel_sigma,parallel_S


#%% body function for parallized running of FC
def def_bodyFC(struc_conn_matrix, restinput, warmupinput=None, ifoutBalloon=False):
    N = struc_conn_matrix.shape[0]

    if warmupinput is None: # 如果不给warmup，就以时间维度随机打乱restinput作为warmup
        warmupinput=bm.random.permutation(restinput,axis=1) # (1,int(duration/bm.dt),N)

    if ifoutBalloon:
        class MODEL(bp.DynamicalSystemNS):
            def __init__(self, size, struc_conn_matrix, G, w, I, H_x_act, S_init):
                super(MODEL, self).__init__()
                self.rnnLayer = DecoModel(size=size, struc_conn_matrix=struc_conn_matrix, G=G, w=w, I=I, H_x_act = H_x_act, S_init=S_init)
                self.outLayer = outBalloon(size=size,)

            def reset_state(self, batch_size=1): # this function defines how to reset the mode states
                self.rnnLayer.reset_state(batch_size)
                self.outLayer.reset_state(batch_size)
            
            def reset_init(self,):
                self.rnnLayer.reset_init()

            def update(self, x = 0):
                return self.outLayer(self.rnnLayer(x))
    else:
        MODEL=DecoModel

    def bodyFC(G,w,I,sigma,S=0.0):
        # create instance of model
        model = MODEL(size = N, struc_conn_matrix = struc_conn_matrix, G=G, w=w, I=I, H_x_act = get_AbbottChance(method = 'custom_jvp'), S_init=S)
        runner = bp.DSRunner(model, progress_bar=False, numpy_mon_after_run=False,)
        
        # running model
        _ = runner.predict(inputs = warmupinput*bm.abs(sigma), reset_state=False) # warm-up
        out = runner.predict(inputs = restinput*bm.abs(sigma), reset_state=False) # bacth-wise give resting-state input (white noise)

        FC = bm.corrcoef(out[0,:,:],rowvar=False) # FC = bp.measure.functional_connectivity(out[0,:,:],numpy=False)

        return out,FC
    
    return bodyFC


#%% body function for parallized running of EC
def def_bodyEC(struc_conn_matrix):

    N = struc_conn_matrix.shape[0]
    Batch_size=N

    # define the time period of stimulus
    duration=4001
    ts = np.linspace(-1, 1, num=duration, endpoint=True)
    stim_onset = int(np.where(ts==0)[0])
    pre_period = (ts>=(-200/1000))&(ts<(-10/1000))    # [-200, -10)ms  和ftract一致
    pos_period = (ts>(10/1000))&(ts<=(200/1000))      # (10, 200]ms

    # get mask pos - pre
    preposmask  = np.zeros((Batch_size,ts.size,N))
    preposmask[:,pre_period,:] =-1.0
    preposmask[:,pos_period,:] = 1.0

    # get stimlus input
    stiminput = np.zeros((Batch_size, duration, N)) # batch, time series, feature
    for target in range(Batch_size):
        stiminput[target,stim_onset,target] = 1

    # convert to bm.array
    struc_conn_matrix = bm.asarray(struc_conn_matrix)
    preposmask  = bm.asarray(preposmask)
    stiminput   = bm.asarray(stiminput)

    # 无法使用@bm.jit，涉及到创建model实例
    def bodyEC(G,w,I,S=0.0):
        # create instance of model
        model = DecoModel(size = N, batch_size=Batch_size, struc_conn_matrix = struc_conn_matrix, G=G, w=w, I=I, H_x_act = get_AbbottChance(method = 'custom_jvp'), S_init=S)
        runner = bp.DSRunner(model, progress_bar=False, numpy_mon_after_run=False,)
        
        # warmup stimulation first to a stable point
        runner.run(duration=2)#2s resting-state
        _   = runner.predict(inputs = stiminput) #give stimulus bacth-wise
        runner.run(duration=2)#2s resting-state

        # running stimulation
        out = runner.predict(inputs = stiminput) #give stimulus bacth-wise

        EC = bm.transpose(bm.mean(out*preposmask,axis=1)) #需要做一个转置，让列是刺激，行是接收，同ftract

        return out,EC

    return bodyEC

