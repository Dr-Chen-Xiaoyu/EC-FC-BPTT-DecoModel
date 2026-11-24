import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np  
import brainpy as bp
import os,scipy,re
import pandas as pd
import seaborn as sns
from nilearn import plotting

#%% plt setting 
def plt_setting(plt):
    plt.rcParams['font.size'] = 8
    plt.rcParams['axes.labelsize'] = 8   # 坐标轴标签字体大小
    plt.rcParams['axes.titlesize'] = 8   # 图表标题字体大小
    plt.rcParams['xtick.labelsize'] = 8  # X 轴刻度字体大小
    plt.rcParams['ytick.labelsize'] = 8  # Y 轴刻度字体大小
    plt.rcParams['legend.fontsize'] = 7  # 图例字体大小

    plt.rcParams['lines.markersize'] = 4    
    plt.rcParams['lines.linewidth'] = 0.5
    plt.rcParams['axes.linewidth'] = 0.5
    plt.rcParams['xtick.major.width'] = 0.5  # X 轴主刻度线宽度
    plt.rcParams['xtick.minor.width'] = 0.5  # X 轴次刻度线宽度
    plt.rcParams['ytick.major.width'] = 0.5  # Y 轴主刻度线宽度
    plt.rcParams['ytick.minor.width'] = 0.5  # Y 轴次刻度线宽度

    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300

    # Specify the path to your local Arial TTF file from https://github.com/gasharper/linux-fonts/blob/master/arial.ttf
    fm.fontManager.addfont('/data/xyc/codes/iEEG_network/4_model_BrainPy/3resultAnalysis/arial.ttf')
    # Set Arial as the default font
    plt.rcParams['font.family'] = 'Arial'

def style_axis(ax, grid=False):
    # Apply restrained Nature-style axis finishing without changing data.
    ax.tick_params(direction='out', length=2.5, width=0.6,)
    if hasattr(ax, 'spines'):
        for side in ('left', 'bottom'):
            if side in ax.spines:
                ax.spines[side].set_linewidth(0.6)
        for side in ('top', 'right'):
            if side in ax.spines:
                ax.spines[side].set_visible(False)
    if grid:
        ax.grid(axis='y', color='#E6E6E6', linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
    return ax

#%% data loading 

def bppaths_to_states(bppaths):
    '''
    load list or array of paths of *.bp to list of states
    '''
    states=[]
    for i in range(len(bppaths)):
        states.append(bp.checkpoints.load_pytree(bppaths[i]))
    return states

def states_to_arrays_dict(states,keys=None):
    """
    Combine specific parameters from a list of state dictionaries into a dictionary of NumPy arrays.

    Parameters
    ----------
    states : List[Dict[str, Any]]
        A list of state dictionaries. Each dictionary should contain the specified keys.
    keys : List[str]
        A list of keys to extract from each state dictionary.

    Returns
    -------
    Dict[str, np.ndarray]
        A dictionary where each key maps to a NumPy array (N of state,* shape of parameter) containing the corresponding parameter from all states.

    """
    
    if keys is None:
        keys=list(states[0].keys())

    arrays=[]

    for key in keys:
        
        example=states[0][key]

        if isinstance(example,str):
            dtype='U200'
        else:
            dtype=float

        arrays.append(np.empty((len(states),*np.shape(example)),dtype=dtype))

    for i,state in enumerate(states):
        for k,key in enumerate(keys):
            arrays[k][i]=state[key]
    
    return dict(zip(keys,arrays))

def get_ancestor_bppaths(current_bppaths,ancestor_dir,ancestor_type = 'EC'):

    ancestor_bppaths = []
    for p in current_bppaths:
        m = re.search(r'{}<([^>]+)>'.format(ancestor_type), p)
        ancestor_bppaths.append(os.path.join(ancestor_dir, m.group(1)+'.bp'))

    if len(ancestor_bppaths) == len(current_bppaths):
        print('checked same length of bppaths')
    else:
        raise ValueError("NOT same length of bppaths")

    return ancestor_bppaths

def indexing_by_names(names,selected_names,data=None):
    '''
    Extracts data slices from the provided data array based on a list of selected names.

    Parameters
    ----------
    data : numpy.ndarray
        The original data array with shape `(N, ... )`.
    names : list of str
        List of `N` strings corresponding to batch/sample axis in `data`.
    selected_names : list of str
        List of `m` strings representing the names of the data slices to retrieve.

    Returns
    -------
    - list of indices
    - numpy.ndarray: A new array containing the data slices corresponding to `selected_names` with shape `(m, ...)`.

    Raises
    ------
    ValueError
        If any of the `selected_names` are not present in the `names` list.
  
    '''

    name_to_index = {name: idx for idx, name in enumerate(names)}
    
    indices = [name_to_index[name] for name in selected_names]

    if data is None:
        return indices
    else:
        return indices, data[indices, ...]


#%% evaluation and analysis

def edge_wise_evaluation_modelled_ECs(model_ECs,real_EC,real_SC):
    # prepare
    state_N=model_ECs.shape[0]

    s=np.copy(real_SC)
    s2=np.linalg.matrix_power(s,2)

    f=np.copy(real_EC)
    f[np.diag_indices_from(f)] = np.nan # 去掉对角

    # model_ECs 结果汇总
    propagation_direct = np.zeros(state_N) # propagation effects for connections with SC
    propagation_indirect = np.zeros(state_N) # propagation effects for connections without SC
    propagation_ratio = np.zeros(state_N) # propagation ratio

    # edge-wise correlations
    r_EC_ftract = np.zeros(state_N) # pattern correlation between modelled EC and ftract EC(for edges with non NaN ftract)
    r_EC_ftract_indirect = np.zeros(state_N) #same as above for indirect edges(SC=0)

    r_EC_SC  = np.zeros(state_N) # pattern correlation between modelled EC and SC (for same edges with non NaN ftract)
    r_EC_SC2 = np.zeros(state_N) #same as above for indirect edges(SC=0)

    for n in np.arange(model_ECs.shape[0]):
        # modelled EC
        EC_ctx = model_ECs[n,:,:]

        # reponse ratio
        SC0_reponse = np.mean(np.diag(EC_ctx))
        SC1_reponse = np.mean(EC_ctx[ s!=0 ])
        SCN_reponse = np.mean(EC_ctx[(s+np.eye(s.shape[0]))==0])

        propagation_direct[n] = SC1_reponse/SC0_reponse
        propagation_indirect[n] = SCN_reponse/SC0_reponse
        propagation_ratio[n] = SCN_reponse/SC1_reponse

        # correlation between modelled EC, SC and F-tract
        ids = ~np.isnan(f) # ftract is not NaN
        r_EC_ftract[n] = float(scipy.stats.pearsonr(f[ids], EC_ctx[ids]).statistic) # Ftract EC
        r_EC_SC[n]     = float(scipy.stats.pearsonr(s[ids], EC_ctx[ids]).statistic) # SC EC

        ids = (~np.isnan(f))&(s==0) # ftract is not NaN & indirect edges
        r_EC_ftract_indirect[n] = float(scipy.stats.pearsonr(f[ids], EC_ctx[ids]).statistic) # Ftract EC
        r_EC_SC2[n] = float(scipy.stats.pearsonr(s2[ids], EC_ctx[ids]).statistic) # SC2 EC
    
    return propagation_direct,propagation_indirect,propagation_ratio,r_EC_ftract,r_EC_ftract_indirect,r_EC_SC,r_EC_SC2

def mypearsonr(x,y):
    try:
        return scipy.stats.pearsonr(x,y)
    except:
        return np.nan,np.nan
    
def edge_wise_evaluation_modelled_connectomes(xs,y,sc,rfunc=mypearsonr):
    # prepare
    state_N=xs.shape[0]

    # edge-wise correlations
    rs = np.zeros(state_N) # pattern correlation between modelled (xs) and empirical (y) conn (for non NaN edges)
    rs_direct = np.zeros(state_N) # same as above for direct edges(SC>0)
    rs_indirect = np.zeros(state_N) # same as above for indirect edges(SC=0)
    ps,ps_direct,ps_indirect = (np.zeros(state_N),np.zeros(state_N),np.zeros(state_N))

    for n in np.arange(state_N):
        
        x = xs[n,:,:] # modelled conn

        ids = (~np.isnan(x))&(~np.isnan(y))  
        r,p = rfunc(x[ids], y[ids]) 
        rs[n]=r 
        ps[n]=p 

        ids = (~np.isnan(x))&(~np.isnan(y))&(sc>0)  
        r,p = rfunc(x[ids], y[ids]) 
        rs_direct[n]=r   
        ps_direct[n]=p 

        ids = (~np.isnan(x))&(~np.isnan(y))&(sc==0)  
        r,p = rfunc(x[ids], y[ids]) 
        rs_indirect[n]=r 
        ps_indirect[n]=p 
    
    return rs,rs_direct,rs_indirect,ps,ps_direct,ps_indirect


#%% tool functions

def best_with_filter(loss,filter,type='min'):
    '''
    Select an index from the given loss array based on the specified criterion, applying a filter to ignore certain elements.

    Parameters
    ----------
    loss : array_like
        The input array representing loss values. Should be a 1D array.
    filter : array_like of bool
        A boolean mask of the same shape as `loss`. Elements corresponding to `False` values in `filter` are not considered in the selection.
    type : {'min', 'max'}, optional
        - 'min': Select the index of the minimum loss value.
        - 'max': Select the index of the maximum loss value.

    Returns
    -------
    ids : int
        The index of the selected element in `loss`. If multiple elements tie for the chosen criterion, the index of the first occurrence is returned.
    '''
    
    if type == 'min':
        select_fun = np.nanargmin
    elif type == 'max':
        select_fun = np.nanargmax

    mask=np.ones_like(loss)
    mask[~filter]=np.nan
    ids = select_fun(loss*mask)

    return ids

def filter_with_filename(bppaths,filename):
    ids = np.asarray([filename in s for s in bppaths])
    return int(ids.nonzero()[0]),ids # （the first, bool array of the fiter)


#%% plotting: training results R-values

def scatter_training_results(ax,x,y,label_name,label_array,alpha=0.9):
    labels = np.unique(label_array)
    for label in labels:
        ids = label_array == label
        ax.plot(x[ids],y[ids],'.',alpha=alpha,label="{}={}".format(label_name,label))

def hist_training_results(ax,x,label_name,label_array,bins=np.linspace(-.1,.75,85+1),alpha=0.5):
    labels = np.unique(label_array)
    for label in labels:
        ids = label_array == label
        ax.hist(x[ids],alpha=alpha,bins=bins,label="{}={}".format(label_name,label))


#%% plotting: training traces 3D or 2D

def concatenate_initials(list_epo_x,list_ini_x):
    list_out_x = []
    for epo_x,ini_x in zip(list_epo_x,list_ini_x):
        list_out_x.append(np.concatenate((ini_x[:,None],epo_x),axis=1))
    return list_out_x # list of initial-concatenated arrays

def set_vmin_vmax(c,vmin,vmax):
    min=np.nanmin(c) 
    max=np.nanmax(c)
    half= (min+max)/2

    if vmin=='min':
        vmin=min
    elif vmin=='half':
        vmin=half
    else:
        vmin=vmin

    if vmax=='max':
        vmax=max
    elif vmax=='half':
        vmax=half
    else:
        vmax=vmax

    return vmin,vmax

def plot_traces3D(
        ax,x,y,z,c,
        if_colored_trace=False,colored_trace_alpha=.5,vmin='min',vmax='half',cmap="viridis_r",if_colorbar=True,
        if_gray_trace=True,gray_trace_alpha=.05,
        ):
    '''
    Plot 3D traces with optional initial points and colored traces.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.Axes3DSubplot
        A 3D axes object from Matplotlib where the traces will be plotted.
    x : array-like, shape (N_initials, N_epochs)
        X coordinates of the traces.
    y : array-like, shape (N_initials, N_epochs)
        Y coordinates of the traces.
    z : array-like, shape (N_initials, N_epochs)
        Z coordinates of the traces.
    c : array-like, shape (N_initials, N_epochs)
        Color values for the traces, e.g., loss
    '''

    # prepare
    vmin,vmax=set_vmin_vmax(c[:,-1],vmin,vmax)

    #画trace
    if if_colored_trace: # 画出trace上每个点的颜色
        img=ax.scatter(x,y,z,c=c, alpha=colored_trace_alpha, cmap=cmap, marker='.',vmin=vmin,vmax=vmax)
    else: # 只画终点的颜色
        img=ax.scatter(x[:,-1],y[:,-1],z[:,-1],c=c[:,-1], alpha=colored_trace_alpha, cmap=cmap, marker='.',vmin=vmin,vmax=vmax)
        
    if if_gray_trace: # 灰色训练轨迹
        for k in range(x.shape[0]):
            ax.plot(x[k,:],y[k,:],z[k,:], 'gray', alpha=gray_trace_alpha,)

    #其他默认设置
    if if_colorbar:
        clb = plt.colorbar(img, fraction=0.02, pad=0.02, location='left')
    else:
        clb = None
    ax.view_init(30, -60) # ax.view_init(30, 60)
    ax.set_box_aspect(None, zoom=0.7)

    return img,clb

def plot_traces2D(
        ax,x,y,c,
        if_colored_trace=False,colored_trace_alpha=.7,vmin='min',vmax='half',cmap="viridis_r",if_colorbar=True,
        if_gray_trace=True,gray_trace_alpha=.05,gray_trace_color='gray',gray_trace_linewidth=1,
        ):
    '''
    Plot 2D traces with optional initial points and colored traces.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.Axes3DSubplot
        A 3D axes object from Matplotlib where the traces will be plotted.
    x : array-like, shape (N_initials, N_epochs)
        X coordinates of the traces.
    y : array-like, shape (N_initials, N_epochs)
        Y coordinates of the traces.
    c : array-like, shape (N_initials, N_epochs)
        Color values for the traces, e.g., loss
    '''

    # prepare
    vmin,vmax=set_vmin_vmax(c[:,-1],vmin,vmax)
    
    #画trace
    if if_colored_trace: # 画出trace上每个点的颜色
        img=ax.scatter(x,y,c=c, alpha=colored_trace_alpha, cmap=cmap, marker='.',vmin=vmin,vmax=vmax)
    else: # 或者只画终点的颜色
        img=ax.scatter(x[:,-1],y[:,-1],c=c[:,-1], alpha=colored_trace_alpha, cmap=cmap, marker='.',vmin=vmin,vmax=vmax,zorder=2)

    if if_gray_trace: # 灰色训练轨迹
        for k in range(x.shape[0]):
            ax.plot(x[k,:],y[k,:], gray_trace_color, alpha=gray_trace_alpha, zorder=1, linewidth=gray_trace_linewidth)


    #其他默认设置
    if if_colorbar:
        clb = plt.colorbar(img,) # fraction=0.046, pad=0.02
    else:
        clb = None
    return img,clb

def plot_dashed_lines(ax,x,y,lin_x=None,lin_y=None,style='k--',):
    ax.plot(x,y,style,linewidth=0.5,zorder=-1)
    ax.set_xlim(lin_x);ax.set_ylim(lin_y)

def set_clb_same_height(ax,clb, width = 0.03, gap = 0.01):
    # same hight as the plot
    pos = ax.get_position()
    cbar_x = pos.x1 + gap  # small gap between main plot and colorbar
    cbar_y = pos.y0
    cbar_height = pos.height
    clb.ax.set_position([cbar_x, cbar_y, width, cbar_height])


#%% plotting: violin of R-values between paramters and Hierarchy

def plot_violin_corr_Parameter_Hierarchy(ax,rs,parameterID,hierarchyID_plt,legend=False,asterisk_gap=.2,asterisk_loc_xs=[-.29, -.02, .25]):

    columns=['spearman ρ-value','Hierarchy','Parameters']
    df = pd.DataFrame(columns=columns)
    for i in range(len(hierarchyID_plt)):
        df=pd.concat([df,pd.DataFrame({columns[0]:rs[:,i],columns[1]:hierarchyID_plt[i],columns[2]:parameterID})])

    sns.violinplot(ax=ax,x=columns[2], y=columns[0] ,hue=columns[1], data=df, legend=legend, widths=.01,)

    for i in range(len(hierarchyID_plt)):
        
        r = rs[:,i]

        tvalue,pvalue=scipy.stats.ttest_1samp(r,0)
        print("One-sample T-test of rous of ({},{}), T={}, p={} ".format(parameterID,hierarchyID_plt[i],tvalue, pvalue))

        if pvalue < .005 :
            sig = '*'
        else:
            sig = 'n.s.'

        if tvalue>0:
            loc_y = r.max() + asterisk_gap
        else:
            loc_y = r.min() - asterisk_gap

        ax.text(asterisk_loc_xs[i], loc_y, sig)


#%% plotting: connectome

def plot_conn_matrix(ax,x,ctx_labels=None,vmin=None,vmax=None,cmap='viridis',if_yticks=True):
        
    plotting.plot_matrix(x, axes=ax, labels=ctx_labels, cmap=cmap,vmin=vmin,vmax=vmax, colorbar=True,)

    ax.axes.set_xticks([])    
    
    if ctx_labels is None:
        ax.axes.set_yticks(np.arange(x.shape[0]))
    ax.tick_params(axis='y', labelsize=5,)

    if not if_yticks:
        ax.axes.set_yticks([])

def plot_conn_comparison_scatter(ax,x,y,s=None,mask=None,text_xy1=(0,0),text_xy2=(0,0),text_xy3=(0,0),cmap=plt.colormaps['tab10'],alphadot=.5,markersize=5,alphaline=1,):
    
    if mask is None:
        mask=~np.isnan(x*y)
    x_=x[mask]
    y_=y[mask]
    

    if s is None:
        # just all
        x=x_
        y=y_
        c=cmap(0)
        sns.regplot(ax=ax,x=x,y=y,scatter=True,scatter_kws={'marker':'.', 's':markersize,'color':c,'alpha':alphadot},line_kws={"color": c,'alpha':alphaline})
        ax.text(text_xy1[0],text_xy1[1],"r = {:.4f}\np = {:.1e}".format(*scipy.stats.pearsonr(x, y)),color=c)

    else:
        s_=s[mask]

        # all 
        x=x_
        y=y_
        c=cmap(7)
        sns.regplot(ax=ax,x=x,y=y,scatter=False,line_kws={"color": c,'alpha':alphaline})
        ax.text(text_xy1[0],text_xy1[1],"r = {:.4f}\np = {:.1e}".format(*scipy.stats.pearsonr(x, y)))

        # indirect
        x=x_[s_==0]
        y=y_[s_==0]
        c=cmap(0)
        sns.regplot(ax=ax,x=x,y=y,label='SC=0',scatter_kws={'marker':'.', 's':markersize,'color':c,'alpha':alphadot},line_kws={"color":c,'alpha':alphaline})
        ax.text(text_xy2[0],text_xy2[1],"r = {:.4f}\np = {:.1e}".format(*scipy.stats.pearsonr(x, y)),color=c)

        # direct
        x=x_[s_!=0]
        y=y_[s_!=0]
        c=cmap(1)
        sns.regplot(ax=ax,x=x,y=y,label='SC>0',scatter_kws={'marker':'.', 's':markersize,"color":c,'alpha':alphadot},line_kws={"color":c,'alpha':alphaline})
        ax.text(text_xy3[0],text_xy3[1],"r = {:.4f}\np = {:.1e}".format(*scipy.stats.pearsonr(x, y)),color=c)

        ax.legend()


#%% plotting: histogram of correlations between the paramter and ENIGMA genes

def plot_hist_genes(ax,hist_r,hist_label,line_r,line_labels,xlabel):

    ax.hist(hist_r, bins=100, label=hist_label,color='#D9EEEE')
    ax.set_xlabel(xlabel)
    ax.set_ylim(ax.get_ylim())
    for i, ID in enumerate(line_labels):
        R=line_r[i]

        if R>0:
            prob = (R<hist_r).sum()/hist_r.size
        else:
            prob = (R>hist_r).sum()/hist_r.size
        if prob < 0.05:
            if_bold = '*'
        else:
            if_bold = ' '
        ax.plot(np.ones(2)*R, ax.get_ylim(), label="{}[p={:0.3f}]{}".format(if_bold,prob,ID))
    ax.legend()
    ax.locator_params(nbins=20, axis='x')

def table_significant_genes(rs,ps,r_name,genesID):
    sig = ps<(0.05/ps.size) # Bonferroni correction
    df = pd.DataFrame(columns=['Genes', r_name, 'p-value'])
    for a,b,c in zip(genesID[sig],rs[sig],ps[sig]): 
        df.loc[len(df.index)] = [a,b,c]  
    df.style.set_caption('Genes with Bonferroni corrected p < 0.05')
    return df


#%% plotting: time course
def baseline_correction(ts,zs,correction=None):
    if correction is None:
        zlabel = 'S'
    elif correction == 'minus':
        zs = zs - np.mean(zs[ts<0,:],axis=0)
        zlabel = 'baseline-corrected S'
    elif correction == 'percent':
        zs = 100*(zs - np.mean(zs[ts<0,:],axis=0))/np.mean(zs[ts<0,:],axis=0)
        zlabel = 'percent of change (%)'
    return zs,zlabel


def plot_time_course_3D(ax,ts,zs,correction=None):

    zs,zlabel = baseline_correction(ts,zs,correction=correction)
        
    node_N = zs.shape[1]
    for i in range(node_N):
        y=i*np.ones_like(ts)
        x=ts
        z=zs[:,i]
        ax.plot(x, y, z, zorder=node_N-i)
    
    ax.set_xlabel('Time(s)')
    ax.set_yticks([])
    ax.set_ylabel('Node')
    ax.set_zlabel(zlabel)