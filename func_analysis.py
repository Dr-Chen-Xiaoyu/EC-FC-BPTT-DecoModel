import numpy as np  
# import brainpy as bp
# import brainpy.math as bm
# import jax
# import jax.numpy as jnp
from scipy.stats import pearsonr, spearmanr, rankdata

#%% tool function

def rescale_EC(x,y_with_NaN=None,if_mask_diag=True,epsilon=0):
    '''
    rescale EC to [0,1] with mask (y, nan elements) and diagonal, also in batch-wise operation

    Parameters
    ----------
    x : ndarray
        The EC matrix or batch of matrices to be rescaled. This can be a (N,N) or (batch,N,N) array.
    y_with_NaN : ndarray, optional
        - An array (typically of shape `(N, N)` or same as `x.shape`) with NaN values to be masked in  `x`
        - None, no mask is applied
    if_mask_diag : bool, optional
        - True, the diagonal elements of `x` are also masked out (set to NaN).
        - False, no mask is applied

    Returns
    -------
    ndarray
        A copy of `x` with each EC rescaled to the range [0, 1], and NaN values by the masks.

    Notes
    -----
    - The rescaling is done by dividing all non-NaN elements by the maximum non-NaN value found in `x` after masking.
    - If `x` is multi-dimensional (e.g. shape `(M, N, N)`), the rescaling is performed independently for each `(N, N)` slice along the outer dimensions.

    '''
    
    if y_with_NaN is None:
        mask = np.ones_like(x)
    else:
        mask = np.where(np.isnan(y_with_NaN), np.nan, 1) # y_with_NaN 一般（N, N)即可

    if if_mask_diag:
        N = x.shape[-1]
        mask_diag = np.ones((N,N))
        np.fill_diagonal(mask_diag, np.nan)
        mask = mask*mask_diag

    x = x*mask

    x = x/(np.nanmax(x,axis=(-1,-2))[...,None,None] + epsilon)  # if x.ndim == 2: x = x/np.nanmax(x)

    return x


#%% node strength

def node_strength(x, y_with_NaN=None):
    '''
    Calculate node strength (both efferent and afferent) based on the input matrix, defined as the mean of edges from the node.

    Parameters
    ----------
    x : numpy.ndarray
        A 2D, 3D or ND numpy array representing the connectivity matrix.
        - If 2D, shape should be (N, N), where N is the number of nodes.
        - If 3D, shape should be (batch, N, N), where batch is the number of batches and N is the number of nodes.
        - (batch1, batch2, ..., N, N)

    y_with_NaN : numpy.ndarray, optional
        A 2D or ND numpy array used to create a mask where NaN values are ignored in node strength calculation.
        - If None, no masking is applied (default is None).
    
    Returns
    -------
    NS_eff, NS_aff : numpy.ndarray
        A 1D or 2D array containing the efferent node strength. 
        - If the input `x` is 2D, the result will be a 1D array of length N.
        - If the input `x` is 3D, the result will be a 2D array with shape (batch, N).
    
    Notes
    -----
    - Efferent node strength (NS_eff) is calculated as the mean of rows across columns (axis=-2), which represents the average response of other nodes when a given node is stimulated.
    - Afferent node strength (NS_aff) is calculated as the mean of columns across rows (axis=-1), representing the average response of this node to other nodes.
    '''

    if y_with_NaN is None:
        mask = np.ones_like(x)
    else:
        mask = np.where(np.isnan(y_with_NaN), np.nan, 1) # 一般（N, N)即可

    x = x*mask # (N, N) or (batch, N, N) or (batch1,...batch2, N, N), mask is broadcasted

    NS_eff = np.nanmean(x,axis=-2) #按行计算求和，得到列的性质（最后第二维）：EC上刺激某节点，其他节点的平均响应（出度）efferent
    NS_aff = np.nanmean(x,axis=-1) #按列计算求和，得到行的性质（最后第一维）：EC某节点的在刺激所有其他节点时的平均响应（入度）afferent

    # old fashioned code，精简整合如上
    # if x.ndim == 2 : # (N, N)
    #     NS_eff=np.nanmean(x,axis=0) #按行计算求和，得到列的性质：EC上刺激某节点，其他节点的平均响应（出度）efferent
    #     NS_aff=np.nanmean(x,axis=1) #按列计算求和，得到行的性质：EC某节点的在刺激所有其他节点时的平均响应（入度）afferent
    # elif x.ndim == 3 : # (batch, N, N)
    #     NS_eff=np.nanmean(x,axis=1) #按行计算求和，得到列的性质：EC上刺激某节点，其他节点的平均响应（出度）efferent
    #     NS_aff=np.nanmean(x,axis=2) #按列计算求和，得到行的性质：EC某节点的在刺激所有其他节点时的平均响应（入度）afferent

    return NS_eff,NS_aff


#%% correlation functions : 

def maskNaN(x,y=None,mask=None):
    '''
    mask掉x和y中的NaN元素或外部指定元素
    如果x，y为二维array，其形状为(sample axis, feature axis) ，则对feature维度进行mask并保证输出向量无NaN元素
    '''
    if y is None:
        y=x

    if x.ndim == 1 :

        if mask is None:
            mask=(~np.isnan(x))&(~np.isnan(y))

        mx=x[mask]
        my=y[mask]

    elif x.ndim == 2 : # (sample axis, feature axis)

        if mask is None:
            mask=(~np.isnan(x.sum(axis=0)))&(~np.isnan(y.sum(axis=0))) # mask NaN for feature axis, so sum along sample axis
        
        mx=x[:,mask]
        my=y[:,mask]

    return mask,mx,my


def corr(x,y, mask=None, style='2vectors', corrtype='pearson',implement='scipy',rankdata=rankdata,):
    """
    Compute correlation between two arrays with options for handling masked NaN values, different correlation types, and implementation methods.

    Parameters:
    ----------
    x,y : np.ndarray
        - two 1D-arrays with shape (feature axis,) 
        - two 2D-arrays with shape (sample axis, feature axis)

    mask : np.ndarray or None, 
        - None for masking NaN elements in x and y
        - A boolean array to mask elements in x and y. True values are masked.
    
    style : str, computation style of correlation 
        - '2vectors': Compute correlation between two 1D vectors.
        - 'pairwise': Compute correlations between pairwise samples of x and y, i.e., matlab's corr style
        - 'axiswise': Compute correlations along the sample axis of x and y. x and y should have same shape
    
    corrtype : str, type of correlation to compute 
        - 'pearson': Pearson correlation coefficient.
        - 'spearman': Spearman rank-order correlation coefficient.
    
    implement : str, implementation method for correlation
        - 'scipy': Use scipy.stats functions
        - 'array': Use array fashioned operations, works for numpy and jax
    
    rankdata : Callable, function to rank x and y along feature axis, used when implement is 'array' and corrtype is 'spearman'.

    Returns:
    -------
    (r-values, p-values) or (r-values,):
        - implement='scipy', returns a tuple of correlation coefficients (array) and p-values (array)
        - implement='array', returns correlation coefficients (array)
    """

    _,x,y=maskNaN(x,y,mask=mask) # preprocessing

    if implement=='scipy': # scipy.stats based correlation functions

        if corrtype == 'pearson':
            if style=='2vectors':
                r,p=pearsonr(x,y)
            elif style=='pairwise':
                r,p=pearsonr(x[:, np.newaxis, :], y, axis=-1)
            elif style=='axiswise':
                r,p=pearsonr(x, y, axis=1)

        elif corrtype == 'spearman':
            if style=='2vectors':
                r,p=spearmanr(x,y)
            elif style=='pairwise':
                r,p=spearmanr(x, y, axis=1) # in corrcoef style, so slice the part we want
                r=r[0:x.shape[0],-y.shape[0]:]
                p=p[0:x.shape[0],-y.shape[0]:]
            elif style=='axiswise':
                r=np.zeros((x.shape[0]))
                p=np.zeros((x.shape[0]))
                for i in range(r.size): # 用for反而更快一些，节约了一些计算量
                    a,b=spearmanr(x[i,:], y[i,:])
                    r[i]=a
                    p[i]=b
    
        return r,p
    
    elif implement=='array': # array operation based correlation functions

        if corrtype == 'spearman':
            if style=='2vectors':
                x = rankdata(x)
                y = rankdata(y)
            else:
                x = rankdata(x,axis=1)
                y = rankdata(y,axis=1)

        if style=='2vectors':
            x = (x-x.mean())/x.std()
            y = (y-y.mean())/y.std()
            N = x.size
            r = x.dot(y)/N
        elif style=='pairwise':
            x = (x-x.mean(axis=1)[:,None])/(x.std(axis=1)[:,None]) # normliza along feature axis
            y = (y-y.mean(axis=1)[:,None])/(y.std(axis=1)[:,None])
            N = x.shape[1] # N of features
            r = x @ y.T / N
        elif style=='axiswise':
            x = (x-x.mean(axis=1)[:,None])/(x.std(axis=1)[:,None]) # normliza along feature axis
            y = (y-y.mean(axis=1)[:,None])/(y.std(axis=1)[:,None])
            r = (x*y).mean(axis=1)
        
        return r


#%%  old correlation functions:
# matlab style: A(samplesA,features) B(samplesB,features) -> r(samplesA,samplesB)
def matlab_corr(x,y,mask=None,corrtype='pearson',rankdata=rankdata): 
    if mask is None: 
        mask,A,B=maskNaN(x,y)
    else:
        A=x[:,mask]
        B=y[:,mask]

    if corrtype == 'spearman':
        A = rankdata(A,axis=1)
        B = rankdata(B,axis=1)

    A=(A-A.mean(axis=1)[:,None])/(A.std(axis=1)[:,None])
    B=(B-B.mean(axis=1)[:,None])/(B.std(axis=1)[:,None])
    sampleN=A.shape[1]

    return A@B.T/sampleN

# axis-wise correlations between two 2D-array: A(samples,features) B(samples,features) -> r(samples,) or  A(samples,features) B(samples,features) -> r(samples,)
def axis_corr(x,y,mask=None,corrtype='pearson',rankdata=rankdata): 
    Nsample = x.shape[0]
    if y.ndim == 1:
        y = np.repeat(y[None,:], repeats=Nsample, axis=0)
    
    if mask is None: 
        mask,A,B=maskNaN(x,y)
    else:
        A=x[:,mask]
        B=y[:,mask]

    if corrtype == 'spearman':
        A = rankdata(A,axis=1)
        B = rankdata(B,axis=1)
        
    A=(A-A.mean(axis=1)[:,None])/(A.std(axis=1)[:,None])
    B=(B-B.mean(axis=1)[:,None])/(B.std(axis=1)[:,None])

    return np.mean(A*B,axis=1)

# axis-wise style correlations between two matrix or matrix and a array(for-loop style)
# A(samples,features) B(samples,features) -> r(samples,) or  A(samples,features) B(samples,features) -> r(samples,)
def axis_corr_forloop(x,y,mask=None,corrtype='pearson'): 
    if corrtype == 'spearman':
        rfunc=spearmanr
    elif corrtype == 'pearson':
        rfunc=pearsonr

    Nsample = x.shape[0]
    if y.ndim == 1:
        y = np.repeat(y[None,:], repeats=Nsample, axis=0)

    if mask is None: 
        mask,A,B=maskNaN(x,y)
    else:
        A=x[:,mask]
        B=y[:,mask]

    rs=np.zeros(Nsample)
    ps=np.zeros(Nsample)
    for i in range(Nsample):
        r,p=rfunc(A[i,:] ,B[i,:] )#calculate correlation
        rs[i]=r
        ps[i]=p
    return rs,ps

# correlation between nodes' edges along time: (time,node,node)+(time,node,node)->(time,node) or (time,node,node)+(node,node)->(time,node) 
def edge_wise_corr_of_each_node(x,y,node_axis='row',corrtype='pearson'): 
    times = x.shape[0]
    nodes = x.shape[1]
    rs=np.zeros((times,nodes))

    if y.ndim == 2:
        y = np.repeat(y[None,:,:], repeats=times, axis=0)

    for i in range(nodes):
        if node_axis == 'row':
            a=x[:,i,:]
            b=y[:,i,:]
        elif node_axis == 'col':
            a=x[:,:,i]
            b=y[:,:,i]
        
        rs[:,i] = axis_corr(a,b,corrtype=corrtype)
    
    return rs


