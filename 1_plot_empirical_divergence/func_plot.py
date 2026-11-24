import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np  
import os,scipy,re
import pandas as pd
import seaborn as sns
from nilearn import plotting

#%% plt setting 
def plt_setting(plt):
    plt.rcParams['font.size'] = 6
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

    plt.rcParams['figure.dpi'] = 200
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


#%%
def min_max_rescale(x):
    xmin=np.nanmin(x)
    xmax=np.nanmax(x)
    return (x-xmin)/(xmax-xmin)


def plot_scatter(ax,x,y,text_xy=None,mask=None, label='',markersize=6,alphaline=1,cmap=plt.colormaps['tab10']):
    if mask is None:
        mask=(~np.isnan(x))&(~np.isnan(y))
    x=np.copy(x[mask])
    y=np.copy(y[mask])
    sns.regplot(ax=ax,x=x,y=y,label=label,scatter_kws={'s':markersize,'linewidth':0},line_kws={"color": cmap(7),'alpha':alphaline})
    
    r,p=scipy.stats.pearsonr(x, y)
    if p>0.05:
        text="r = {:.3f}\np = {:.3f} (n.s.)".format(r,p)
    elif p<0.001:
        text="r = {:.3f}\np = {:.1e}".format(r,p)
    else:
        text="r = {:.3f}\np = {:.3f}".format(r,p)
    if text_xy is None:
        text_xy0=1.1*x.min()
        text_xy1=0.8*y.max()
        ax.text(text_xy0,text_xy1,text)
    else:
        ax.text(text_xy[0],text_xy[1],text)


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

def node_strength(x, y_with_NaN=None):

    if y_with_NaN is None:
        mask = np.ones_like(x)
    else:
        mask = np.where(np.isnan(y_with_NaN), np.nan, 1) # 一般（N, N)即可

    x = x*mask # (N, N) or (batch, N, N) or (batch1,...batch2, N, N), mask is broadcasted

    NS_eff = np.nanmean(x,axis=-2) #按行计算求和，得到列的性质（最后第二维）：EC上刺激某节点，其他节点的平均响应（出度）efferent
    n_eff  = np.sum(~np.isnan(x),axis=-2) # sample 数量
    NS_aff = np.nanmean(x,axis=-1) #按列计算求和，得到行的性质（最后第一维）：EC某节点的在刺激所有其他节点时的平均响应（入度）afferent
    n_aff  = np.sum(~np.isnan(x),axis=-1) # sample 数量
 
    return NS_eff,NS_aff,n_eff,n_aff