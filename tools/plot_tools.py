import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#####################################################
### FIG AND AXES DEF
#####################################################


def get_axes(L, max_col=3, fig_frame=(5,4), res=100):
    cols = L if L <= max_col else max_col
    rows = int(L / max_col) + int(L % max_col != 0)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * fig_frame[0], rows * fig_frame[1]), dpi=res)
    if L > 1:
    	axes =  axes.flatten()
    for s in range(L,max_col*rows):
        for side in ['bottom','right','top','left']:
            axes[s].spines[side].set_visible(False)
        axes[s].set_yticks([])
        axes[s].set_xticks([])
        axes[s].xaxis.set_ticks_position('none')
        axes[s].yaxis.set_ticks_position('none')

    return fig, axes

def single_scatter_fig(data, color='tab:blue', fig_frame=(5,4), res=100, s=10, alpha=1.0, priority=1, chunk=None):
    fig, axes = plt.subplots(1, 1, figsize=(1 * fig_frame[0], 1 * fig_frame[1]), dpi=res)
    if chunk:
    	axes.scatter(data[:chunk,0], data[:chunk,1], 
    				s=s, marker="o", alpha=alpha, c=color, edgecolor='k', linewidth=0.5, zorder=priority)
    else:
    	axes.scatter(data[:,0], data[:,1], 
    				 s=s, marker="o", alpha=alpha, c=color, edgecolor='k', linewidth=0.5, zorder=priority)
    for side in ['bottom','left']:
    	axes.spines[side].set_linewidth(2)
    for side in ['right','top']:
    	axes.spines[side].set_visible(False)
    return fig, axes


def single_scatter_axe(data, axes, color='tab:blue', s=10, alpha=1.0, priority=1, chunk=None):
    if chunk:
    	axes.scatter(data[:chunk,0], data[:chunk,1], 
    				s=s, marker="o", alpha=alpha, c=color, edgecolor='k', linewidth=0.5, zorder=priority)
    else:
    	axes.scatter(data[:,0], data[:,1], 
    				 s=s, marker="o", alpha=alpha, c=color, edgecolor='k', linewidth=0.5, zorder=priority)
    for side in ['bottom','left']:
    	axes.spines[side].set_linewidth(2)
    for side in ['right','top']:
    	axes.spines[side].set_visible(False)
    return axes


####################################################
### COLORS
#####################################################


def myPalette(n_colors, palette='tab10'):
	return sns.color_palette(palette, n_colors)


def make_colors(clust, mode='tab10'):
    if np.min(clust) == -1:
        N = np.unique(clust).shape[0] - 1
        colors = myPalette(N, mode) + [(0,0,0)]
    else:
        N = np.unique(clust).shape[0]
        colors = myPalette(N, mode)
    return colors


#####################################################
### ARROWED AXES
#####################################################


def axarrows(fig,ax,labels=None):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    for side in ['bottom','right','top','left']:
        ax.spines[side].set_visible(False)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    # get width and height of axes object to compute
    # matching arrowhead length and width
    dps = fig.dpi_scale_trans.inverted()
    bbox = ax.get_window_extent().transformed(dps)
    width, height = bbox.width, bbox.height

    # manual arrowhead width and length
    hw = 1./20.*(ymax-ymin)
    hl = 1./20.*(xmax-xmin)
    lw = 1. # axis line width
    ohg = 0.3 # arrow overhang

    # compute matching arrowhead length and width
    yhw = hw/(ymax-ymin)*(xmax-xmin)* height/width
    yhl = hl/(xmax-xmin)*(ymax-ymin)* width/height

    # draw x and y axis
    ax.arrow(xmin, ymin, xmax-xmin, 0., fc='k', ec='k', lw = lw,
             head_width=hw, head_length=hl, overhang = ohg,
             length_includes_head= True, clip_on = False)
    ax.arrow(xmin, ymin, 0., ymax-ymin, fc='k', ec='k', lw = lw,
             head_width=yhw, head_length=yhl, overhang = ohg,
             length_includes_head= True, clip_on = False)
    if labels:
    	ax.set_xlabel(r''+labels[0],size=18)
    	ax.set_ylabel(r''+labels[1],size=18)
    
    return ax


#####################################################
### HISTOGRAMS
#####################################################


def plot_donut(data, labels, palette, axes, size=0.3, radius=1, annotate=1):
	circle = plt.Circle((0,0),0.60,fc='white')
	wedges, texts = axes.pie(data, startangle=90, radius=radius,
		                     colors=palette, 
		                     wedgeprops=dict(width=size, edgecolor='k'))
	axes.add_patch(circle)

	if annotate == 1:
		bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
		kw = dict(arrowprops=dict(arrowstyle="-"), 
			      bbox=bbox_props, zorder=0, va="center")

		for i, p in enumerate(wedges):
			ang = (p.theta2 - p.theta1)/2. + p.theta1
			y = np.sin(np.deg2rad(ang))
			x = np.cos(np.deg2rad(ang))
			horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
			connectionstyle = "angle,angleA=0,angleB={}".format(ang)
			kw["arrowprops"].update({"connectionstyle": connectionstyle})
			axes.annotate(labels[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                		horizontalalignment=horizontalalignment, **kw)
	elif annotate == 0:
		axes.legend(wedges, labels, loc="center left",
					bbox_to_anchor=(1, 0, 0.5, 1), prop=dict(size=15))

	return axes


#####################################################
### MATRICES
#####################################################


def plot_matrix_nolabels(dataMatrix, axes, palette_name="viridis", s=16,lw=1):
    sns.heatmap(dataMatrix, annot=True, fmt=".2f", 
                cbar=False, ax=axes, cmap=sns.color_palette(palette_name),
                annot_kws={"fontsize":s, "fontweight":'bold'}, linewidths=lw, linecolor='w')
    axes.xaxis.tick_top()
    axes.tick_params(labeltop=False,labelleft=False)
    
    return axes


def plot_matrix(dataMatrix, labels, axes, palette_name="viridis", s=45,lw=3,lab_s=24):
	sns.heatmap(dataMatrix, annot=True, fmt=".2f", 
                cbar=False, ax=axes, cmap=sns.color_palette(palette_name), as_cmap=True,
                annot_kws={"fontsize":s, "fontweight":'bold'}, linewidths=lw, linecolor='k')
	axes.xaxis.tick_top()
	axes.set_xticklabels(labels, size=lab_s)
	for tick in axes.get_xticklabels():
		tick.set_rotation(45)
	axes.set_yticklabels(labels, size=lab_s)
	axes.tick_params(labeltop=None,labelleft=None,
				   labelsize=lab_s,width=3,size=7)
	for axis in ['top','bottom','left','right']:
		axes.spines[axis].set_linewidth(lw+0.5)

	return axes


#####################################################
### PES & FES
#####################################################


def convtofes1D(X,bins,kb='kcal',temp=300,interval=None):
    kb_ = {
        'kJ'   : 0.00831446261815324,
        'kcal' : 0.00198720425864083,
        'unit' : 1.0
    }
    KBT = kb_[kb] * temp
    hist, edges = np.histogram(X,bins=bins,range=interval,density=True)
    FES =  -1 * KBT * np.log(hist)
    FES = FES - np.min(FES)
    return FES, edges


def convtofes2D(X,Y,bins,kb='kcal',temp=300,fill_empty=True,interval=None):
    kb_ = {
        'kJ'   : 0.00831446261815324,
        'kcal' : 0.00198720425864083,
        'unit' : 1.0
    }
    KBT = kb_[kb] * temp
    H, xedges, yedges = np.histogram2d(X,Y,bins=bins,range=interval,density=True)
    FES =  -1 * KBT * np.log(H)
    FES = FES - np.min(FES)
    if fill_empty == True:
        max_ = np.max(FES[FES != np.inf])
        FES[FES == np.inf] = max_
    return FES, xedges, yedges


def plot_fes(x,y,bins,levels,axes,kbt='kcal',temp=300,fontsize=12,fill_empty=True,range_fes=None):
    X = np.asarray(x)
    Y = np.asarray(y)

    if range_fes:
        FES, _, _ = convtofes2D(X.flatten(),Y.flatten(),bins,
                          kb=kbt,temp=temp,
                          fill_empty=fill_empty,interval=range_fes)

        Xmin = range_fes[0][0]
        Xmax = range_fes[0][1]
        Ymin = range_fes[1][0]
        Ymax = range_fes[1][1]
    
    if not range_fes:
        FES, _, _ = convtofes2D(X.flatten(),Y.flatten(),bins,
                                kb=kbt,temp=temp,
                                fill_empty=fill_empty,interval=None)

        Xmin = X.min()
        Xmax = X.max()
        Ymin = Y.min()
        Ymax = Y.max()
        
    xx = np.arange(Xmin, Xmax, ((Xmax-Xmin)/bins))
    yy = np.arange(Ymin, Ymax, ((Ymax-Ymin)/bins))
    XX, YY = np.meshgrid(xx, yy)

    axes.set_axis_off()
    cfset = axes.contourf(XX, YY, FES.T, cmap='coolwarm_r')
    m = axes.imshow(np.rot90(FES), cmap='coolwarm_r', extent=[Xmin+0.5, Xmax-0.5, Ymin+0.5, Ymax-0.5], 
                    interpolation='gaussian', alpha=1, aspect='auto')
    cset = axes.contour(XX, YY, FES.T, np.arange(0,5,0.5), colors='black', linewidths=[0.5], alpha=0.6)
#     axes.clabel(cset, inline=1, fontsize=12, fmt = '%2.1f')
    cbar = plt.colorbar(m, shrink=0.5, aspect=10, orientation='vertical', ticks=LEVELS, ax=axes)
    cbar.set_label('FES [kcal / mol]')
    cbar.axes.tick_params(labelsize=fontsize)
    for axis in ['top', 'bottom', 'left', 'right']:
        axes.spines[axis].set_linewidth(1)
    return axes


def plot_ghost_fes(x,y,bins,levels,axes,kbt='kcal',temp=300,fontsize=12,fill_empty=True,range_fes=None):
    X = np.asarray(x)
    Y = np.asarray(y)

    if range_fes:
        FES, _, _ = convtofes2D(X.flatten(),Y.flatten(),bins,
                          kb=kbt,temp=temp,
                          fill_empty=fill_empty,interval=range_fes)

        Xmin = range_fes[0][0]
        Xmax = range_fes[0][1]
        Ymin = range_fes[1][0]
        Ymax = range_fes[1][1]
    
    if not range_fes:
        FES, _, _ = convtofes2D(X.flatten(),Y.flatten(),bins,
                                kb=kbt,temp=temp,
                                fill_empty=fill_empty,interval=None)

        Xmin = X.min()
        Xmax = X.max()
        Ymin = Y.min()
        Ymax = Y.max()
        
    xx = np.arange(Xmin, Xmax, ((Xmax-Xmin)/bins))
    yy = np.arange(Ymin, Ymax, ((Ymax-Ymin)/bins))
    XX, YY = np.meshgrid(xx, yy)

    cset = axes.contour(XX, YY, FES.T, levels, colors='black', linewidths=[0.7], alpha=0.7)
    return axes
