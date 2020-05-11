from matplotlib import pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)
from matplotlib.ticker import NullFormatter
from matplotlib.image import NonUniformImage
import matplotlib.pyplot as plt
import numpy as np
import corner
import Chain
#mcmc_result(cha.samples)
#trace_plot(cha.samples)
#omega_m, omega_lambda, prob = samples_process(samples=cha.samples, x_range=[0, 1.6], y_range = [0, 2.5], xbin=30, ybin=40)
#fig18(omega_m, omega_lambda, prob_nosys=prob, prob_sys=[])
def samples_process(samples, x_range=[0, 1.6], y_range = [0, 2.5], xbin=30, ybin=40):
    _omega_m = []
    _omega_L = []
    for _pars in samples:
        _omega_m.append(_pars.get('Omega_m'))
        _omega_L.append(_pars.get('Omega_lambda'))
    omega_m = np.linspace(x_range[0],x_range[1],xbin)
    omega_lambda = np.linspace(y_range[0],y_range[1],ybin)
    prob, omega_m, omega_lambda = np.histogram2d(_omega_m, _omega_L, bins=(omega_m, omega_lambda))
    omega_m = omega_m[:-1]
    omega_lambda = omega_lambda[:-1]
    prob = prob/np.sum(prob)
    return omega_m, omega_lambda, prob

def fig18(omega_m=[], omega_lambda=[], prob_sys=[], prob_nosys=[]):
    params = {'legend.fontsize': 16,
              'figure.figsize': (15, 5),
             'axes.labelsize': 17,
             'axes.titlesize':20,
             'xtick.labelsize':10,
             'ytick.labelsize':10,
             'ytick.major.size': 5.5,
             'axes.linewidth': 2}
    plt.rcParams.update(params)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    fig, axs = plt.subplots(1, 1, figsize=(10,10))
    
    ax = axs
    #c = ax.pcolor(omega_m, omega_lambda, np.transpose(prob), cmap='RdBu', 
    #              vmin=prob.min(), vmax=prob.max(),edgecolors='k', linewidths=0.1)
    if len(prob_sys) != 0:
        _chi2_sys = -2*np.log(prob_sys)
        ax.contourf(omega_m, omega_lambda, np.transpose(prob_sys), 
                    levels=[np.exp((6.17+_chi2_sys.min())/(-2)),np.exp((2.3+_chi2_sys.min())/(-2)),1],
                    origin='lower', 
                    colors=[(0.80078125, 0.328125 , 0.34765625), (0.76171875, 0.1640625 , 0.18359375)], 
                    alpha=0.9, linewidths=2)
        ax.text(0.5,1.1,"Pantheon", color='red',rotation=40,alpha=0.7, size=15)
    if len(prob_nosys) != 0:
        _chi2_nosys = -2*np.log(prob_nosys)
        ax.contourf(omega_m, omega_lambda, np.transpose(prob_nosys), 
                    levels=[np.exp((6.17+_chi2_nosys.min())/(-2)),np.exp((2.3+_chi2_nosys.min())/(-2)),1],
                    origin='lower', 
                    colors=[(0.5,0.5,0.5), (0.4,0.4,0.4)],
                    alpha=0.9, linewidths=2)
        ax.text(0.21,1.1,"Pantheon (Stat)", color='gray',rotation=40,alpha=0.7, size=15)
    ax.set_title(r'$o \rm{CDM}$'+' Constraints For SN-only Sample')
    ax.set_xlabel(r"$\Omega_m$", size=25)
    ax.set_ylabel(r"$\Omega_{\Lambda}$", size=25)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 17, length = 8, width = 2)
    ax.tick_params(axis = 'both', which = 'minor', labelsize = 12, length = 4, width = 1)
    
    
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02
    rect_ax2 = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]
    
    ax2 = plt.axes(rect_ax2)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)
    # Manually set the position and relative size of the inset axes within ax1
    ax2.set_axes_locator(InsetPosition(ax, [left+0.4, bottom+0.4, width/2, height/2]))
    axHistx.set_axes_locator(InsetPosition(ax, [left+0.4, bottom_h+0.08, width/2, 0.2/2]))
    axHisty.set_axes_locator(InsetPosition(ax, [left_h+0.08, bottom+0.4, 0.2/2, height/2]))
    for ax in [ax2, axHistx, axHisty]:
        #ax.label_outer()
        ax.minorticks_on()
        ax.tick_params(axis = 'both', which = 'major', labelsize = 17, length = 8, width = 2)
        ax.tick_params(axis = 'both', which = 'minor', labelsize = 12, length = 4, width = 1)
        ax.grid(True)
        
    # no labels
    nullfmt = NullFormatter()
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)
    # darw the histogram
    binwidth = 0.05
    xbins = np.arange(0, 0.5 + binwidth, binwidth)
    binwidth = 0.1
    ybins = np.arange(0, 0.5 + binwidth, binwidth)
    if len(prob_sys) != 0:
        axHistx.plot(omega_m, np.sum(np.transpose(prob_sys), axis=0),
                     color = (0.76171875, 0.1640625 , 0.18359375))
        axHisty.plot(np.sum(np.transpose(prob_sys), axis=1), omega_lambda,
                     color = (0.76171875, 0.1640625 , 0.18359375))
        ax2.contourf(omega_m, omega_lambda, np.transpose(prob_sys), 
                     levels=[np.exp((6.17+_chi2_sys.min())/(-2)),np.exp((2.3+_chi2_sys.min())/(-2)),1],
                     origin='lower', 
                     colors=[(0.80078125, 0.328125 , 0.34765625), (0.76171875, 0.1640625 , 0.18359375)],
                     alpha=0.7, linewidths=2)
        ax2.text(0.4,0.8,"Pantheon", color='red',rotation=40,alpha=0.7, size=15)
    if len(prob_nosys) != 0:
        axHistx.plot(omega_m, np.sum(np.transpose(prob_nosys), axis=0),
                     color = (0.4,0.4,0.4))
        axHisty.plot(np.sum(np.transpose(prob_nosys), axis=1), omega_lambda,
                     color = (0.4,0.4,0.4))
        ax2.contourf(omega_m, omega_lambda, np.transpose(prob_nosys), 
                     levels=[np.exp((6.17+_chi2_nosys.min())/(-2)),np.exp((2.3+_chi2_nosys.min())/(-2)),1],
                     origin='lower', 
                     colors=[(0.5,0.5,0.5), (0.4,0.4,0.4)],
                     alpha=0.7, linewidths=2)
        ax2.text(0.21,1.05,"Pantheon (Stat)", color='gray',rotation=40,alpha=0.7, size=15)
    xmin = 0.1
    xmax = 0.59
    ymin = 0.4
    ymax = 1.2
    axHistx.set_xlim(xmin,xmax)
    axHisty.set_ylim(ymin,ymax)
    ax2.set_xlim(xmin,xmax)
    ax2.set_ylim(ymin,ymax)
    
    plt.show()

def mcmc_result(parameters):
    keys = parameters[1].keys()
    latex_dic = {'Omega_m': r"$\Omega_m$",
                 'Omega_lambda': r"$\Omega_{\Lambda}$",
                 'H0': r"$H_0$",
                 'M_nuisance': r"$M$",
                 'Omega_k': r"$\Omega_k$"}
    _label = []
    for i in keys:
        _label.append(latex_dic[i])
    pars_array = []
    for _pars in parameters:
        pars_array.append(list(_pars.values()))
    pars_array = np.array(pars_array)
    figure = corner.corner(pars_array, labels=_label,
                           color='k', quantiles=[0.16, 0.5, 0.84],
                           show_titles=True, title_kwargs={"fontsize": 12})
    corner_axes = np.array(figure.get_axes()).reshape(len(keys), len(keys))
    for ax in np.diag(corner_axes):
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('none')
    plt.show()

def trace_plot(parameters):
    keys = parameters[1].keys()
    latex_dic = {'Omega_m': r"$\Omega_m$",
                 'Omega_lambda': r"$\Omega_{\Lambda}$",
                 'H0': r"$H_0$",
                 'M_nuisance': r"$M$",
                 'Omega_k': r"$\Omega_k$"}
    _label = []
    for i in keys:
        _label.append(latex_dic[i])

    params = {'legend.fontsize': 16,
          'figure.figsize': (15, 5),
         'axes.labelsize': 17,
         'axes.titlesize':20,
         'xtick.labelsize':10,
         'ytick.labelsize':10,
         'ytick.major.size': 5.5,
         'axes.linewidth': 2}
    plt.rcParams.update(params)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    fig, axs = plt.subplots(nrows=len(keys), ncols=1, sharex=True, figsize=(15, 10), gridspec_kw={'hspace': 0})
    pars_array = []
    for _pars in parameters:
        pars_array.append(list(_pars.values()))
    pars_array = np.array(pars_array)
    for i in np.arange(len(keys)):
        axs[i].plot(np.arange(len(parameters)) + 1, pars_array[:, i])
        axs[i].set_ylabel(_label[i], size=15)

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel('steps', size=20, labelpad=12)
    plt.title('Trace Plot')
    plt.show()



if __name__ == '__main__':

    samples = Chain.simulator(1000)  #plot data from simulator, just a test
    
    mcmc_result(samples) #check all the parameters
    
    trace_plot(samples) #trace plot as a sanity check
    
    omega_m, omega_lambda, prob = samples_process(samples=samples, x_range=[0, 1.6], y_range = [0, 2.5], xbin=30, ybin=40) #fig 18
    fig18(omega_m, omega_lambda, prob_nosys=prob, prob_sys=[])  #fig 18
    
    print ("plot_mc.py is tested!")