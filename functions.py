# Basic data analysis for protein proteomics

# First, import things

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

import numpy as np
from sklearn.decomposition import PCA, NMF
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, normalize
from math import log10, log2, ceil, floor, sqrt, log, e
from missforest import MissForest
from venny4py.venny4py import venny4py
import os

from functions import *

from helpers import general_helpers as gh
from helpers import stats_helpers as sh
from helpers import proteomics_helpers as ph
from helpers import mpl_plotting_helpers as mph


###################################################################################################


def safe_log2(number):
    try:
        return log2(number)
    except:
        return float("nan")

def safe_log10(number):
    try:
        return log10(number)
    except:
        return float("nan")

def keep_first(a_matrix, keycol = "U_ID", heads = 0):
    keepers = []
    seen = []
    keyind = a_matrix[heads].index(keycol)
    i=0
    for row in a_matrix:
        if row[keyind] not in seen:
            keepers.append(row)
            seen.append(row[keyind])
        if i%100000 == 0:
            print(f"\t{i}")
        i+=1
    return keepers

###################################################################################################



# PCA function to make my life less miserable
def square_bounds(mpl_axes, inplace = False):
    ticks = list(mpl_axes.get_xticks()) + list(mpl_axes.get_yticks())
    if inplace:
        mpl_axes.set_xlim(min(ticks), max(ticks))
        mpl_axes.set_ylim(min(ticks), max(ticks))
    else:
        return min(ticks), max(ticks)

def pca_analysis(a_df, pca_kwargs = dict(n_components = 2,
                                         whiten = False,
                                         svd_solver = "full",
                                         tol = 0)):
    std_scalar = StandardScaler()
    scaled = std_scalar.fit_transform(a_df.transpose())
    pca_analysis = PCA(**pca_kwargs)
    components = pca_analysis.fit_transform(scaled)
    components = gh.transpose(*[list(point) for point in list(components)])
    return components, pca_analysis

def nmf_analysis(a_df, nmf_kwargs = dict(n_components = 2, 
                                     init = "nndsvd", # preserves sparseness
                                     solver = "mu", # multiplicative update
                                     beta_loss = "frobenius", # stable, but slow
                                     alpha_W = 0,  # default
                                     alpha_H = 0,  # default
                                     l1_ratio = 0  # default
                                    )):
    #std_scalar = StandardScaler()
    #scaled = std_scalar.fit_transform(a_df.transpose())
    norms = normalize(a_df)
    nmf_analysis = NMF(**nmf_kwargs)
    W = nmf_analysis.fit_transform(norms)
    H = nmf_analysis.components_
    return H, W
    
def cluster_plotting(dataframes, # list with minimum 1 df
                 groups,     
                 expnames, # should match len(dataframes)
                 filenames,# should match len(dataframes)
                 group_slices, #assumes reps are clustered in list
                 labels,       # should correspons to len(group_slices)
                 colours,      # list of lists, each sublist should correspond to len(group_slices)
                 markers = ["o","^", "s"],
                 cluster = 'PCA', # other option is NNMF
                 markersize=100, 
                 textdict = dict(fontfamily = "sans-serif",
                 font = "Arial",
                 fontweight = "bold",
                 fontsize = 10),
                 square = True,
                 pca_kwargs = dict(n_components = 30,
                                   whiten = False,
                                   svd_solver = "full",
                                   tol = 0),
                 nmf_kwargs = dict(n_components = 2, 
                                   init = "nndsvd", # preserves sparseness
                                   solver = "mu", # multiplicative update
                                   beta_loss = "frobenius", # stable, but slow
                                   alpha_W = 0,  # default
                                   alpha_H = 0,  # default
                                   l1_ratio = 0  # default
                                    )):
    # Get the data columns from the dataframes, remove
    # missing values, run PCA analysis with sklearn,
    # scatter
    
    # Grab the columns corresponding to the groups of data. Assumes the 'groups' strings
    # are a substring of the column headers
    dfs = [df[[name for name in list(df.columns) if any(s in name for s in groups)]] for df in dataframes]
    # Remove any row with missing values, as PCA doesn't tolerate MVs
    dfs = [df.dropna() for df in dfs]
    axes = []
    i = 0
    print(cluster)
    for df in dfs:
        if cluster.lower() == "pca":
            print('here')
            components,pca = pca_analysis(df, pca_kwargs = pca_kwargs)
        else:
            # will add nmf soon
            print('don')
            components,nmf = nmf_analysis(df, nmf_kwargs = nmf_kwargs)
        fig, ax = plt.subplots(figsize = (6,6))
        # Next, loop over the slices and scatter
        j = 0
        print(len(components[0]))
        for g in group_slices:
            ax.scatter(components[0][g], components[1][g], 
                       color = colours[i][j],
                       marker = markers[j], 
                       s = markersize, 
                       alpha = 0.75,          # my preference
                       label = labels[j],
                       edgecolor = "black",   # my preference
                      )
            j+=1
        ax.set_title(expnames[i], **textdict)
        if cluster.lower() == "pca":
            ax.set_xlabel(f"PC1 ({100*pca.explained_variance_ratio_[0]:.2f}%)",**textdict)
            ax.set_ylabel(f"PC2 ({100*pca.explained_variance_ratio_[1]:.2f}%)", **textdict)
            #square_bounds(ax, inplace = True)
        else:
            # will add nmf soon
            ax.set_xlabel(f"Component 1", **textdict)
            ax.set_ylabel(f"Component 2", **textdict)
        if square:
            square_bounds(ax, inplace = True)
        mph.update_ticks(ax, which = "x")
        mph.update_ticks(ax, which ="y")
        ax.legend()
        plt.tight_layout()
        plt.savefig(filenames[i])
        axes.append(ax)
        plt.close()
        i+=1
    return axes


###################################################################################################
## Multiple Regression, because fuck doing the pairwise bullshit

def format_linreg_strs(coeffs, r2, intercept, label = "Biggest Dong"):
    outstr = f"{label}\n$y={intercept:.3f}"
    for i in range(len(coeffs)):
        outstr += fr"+{coeffs[i]:.3f}x_{{{i+1}}}"
    outstr += f"$\n$R={r2:.3f}$"
    return outstr

def plot_linreg_strs(strs, save = "test.pdf",
                    fontdict = dict(fontfamily = "sans-serif",
                                      font = "Arial",
                                      fontweight = "bold",
                                      fontsize = 2)):
    """
    Just plot the strings to exploit LaTeX math formatting
    """
    fig, ax = plt.subplots()
    # Check how many strings there are, and adjust the axes accordingly
    num = len(strs)
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,num)
    ax.set_yticks(list(range(num)))
    # turn off the bounding box
    ax.axis("off")
    # plot the strings
    for i in range(num):
        ax.text(0, i, strs[i], ha = "center", va = "center", **fontdict)
    plt.savefig(save)
    plt.close()
    
def multi_reg_lineplot(file, #dataframe
                       groups = ["t1", "t2", "t3"], #substring of header, group indicator
                       labels = ["0 min", "2 min", "5 min"], # Goes above the strings
                       log2_trans = True,
                       savefile = "test.pdf", # path/to/file.pdf
                       fontdict= dict(fontfamily = "sans-serif",
                                      font = "Arial",
                                      fontweight = "bold",
                                      fontsize = 2)
                       ):
    g_num = len(groups)
    split_f = [file[[c for c in list(file.columns) if groups[i] in c]] for i in range(g_num)]
    if log2_trans:
        split_f = [[list(row) for row in list(g.astype(float).transform(np.log2).to_numpy())] for g in split_f]
    else:
        split_f = [[list(row) for row in list(g.astype(float).to_numpy())] for g in split_f]
    # LinearRegression can't take missing values
    split_f = [[row for row in g if all([item == item for item in row])] for g in split_f]
    xs = [[row[:-1] for row in g] for g in split_f]
    ys = [gh.transpose(*[[row[-1]] for row in g])[0] for g in split_f]
    # Set up the model
    linmods = [LinearRegression() for _ in range(g_num)]
    # and fit it, always assume y is the last replicate in a group
    regs = [linmods[i].fit(xs[i], ys[i]) for i in range(g_num)]
    scores = [sqrt(regs[i].score(xs[i],ys[i])) for i in range(g_num)]
    # Now we make the strings
    strs = [format_linreg_strs(regs[i].coef_, scores[i], regs[i].intercept_, labels[i]) for i in range(g_num)]
    # And pass them to the plotter
    plot_linreg_strs(strs, save=savefile, fontdict = fontdict)
    return None

###################################################################################################

def hist_matrix(a_matrix, # Assumes no headers, remove them
                 ylabels,
                xlabels,
                bins = 33,
               imputed = None, # Must have the same shape as a_matrix, minus the number of points
               ymax = 150,
                xlims = (15,35),
                ygroup_size = None, # Should be top to bottom
               y_groups = None,     # Should be top to bottom
                title = "Lick mah balls",
                fontdict = dict(fontfamily = "sans-serif",
                                font = "Arial",
                                fontstyle = "normal"),
                primary = ("lavendar", "mediumpurple"),
                secondary = ("pink", "mediumvioletred"),
                figsize = (16,12),
                savefile = "histogram_matrix.pdf",
                missing = [],
               ):
    print(missing)
    font = font_manager.FontProperties(family='sans-serif',
                                   style='normal', size=8)
    fig, ax = plt.subplots(len(ylabels), len(xlabels),
                           figsize = figsize)
    # This is the number of rows in the figure
    for i in range(len(ylabels)):
        # This is the number of cols in the figure,
        # should equal len(xlabels)
        for j in range(len(xlabels)):
            ax[i][j].set_xlim(*xlims)
            ax[i][j].set_ylim(0,ymax)
            if (i,j) not in missing:
                hist_med = sh.median(a_matrix[i][j])
                ax[i][j].hist(a_matrix[i][j], bins = bins,
                              color = primary[0], edgecolor = primary[1],
                              label = fr"$n={len([d for d in a_matrix[i][j] if d == d])}$")
                ax[i][j].plot([hist_med, hist_med], [0,ymax], color = primary[0],
                             linestyle = ":",
                             label = fr"$n_{{MED}}={hist_med:.2f}$")
                if imputed != None:
                    imp_med = sh.median(imputed[i][j])
                    ax[i][j].hist(imputed[i][j], bins = bins,
                                  color = secondary[0], edgecolor = secondary[1],
                                  label = fr"$i={len([d for d in imputed[i][j] if d == d])}$")
                    ax[i][j].plot([imp_med, imp_med], [0,ymax], color = secondary[0],
                                 linestyle = ":",
                                 label = fr"$i_{{MED}}={imp_med:.2f}$")

                # Lifestyle choices 
                ax[i][j].legend(loc = "upper right", prop=font)
            if j != 0:
                ytcks = list(ax[i][j].get_yticks()[1:])
                ax[i][j].set_yticks(ytcks)
                ax[i][j].set_yticklabels(["" for _ in range(len(ytcks))])
            elif (i,j) in missing:
                ytcks = [0] + list(ax[0][0].get_yticks())
                ax[i][j].set_yticks(ytcks)
                mph.update_ticks(ax[i][j],
                                 which = "y",
                                  fontdict = {"fontfamily" : "sans-serif",
                                   "font" : "Arial",
                                   #"ha" : "center",
                                   "fontweight" : "bold",
                                   "fontsize" : "8"})
                ax[i][j].set_ylabel(ylabels[i], fontweight = "bold", fontsize= "8",
                                         **fontdict)
            else:
                mph.update_ticks(ax[i][j],
                                 which = "y",
                                  fontdict = {"fontfamily" : "sans-serif",
                                   "font" : "Arial",
                                   #"ha" : "center",
                                   "fontweight" : "bold",
                                   "fontsize" : "8"})
                ax[i][j].set_ylabel(ylabels[i], fontweight = "bold", fontsize= "8",
                                         **fontdict)
            if i != len(ylabels)-1:
                xtcks = list(ax[i][j].get_xticks()[1:])
                ax[i][j].set_xticks(xtcks)
                ax[i][j].set_xticklabels(["" for _ in range(len(xtcks))])
            elif (i,j) in missing:
                xtcks = [xlims[0]] + list(ax[0][0].get_xticks())
                ax[i][j].set_xticks(xtcks)
                mph.update_ticks(ax[i][j],
                                 which = "x",
                                  fontdict = {"fontfamily" : "sans-serif",
                                   "font" : "Arial",
                                   #"ha" : "center",
                                   "fontweight" : "bold",
                                   "fontsize" : "8"})
                ax[i][j].spines[:].set_visible(False)
            else:
                mph.update_ticks(ax[i][j],
                                     which = "x",
                                      fontdict = {"fontfamily" : "sans-serif",
                                       "font" : "Arial",
                                       #"ha" : "center",
                                       "fontweight" : "bold",
                                       "fontsize" : "8"})
                ax[i][j].set_xlabel(xlabels[j], fontweight = "bold", fontsize = 8,
                                        **fontdict)
    # Get the spacing correct
    plt.subplots_adjust(wspace = 0.1, hspace = 0.1)
    if y_groups != None and ygroup_size != None:
        # Place the text strings where they should go
        # These are in figure units, not axes units
        fig.text(0.02,0.45, title, rotation = 90,
                fontweight = "bold", fontsize = "14",
                 **fontdict)
        # Loop over the group sizes in reverse order
        displacement = 0
        index = 0
        for gsize in [ygroup_size[-i] for i in range(len(ygroup_size))]:
            # Calculate displacement for group
            center = (gsize/sum(ygroup_size))/2 + displacement
            fig.text(0.05, center, y_groups[index], rotation = 90,
                     va = "center", ha = "center",
                     fontweight = "bold", fontsize = "14",
                     **fontdict)
            displacement += gsize/sum(ygroup_size)
            index += 1
    plt.savefig(savefile)
    plt.close()
    return None

###################################################################################################

def plot_counts(transpose_file, 
                file_root,
                group_labels,
                group_indices,
                sample_counts = True,
                dotplot_kwargs = dict(rotation = 90, 
                                      ylabel = "Counts",
                                      colours = ["grey" for _ in range(20)],
                                      markersize = 25,
                                      figsize = (3,6),
                                      ymin = 0)):
    if not sample_counts:
        sample_counts = [[item for item in col if item == item] for col in transpose_file]
        sample_counts= [len(col) for col in sample_counts]
        sample_counts = [[group_labels[i], [sample_counts[index] for index in group_indices[i]]]
                         for i in range(len(group_indices))]
    else:
        sample_counts = transpose_file
    print(sample_counts)
    stats = sh.HolmSidak(*sample_counts, override = True)
    if os.path.exists(f"{file_root}_stats.csv"):
        os.remove(f"{file_root}_stats.csv")
    stats.write_output(filename = f"{file_root}_stats")
    mph.dotplot(sample_counts, comparisons = stats.output[2],
                filename = f"{file_root}_fig.pdf",
                **dotplot_kwargs)
    return None

###################################################################################################

def cv_perc(std_dev):
    s2 = (std_dev*log(2))**2
    return sqrt(e**s2 - 1) * 100

def venn_4set(transpose_file, 
              group_inds, 
              group_labels, 
              index_col = 0,
              colours = ["grey" for _ in range(20)],
              stringency = 1,
              filename = "nonsense.pdf"):
    sets = {group_labels[i] : [transpose_file[index_col]] + [transpose_file[index] for index in group_inds[i]]
            for i in range(len(group_labels))}
    sets = {key : gh.transpose(*value) for key, value in sets.items()}
    sets = {key : [row for row in value if len([1 for _ in row[1:] if _==_]) >= stringency]
            for key, value in sets.items()}
    sets = {key : set(gh.transpose(*value)[0]) for key, value in sets.items()}
    venny4py(sets = sets, 
             colors = colours)
    plt.savefig(filename)
    return None

###################################################################################################
