import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_corr_matrix(df, title):
  corr_matrix = np.abs(df.corr()).round(decimals=3)

  mask = np.zeros_like(corr_matrix, dtype=bool)
  mask[np.triu_indices_from(mask)]= True

  f, ax = plt.subplots(figsize=(15, 20)) 
  heatmap = sns.heatmap(corr_matrix, 
                        mask = mask,
                        square = True,
                        linewidths = .5,
                        cmap = "OrRd",
                        cbar_kws = {'shrink': .6, "ticks" : [0, 0.5, 1]},
                        vmin = 0, 
                        vmax = 1,
                        annot = True,
                        annot_kws = {"size": 12})
  #add the column names as labels
  ax.set_title(title)
  ax.set_yticklabels(corr_matrix.columns, rotation = 0)
  ax.set_xticklabels(corr_matrix.columns)
  sns.set_style({'xtick.bottom': True}, {'ytick.left': True})