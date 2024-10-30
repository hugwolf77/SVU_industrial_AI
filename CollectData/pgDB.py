import psycopg2 as pg2
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
# import numpy as np

# load datafile
data = './data/economic_books_50_SVU_02.xlsx'
df = pd.read_excel(data, header=0, index_col= 'ID')
col = ['MoneyVsReal','ComVsInd','IrrationalVsRation']


fig = plt.figure(figsize=(10,6))
axs= fig.add_subplot(projection="3d")

fontlabel = {"fontsize":"large", "color":"gray", "fontweight":"bold"}
colors = ('r', 'g', 'b', 'k')

axs.scatter(df[col[0]], df[col[1]], df[col[2]], c=df['class'], cmap=colors, s=20, alpha=0.5)
axs.legend()
axs.set_xlim(-1,1)
axs.set_ylim(-1,1)
axs.set_zlim(-1,1)
axs.set_xlabel("Monney vs Real", fontdict=fontlabel, labelpad=16)
axs.set_ylabel("Community vs Individual", fontdict=fontlabel, labelpad=16)
axs.set_zlabel("Irration vs Ration", fontdict=fontlabel, labelpad=16)
axs.set_title("classification Economic Author", fontdict=fontlabel)
axs.view_init(elev=30) #, azim=360)    # 각도 지정
plt.show()


# def init():
#     axs.scatter(df[col[0]], df[col[1]], df[col[2]], c=df[col[2]], cmap=cm.coolwarm, s=20, alpha=0.5)
#     axs.legend()
#     axs.set_xlim(-1,1)
#     axs.set_ylim(-1,1)
#     axs.set_zlim(-1,1)
#     axs.set_xlabel("Monney vs Real", fontdict=fontlabel, labelpad=16)
#     axs.set_ylabel("Community vs Individual", fontdict=fontlabel, labelpad=16)
#     axs.set_zlabel("Irration vs Ration", fontdict=fontlabel, labelpad=16)
#     axs.set_title("classification Economic Author", fontdict=fontlabel)
#     axs.view_init(elev=30.)#, azim=90)    # 각도 지정
#     return fig,

# def animate(i):
#     axs.view_init(elev=30., azim=i)
#     return fig,

# # Animate
# anim = animation.FuncAnimation(fig, animate, init_func=init, frames=360, interval=30, blit=True)

# # Save
# anim.save('mpl3d_scatter.gif', fps=30)