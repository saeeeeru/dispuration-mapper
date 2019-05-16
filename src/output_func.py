import os

import numpy as np
import matplotlib.pyplot as plt

from utils import reverse_colormap

xmax, ymin = 120, -80
zmax, zmin = 1, -2

# function of drawing base pitch
# please change your data collection environment
def draw_pitch(ax):
	ax.spines['right'].set_visible(False)
	ax.spines['left'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.spines['bottom'].set_visible(False)
	
	ax.plot([0,xmax], [0, 0], color='black')
	ax.plot([xmax,xmax], [0, ymin], color='black')
	ax.plot([xmax,0], [ymin,ymin], color='black')
	ax.plot([0, 0], [ymin, 0], color='black')
	ax.plot([xmax/2, xmax/2], [0, ymin], color='black')
	ax.plot([0, 0], [-36, -44], color='black', linewidth=10)
	ax.plot([xmax, xmax], [-36, -44], color='black', linewidth=10)

#         ax.scatter(60, 40, color='black')
	centreCircle = plt.Circle((60, -40), 12, color="black",fill=False)
	ax.add_patch(centreCircle)

	ax.plot([0, 18],  [-18, -18], color='black')
	ax.plot([18, 18],  [-18, -62], color='black')
	ax.plot([18, 0],  [-62, -62], color='black')
	ax.plot([0, 6],  [-30, -30], color='black')
	ax.plot([6, 6],  [-30, -50], color='black')
	ax.plot([6, 0],  [-50, -50], color='black')
#         ax.scatter(12, 40, color='black')

	ax.plot([xmax, 102],  [-18, -18], color='black')
	ax.plot([102, 102],  [-18, -62], color='black')
	ax.plot([102, xmax],  [-62, -62], color='black')
	ax.plot([xmax, 114],  [-30, -30], color='black')
	ax.plot([114, 114],  [-30, -50], color='black')
	ax.plot([114, xmax],  [-50, -50], color='black')
#         ax.scatter(108, 40, color='black')

# function of drawing multi pitches
def draw_multi_pitch(nrows=2, ncols=2):
	period_list = ['1st', '2nd']
	fig, axes = plt.subplots(nrows, ncols, figsize=(10*ncols/2, 7*nrows/2))
	for i, period in enumerate(period_list):
		axes[i, 0].set_ylabel(period)

	for i, axes_tmp in enumerate(axes):
		for j, ax in enumerate(axes_tmp):
#             ax.set_title(period)

			draw_pitch(ax)
	
	return fig, axes

# function of drawing disruption map
def plot_surface(model, surface, team, vs_team=None):
	"""
	args:
		- model : gaussian process model object
		- surface : the visualized surface
		- team : name of team which has gaussian process model
		- vs_team : name of opponent team
	"""

	title = f'vs {vs_team}' if vs_team else team
	filename = f'{team}(vs {vs_team})' if vs_team else team

	n_grid = 100
	x = np.linspace(0, xmax, n_grid)
	y = np.linspace(0, ymin, n_grid)
	xx, yy = np.meshgrid(x, y)

	s_test = np.vstack([xx.ravel(), yy.ravel()]).transpose(1, 0)
	# mean = Z.predict(s_test)
	mean = model.predict_Z(s_test)

	mean[mean < zmin] = zmin
	mean[mean > zmax] = zmax

	fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10+2, 7))
	fig.suptitle(title, fontsize=25)
	draw_pitch(ax)

	if surface == 'CS':
		# set atacking direction
		ax.arrow(x=50,y=2,dx=20,dy=0,width=0.05,head_width=1,head_length=2,length_includes_head=True,color='black')
		ax.text(47.5, 2, 'attacking', va='center', ha='right', fontsize=18)
	else:
		# set defencing direction
		ax.arrow(x=70,y=2,dx=-20,dy=0,width=0.01,head_width=1,head_length=2,length_includes_head=True,color='black')
		ax.text(72.5, 2, 'defencing', va='center', ha='left', fontsize=18)

	im = ax.pcolormesh(xx, yy, mean.reshape(n_grid, n_grid), cmap='coolwarm')
	fig.colorbar(im, ax=ax)
	im.set_clim(vmin=zmin, vmax=zmax)

	ax.set_ylim(ymin, 4)
	outfile = os.path.join('..','fig',f'{filename}.png')
	if os.path.exists(outfile): outfile = os.path.join('_fig',surface,f'{filename}_.png')
	plt.savefig(outfile, bbox_inces='tight')
	plt.close()

if __name__ == '__main__':
	fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 7))
	draw_pitch(ax)
	plt.show()