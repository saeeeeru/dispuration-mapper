import os, time, random

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from gpc import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import Matern

from output_func import plot_surface

period_list = ['1st', '2nd']

class GLSRModel(object):
	"""docstring for GLSRModel"""
	def __init__(self, J):
		# J = length of team_list

		# coefficients
		self.beta_atk = np.zeros(J)
		self.beta_def = np.zeros(J)

		# self.alpha = np.random.rand()

	def estimate_gp(self, data_dict, flag_period=False):
		"""
		args:
			- data_dict : input dataset (type=dict)
			- flag_period : flag whether to split the periods
		"""

		# gaussian process
		kernel = Matern()  # kernel for gaussian process
		N = len(data_dict['team_list'])

		if flag_period:
			for i, team in enumerate(data_dict['team_list']):
				print(f'{team} ({i+1} of {N})')	
				for j, period in enumerate(np.unique(data_dict['period'])):
					st = time.time()
					idx = np.array([f1 and f2 for f1, f2 in zip(data_dict['X_def'][:, i]==1, data_dict['period']==period)])

					vs_team = [team_tmp for team_tmp in data_dict['team_list'] if team_tmp != team][0]
					vs_team = f'{vs_team}[{period_list[j]}]'
					print(f'{team}(vs {vs_team}), length={idx.sum()}')

					model = GaussianProcessClassifier(kernel=kernel).fit(data_dict['S'][idx], data_dict['Y'][idx])
					plot_surface(model, 'DS', team=team, vs_team=vs_team)
					print(f'elapsed time: {time.time()-st:.3f}[sec]')

		else:
			for i, team in enumerate(data_dict['team_list']):
				print(f'{team} ({i+1} of {N})')
				idx = data_dict['X_def'][:, i]==1
				st = time.time()
				
				model = GaussianProcessClassifier(kernel=kernel).fit(data_dict['S'][idx], data_dict['Y'][idx])

				vs_team = [team_tmp for team_tmp in data_dict['team_list'] if team_tmp != team][0]
				print(f'{team}(vs {vs_team}), length={idx.sum()}')
				plot_surface(model, 'DS', team=team, vs_team=vs_team)
				print(f'elapsed time: {time.time()-st:.3f}[sec]')