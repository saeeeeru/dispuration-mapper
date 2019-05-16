import os

import numpy as np
import pandas as pd

from utils import one_hot_vector

# event_list = ['pass', 'dribble', 'miscontrol', 'duel']
possess = 'Complete'
loss = 'Incomplete'

def produce_inputs(infile):
	"""
	args:
		- infile : input filename (.csv)

	return:
		- data_dict : keys() = ['Y', 'X_atk', 'X_def', 'S', 'period', 'team_list']

	"""
	
	# read event data from infile
	inputs_df = pd.read_csv(os.path.join('..','dat',infile))

	# generate inputs X, Y, Z for estimating GLSR model
	team_list = inputs_df.possession_team.unique().tolist()

	Y, X_atk, X_def, S, period_list = [], [], [], [], []
	df_possession_team, df_outcome, df_period = inputs_df.possession_team, inputs_df.outcome, inputs_df.period
	df_x, df_y = inputs_df.start_location_x, inputs_df.start_location_y

	for idx in range(inputs_df.shape[0]):
		possession_team = df_possession_team.iloc[idx]
		outcome = df_outcome.iloc[idx]
		period = df_period.iloc[idx]
		start_location_x, start_location_y = df_x.iloc[idx], df_y.iloc[idx]

		X_atk.append(one_hot_vector(possession_team, team_list))
		defence_team = [team for team in team_list if team != possession_team][0]  # must change this row
		X_def.append(one_hot_vector(defence_team, team_list))

		y = 1 if outcome == loss else 0
		Y.append(y)

		S.append([start_location_x, -start_location_y])

		period_list.append(period)

	return {'Y': np.array(Y).reshape(-1,1), 'X_atk': np.array(X_atk), 'X_def': np.array(X_def), 
				'S': np.array(S), 'period':np.array(period_list), 'team_list':team_list}