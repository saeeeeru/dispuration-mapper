import os, time, pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import generate_params
from preprocess import produce_inputs
from model import GLSRModel

def run_one_match(params_dict):

	# produce inputs from infile
	print('generating input data...')
	data_dict = produce_inputs(params_dict['infile'])

	# estimate glsr model
	glsr_model = GLSRModel(J=len(data_dict['team_list']))

	print('estimating GLSR model...')
	glsr_model.estimate_gp(data_dict, flag_period=True)

def main():
	# parse params
	params_dict = generate_params()

	run_one_match(params_dict)


if __name__ == '__main__':
	main()