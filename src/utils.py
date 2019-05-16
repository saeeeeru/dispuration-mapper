import configargparse, math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def generate_params():
	parser = configargparse.ArgParser()
	parser.add('-infile', '--infile', dest='infile', type=str, help='input filename (.csv)')
	params_dict = vars(parser.parse_args())

	return params_dict

def one_hot_vector(value, value_list):
	n_label = len(value_list)
	idx = value_list.index(value)
	return np.eye(n_label, dtype=int)[idx].tolist()

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def reverse_colormap(cmap, name = 'my_cmap_r'):
	"""
	In: 
	cmap, name 
	Out:
	my_cmap_r

	Explanation:
	t[0] goes from 0 to 1
	row i:   x  y0  y1 -> t[0] t[1] t[2]
				   /
				  /
	row i+1: x  y0  y1 -> t[n] t[1] t[2]

	so the inverse should do the same:
	row i+1: x  y1  y0 -> 1-t[0] t[2] t[1]
				   /
				  /
	row i:   x  y1  y0 -> 1-t[n] t[2] t[1]
	"""        
	reverse = []
	k = []   

	for key in cmap._segmentdata:    
		k.append(key)
		channel = cmap._segmentdata[key]
		data = []

		for t in channel:                    
			data.append((1-t[0],t[2],t[1]))            
		reverse.append(sorted(data))    

	LinearL = dict(zip(k,reverse))
	my_cmap_r = mpl.colors.LinearSegmentedColormap(name, LinearL) 
	return my_cmap_r