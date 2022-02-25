import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import scipy.io

from sklearn import preprocessing
import datetime
import copy
import math
import random
from random import uniform as uni

pd.set_option('chained_assignment', None)

def cv_split_data(df2, predictor_headers, outcome_header, task = 'classification', dummy_cols = [], synth=False):
	df2 = df2.sample(frac=1, random_state=0).reset_index(drop=True)
	if max_dataset_size >= 0 and max_dataset_size < len(df2):
		df2 = df2.sample(n=max_dataset_size, random_state=0)

	if task == 'regression':
		df_X = df2[predictor_headers]
		df_Y = df2[[outcome_header]]
	else:
		df_Y1 = df2.iloc[:,-2]
		df_Y2 = df2.iloc[:,-1]

		df_X = df2[predictor_headers]
		df_Y = pd.concat([df_Y1, df_Y2], axis=1)
	all_folds = []
	if task == 'regression':
		df_Y_slice = df_Y.iloc[:,0]
	else:
		df_Y_slice = df_Y.iloc[:,1]
	# random.seed(0)

	if synth:
		num_folds = 3
	else:
		num_folds = 10

	if task == 'regression':
		kf = KFold(n_splits=num_folds, random_state=0, shuffle=True)
	else:
		kf = StratifiedKFold(n_splits=num_folds, random_state=0, shuffle=True)
	fold_idx = -1
	for d1, test in kf.split(df_X, df_Y_slice):
		fold_idx += 1
		if only_fold_index == 0:
			if fold_idx != only_fold_index:
				continue
			else:
				pass
				# print ('\nOnly Using Fold ' + str(fold_idx+1) + '\n')

		d1_X = df_X.iloc[d1]
		d1_Y = df_Y.iloc[d1]
		te_x = df_X.iloc[test]
		te_y = df_Y.iloc[test]

		if task == 'regression':
			d1_Y_slice = d1_Y.iloc[:,0]
		else:
			d1_Y_slice = d1_Y.iloc[:,1]

		# skf2 = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)

		if synth:
			test_size = 0.5
		else:
			test_size = 0.111111111111

		if task == 'classification':


			sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=0)
			train, valid = [x for x in sss.split(d1_X, d1_Y_slice)][0]
		elif task == 'regression':
			train, valid = train_test_split(list(range(len(d1_Y_slice))), test_size=test_size, random_state=0)

		# for preserving random seed effects
		if only_fold_index > 0:
			if fold_idx != only_fold_index:
				continue
			else:
				print ('\nOnly Using Fold ' + str(fold_idx+1) + '\n')

		tr_x = d1_X.iloc[train]
		tr_y = d1_Y.iloc[train]
		va_x = d1_X.iloc[valid]
		va_y = d1_Y.iloc[valid]

		assert list(tr_x.index.values) == list(tr_y.index.values)
		assert list(va_x.index.values) == list(va_y.index.values)
		assert list(te_x.index.values) == list(te_y.index.values)

		assert df_X.shape[0] == len(tr_x.index.values) + len(va_x.index.values) + len(te_x.index.values)

		all_folds.append({'tr_x':tr_x, 'tr_y':tr_y, 'va_x':va_x, 'va_y':va_y, 'te_x':te_x, 'te_y':te_y,
			'tr_indices':tr_y.index.values, 'va_indices':va_y.index.values, 'te_indices':te_y.index.values, 'dummy_cols':dummy_cols})

		if synth: break
	return all_folds

def oversample_data(all_folds, predictor_headers, outcome_header, task='classification'):
	for fold in all_folds:
		tr_x = fold['tr_x']
		tr_y = fold['tr_y']

		# recombine tr_x and tr_y momentarily
		tr_both = pd.concat([tr_x, tr_y], axis=1)
		temp1 = tr_both.loc[tr_y.iloc[:,0] == 1]
		temp2 = tr_both.loc[tr_y.iloc[:,0] == 0]

		if len(temp1) > len(temp2):
			tr0 = temp1
			tr1 = temp2
		else:
			tr0 = temp2
			tr1 = temp1
		assert(len(tr0) > len(tr1))
		tile1 = tr1.copy()

		while(len(tr1) + len(tile1)) <= len(tr0):
			tr1 = pd.concat([tr1, tile1])
		size_diff = len(tr0) - len(tr1)
		if size_diff != 0:
			# continue oversampling by randomly selecting without replacement
			np.random.seed(0)
			remaining_indices = np.random.choice(len(tile1), size_diff, replace=False)
			tile1_subset = tile1.iloc[remaining_indices,:]
			tr1 = pd.concat([tr1, tile1_subset])
		#rejoin balanced classes and shuffle rows
		tr = pd.concat([tr0,tr1])
		tr = tr.sample(frac=1, random_state=0).reset_index(drop=True)
		#split tr_x and tr_y again
		tr_x = tr[predictor_headers]
		if task == 'regression':
			tr_y = tr.iloc[:,-1:]
		else:
			tr_y = tr.iloc[:,-2:]
		#return the splits to the all_folds dict
		fold['tr_x'] = tr_x
		fold['tr_y'] = tr_y
	return all_folds

def scale_data(all_folds, impute_mean = False, features_to_scale = []):
	#scale only numeric predictors with standard scaler (removing mean and dividing by variance)
	for fold in all_folds:
		fold['tr_x_orig'] = fold['tr_x']
		fold['te_x_orig'] = fold['te_x']
		tr_x = fold['tr_x'].copy()
		tr_y = fold['tr_y'].copy()
		fold['tr_y_orig'] = fold['tr_y']
		fold['te_y_orig'] = fold['te_y']
		te_x = fold['te_x'].copy()
		te_y = fold['te_y'].copy()
		if 'va_x' in fold: valid = True
		else: valid = False
		if valid:
			fold['va_x_orig'] = fold['va_x']
			fold['va_y_orig'] = fold['va_y']
			va_x = fold['va_x'].copy()
			va_y = fold['va_y'].copy()

		if impute_mean:
			imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0).fit(tr_x)
			tr_x_temp = imp.transform(tr_x)
			tr_x = pd.DataFrame(tr_x_temp, index=tr_x.index, columns=tr_x.columns)
			if valid:
				va_x_temp = imp.transform(va_x)
				va_x = pd.DataFrame(va_x_temp, index=va_x.index, columns=va_x.columns)		
			te_x_temp = imp.transform(te_x)
			te_x = pd.DataFrame(te_x_temp, index=te_x.index, columns=te_x.columns)
			fold['imputer'] = imp

		if features_to_scale:
			scaler_x = preprocessing.StandardScaler().fit(tr_x[features_to_scale].values)
			tr_x_scaled = scaler.transform(tr_x[features_to_scale].copy().values)
			if valid: va_x_scaled = scaler.transform(va_x[features_to_scale].copy().values)
			te_x_scaled = scaler.transform(te_x[features_to_scale].copy().values)

			tr_x[features_to_scale] = tr_x_scaled
			if valid: va_x[features_to_scale] = va_x_scaled
			te_x[features_to_scale] = te_x_scaled
		else:
			scaler_x = preprocessing.StandardScaler().fit(tr_x)
			tr_x_temp = scaler_x.transform(tr_x)
			tr_x = pd.DataFrame(tr_x_temp, index=tr_x.index, columns=tr_x.columns)
			if valid:
				va_x_temp = scaler_x.transform(va_x)
				va_x = pd.DataFrame(va_x_temp, index=va_x.index, columns=va_x.columns)
			te_x_temp = scaler_x.transform(te_x)
			te_x = pd.DataFrame(te_x_temp, index=te_x.index, columns=te_x.columns)
		if tr_y.shape[1] == 1:
			scaler_y = preprocessing.StandardScaler().fit(tr_y)
			tr_y_temp = scaler_y.transform(tr_y)
			tr_y = pd.DataFrame(tr_y_temp, index=tr_y.index, columns=tr_y.columns)
			if valid:
				va_y_temp = scaler_y.transform(va_y)
				va_y = pd.DataFrame(va_y_temp, index=va_y.index, columns=va_y.columns)
			te_y_temp = scaler_y.transform(te_y)
			te_y = pd.DataFrame(te_y_temp, index=te_y.index, columns=te_y.columns)
			fold['scaler_y'] = scaler_y


		fold['tr_x'] = tr_x
		if valid: fold['va_x'] = va_x
		fold['te_x'] = te_x
		fold['tr_y'] = tr_y
		if valid: fold['va_y'] = va_y
		fold['te_y'] = te_y
		fold['scaler_x'] = scaler_x
	return all_folds



def unscale_data(all_folds, features_to_scale = []):
	#scale only numeric predictors with standard scaler (removing mean and dividing by variance)

	assert (not features_to_scale)
	for fold in all_folds:


		tr_x = fold['tr_x'].copy()
		tr_y = fold['tr_y'].copy()
		te_x = fold['te_x'].copy()
		te_y = fold['te_y'].copy()

		if 'va_x' in fold: valid = True
		else: valid = False
		if valid:
			va_x = fold['va_x'].copy()
			va_y = fold['va_y'].copy()

		assert('te_y_orig' in fold)
		assert ('scaler_x' in fold)

		scaler_x = fold['scaler_x']

		tr_x_temp = scaler_x.inverse_transform(tr_x)
		tr_x = pd.DataFrame(tr_x_temp, index=tr_x.index, columns=tr_x.columns)
		if valid:
			va_x_temp = scaler_x.inverse_transform(va_x)
			va_x = pd.DataFrame(va_x_temp, index=va_x.index, columns=va_x.columns)
		te_x_temp = scaler_x.inverse_transform(te_x)
		te_x = pd.DataFrame(te_x_temp, index=te_x.index, columns=te_x.columns)

		if 'scaler_y' in fold:
			scaler_y = fold['scaler_y']

			tr_y_temp = scaler_y.inverse_transform(tr_y)
			tr_y = pd.DataFrame(tr_y_temp, index=tr_y.index, columns=tr_y.columns)
			if valid:
				va_y_temp = scaler_y.inverse_transform(va_y)
				va_y = pd.DataFrame(va_y_temp, index=va_y.index, columns=va_y.columns)
			te_y_temp = scaler_y.inverse_transform(te_y)
			te_y = pd.DataFrame(te_y_temp, index=te_y.index, columns=te_y.columns)


		fold['tr_x'] = tr_x
		if valid: fold['va_x'] = va_x
		fold['te_x'] = te_x
		fold['tr_y'] = tr_y
		if valid: fold['va_y'] = va_y
		fold['te_y'] = te_y
	return all_folds


def get_mapping(df, col):
	mapping = dict()
	total = len(df[col])
	val_cnts = df[col].value_counts()
	keys = list(val_cnts.keys())
	vals = list(val_cnts)
	sum_percent = 0
	first = True
	mapped_val = 0
	for k, key in enumerate(keys):
		percent = vals[k]*1./total
		sum_percent += percent
		if sum_percent <= 0.95:
			mapping[key] = str(keys[mapped_val])
			mapped_val += 1
		elif first == True:
			mapping[key] = str(keys[mapped_val])
			mapped_val += 1
			first = False
		else:
			if len(keys[k:]) == 1 and mapping[keys[k-1]] != 'other':
				mapping[key] = str(keys[mapped_val])
				mapped_val += 1
			else: mapping[key] = 'other'
	return mapping

# def add_noise(, ):

#     return 


def binarize(df, col):
	col_vals = sorted(list(set(df[col])))
	if len(col_vals)==2:
		return df[col].map({col_vals[0]:0, col_vals[1]:1}), True
	else:
		return df[col], False

def dummify(df, dummy_cols, outcome_header_to_dummify=None):
	dummy_cols2 = copy.copy(dummy_cols)
	for col_name in dummy_cols:
		df[col_name] = df[col_name].replace(get_mapping(df, col_name))
		df[col_name], binary = binarize(df,col_name)
		if binary: dummy_cols2.remove(col_name)
	if outcome_header_to_dummify is not None:
		df = pd.get_dummies(df, columns=dummy_cols2 + [outcome_header_to_dummify])
	else:
		df = pd.get_dummies(df, columns=dummy_cols2)
	return df, dummy_cols2


#TODO: make this work for datasets with labels, e.g. randomly swap 0 and 1
def add_noise(all_folds, mu=0, sigma=0.1):
	for fold_idx, fold in enumerate(all_folds):

		tr_x = fold['tr_x']
		tr_y = fold['tr_y']

		va_x = fold['va_x']
		va_y = fold['va_y']

		te_x = fold['te_x']
		te_y = fold['te_y']

		np.random.seed(fold_idx)

		noise_tr_x = np.random.normal(mu, sigma, tr_x.shape) 
		noise_tr_y = np.random.normal(mu, sigma, tr_y.shape) 

		noise_va_x = np.random.normal(mu, sigma, va_x.shape) 
		noise_va_y = np.random.normal(mu, sigma, va_y.shape) 

		noise_te_x = np.random.normal(mu, sigma, te_x.shape) 
		noise_te_y = np.random.normal(mu, sigma, te_y.shape) 

		fold['tr_x'] = tr_x + noise_tr_x
		fold['tr_y'] = tr_y + noise_tr_y

		fold['va_x'] = va_x + noise_va_x
		fold['va_y'] = va_y + noise_va_y

		fold['te_x'] = te_x + noise_te_x
		fold['te_y'] = te_y + noise_te_y

	return all_folds


def get_synthetic_test_suite(size, test_id, noise = None):

	if test_id == '':
		print('no test_id specified')
		assert(False)

	assert(type(size) == int)

	X = []
	y = []

	random.seed(0)

	for _ in range(size):

		if test_id in [1]:
			x4 = uni(0.6,1)
			x5 = uni(0.6,1)
			x8 = uni(0.6,1)
			x10 = uni(0.6,1)

			x1 = uni(0,1)
			x2 = uni(0,1)
			x3 = uni(0,1)
			x6 = uni(0,1)
			x7 = uni(0,1)
			x9 = uni(0,1)

		else:
			x1 = uni(-1,1)
			x2 = uni(-1,1)
			x3 = uni(-1,1)
			x4 = uni(-1,1)
			x5 = uni(-1,1)
			x6 = uni(-1,1)
			x7 = uni(-1,1)
			x8 = uni(-1,1)
			x9 = uni(-1,1)
			x10 = uni(-1,1)
			x11 = uni(-1,1)
			x12 = uni(-1,1)


		Xvec = [x1,x2,x3,x4,x5,x6,x7,x8,x9,x10]

		X.append(Xvec)			
	        
        # assert(noise)
		if test_id == 1:
			pred = math.pi**(x1*x2) * math.sqrt(2*x3) - math.asin(x4) + math.log(x3+x5) - (x9/x10)*math.sqrt(x7*1.0/x8) - x2*x7
		elif test_id == 2:
			pred = math.pi**(x1*x2) * math.sqrt(2*math.fabs(x3)) - math.asin(0.5*x4) + math.log(math.fabs(x3+x5)+1) - (x9/(math.fabs(x10)+1))*math.sqrt(math.fabs(x7)*1.0/(math.fabs(x8)+1)) - x2*x7
		elif test_id == 3:
			pred = math.exp(math.fabs(x1-x2))+ math.fabs(x2*x3)-(x3**2)**math.fabs(x4) + math.log(x4**2+x5**2+x7**2+x8**2 ) + x9 + 1./(1+x10**2)
		elif test_id == 4:
			pred = math.exp(math.fabs(x1-x2))+ math.fabs(x2*x3)-(x3**2)**math.fabs(x4)+ (x1*x4)**2 + math.log(x4**2+x5**2+x7**2+x8**2 ) + x9 + 1/(1+x10**2)
		elif test_id == 5:
			pred = 1./(1+ x1**2 + x2**2 + x3**2) + math.sqrt(math.fabs(x4 + x5)) + math.fabs(x6 + x7) + x8*x9*x10        
		elif test_id == 6:
			pred = math.exp( math.fabs( x1*x2 ) + 1 ) - math.exp( math.fabs( x3+x4 ) + 1 ) + math.cos( x5+x6-x8 ) + math.sqrt(x8**2 + x9**2 + x10**2)
		elif test_id == 7: 
			pred = (math.atan(x1)+math.atan(x2))**2 + max(x3*x4 + x6,0) - 1./(1+(x4*x5*x6*x7*x8)**2) + (math.fabs(x7) *1./ (1+math.fabs(x9)))**5 + x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10
		elif test_id == 8: 
			pred =  x1*x2 + 2**(x3+x5+x6) + 2**(x3+x4+x5+x7) + math.sin(x7*math.sin(x8+x9)) + math.acos(0.9*x10) 
		elif test_id == 9:
			pred = math.tanh(x1*x2+x3*x4)*math.sqrt(math.fabs(x5)) + math.exp(x5+x6) + math.log((x6*x7*x8)**2 + 1) + x9*x10 + 1./(1+math.fabs(x10))
		elif test_id == 10:
			pred = math.sinh(x1+x2) + math.acos(math.tanh(x3+x5+x7)) + math.cos(x4+x5) + (math.cos(x7*x9))**(-1.)

		else:
			print('invalid test id')
			assert(False)

		y.append(pred)

	X = np.array(X)
	y = np.expand_dims(np.array(y), axis=1)
	Xy = np.concatenate([X,y], axis=1)

	df = pd.DataFrame(data=Xy)

	all_folds = cv_split_data(df, list(range(df.shape[1]-1)), df.shape[1]-1, task='regression', synth=True)
	if test_id in [1]: 
		if noise >= 0:
			all_folds = add_noise(all_folds, mu=0, sigma=noise)	
			all_folds = unscale_data(all_folds)
		else:
			all_folds = scale_data(all_folds)
			all_folds = unscale_data(all_folds)

	else: 
		all_folds = scale_data(all_folds)
		if noise >= 0:
			all_folds = add_noise(all_folds, mu=0, sigma=noise)

	return all_folds



def get_high_dimensional_synthetic_data(x_dim = 1000, n_samples = 3000, density = 0.02, K = 5):
	mu, sigma = 0, 1 

	X = np.random.normal(mu, sigma, (n_samples,x_dim))
	beta = sparse.random(x_dim, 1, density=density, data_rvs=np.random.randn).todense()
	assert(K>0)
	W = 0
	for k in range(0,K):
		a_k = sparse.random(x_dim, 1, density=density, data_rvs=np.random.randn).todense()
		a_k_matmul = np.matmul(a_k, np.transpose(a_k))
		W = W + a_k_matmul

	eps = np.random.normal(mu, 0.1, (n_samples,1))

	y = []
	for i, Xi in enumerate(X):
		Xi2 = np.expand_dims(Xi, axis = 1)
		outcome = np.matmul(np.transpose(beta), Xi2) + np.matmul(np.matmul(np.transpose(Xi2), W), Xi2) + eps[i]
		y.append( outcome )#+ eps[i])

	y = np.expand_dims(np.squeeze(y),axis=1)

	dataset = np.concatenate([X,y], axis=1)
	df = pd.DataFrame.from_records(dataset)

	all_folds = cv_split_data(df, list(range(df.shape[1]-1)), df.shape[1]-1, task='regression', synth=True)
	all_folds = scale_data(all_folds)

	all_folds[0]['pairwise_weights'] = W
	all_folds[0]['main_effect_weights'] = beta

	return all_folds



def get_dataset(dataset, path, note='', n_folds=10, max_size=-1, only_fold_idx = -1, noise=0, dataset_size=30000):
	global num_folds
	global only_fold_index
	global max_dataset_size
	num_folds = n_folds

	only_fold_index = only_fold_idx
	max_dataset_size = max_size
	if dataset == 'synthetic_test_suite':
		return get_synthetic_test_suite(dataset_size, note, noise)
	if dataset == 'high_dimensional_synthetic_data':
		return get_high_dimensional_synthetic_data(n_samples = dataset_size)

	else:
		assert False


