#python train.py 

import os
import sys
import time
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import OrderedDict
from torch import torch, nn, optim, utils

from mylib import arch, function, util
torch.backends.cudnn.benchmark = True
device = torch.device("cuda:0")


trainmap = 'Tr1' # Tr12345 Tr1 Tr2
valmap = 'Va2' # Va12345 Va2 Va1
config = {
	'data_dir'			: ['dataset/'+trainmap[:2]+'Set/', 'dataset/'+valmap[:2]+'Set/'], 
	'data_split_info'	: ['dataset/'+trainmap+'/data_info.yml', 'dataset/'+valmap+'/data_info.yml'], 
	'input'				: ['dvs_f', 'dvs_l', 'dvs_ri', 'dvs_r', 'rgb_f', 'rgb_l', 'rgb_ri', 'rgb_r', 'lid_depth_top'],
	'task'				: ['depth_f', 'depth_l', 'depth_ri', 'depth_r', 'segmentation_f', 'segmentation_l', 'segmentation_ri', 'segmentation_r', 'lid_seg_top', 'bird_view',],
	'mod_dir'			: 'model/perception_'+trainmap+valmap+'/',
	'arch'				: 'E0', #E0 For CARLA, E1 for NuScene
	'tensor_dim'		: [6, [2, 3, 15], 128, 128], #format pytorch: batch_size x (channel DVS, RGB, lidar) x H x W
	'adaptive_lw'		: True,
	'loss_weights'		: [1, 1, 1, 1], #INITIAL LW DE, SS, BEVP, LS
	'bottleneck'		: [168, 530], #check BOTTLENECK
	'stop_counter'		: 25, #early stopping
	}

#load data split info
with open(config['data_split_info'][0], 'r') as g:
	info = yaml.load(g)
with open(config['data_split_info'][1], 'r') as g:
	info_val = yaml.load(g)



config['mod_dir'] += config['arch'] 
if config['adaptive_lw']:
	config['mod_dir'] += "_alw"
else:
	config['mod_dir'] += "_slw"
if config['tensor_dim'][1][2] != 1: 
	config['mod_dir'] += "x"
os.makedirs(config['mod_dir'], exist_ok=True) 



#fungsi renormalize loss weights 
def renormalize_params_lw(current_lw):
	lw = np.array([tens.cpu().detach().numpy() for tens in current_lw])
	lws = np.array([lw[i][0] for i in range(len(lw))])
	coef = np.array(config['loss_weights']).sum()/lws.sum()
	new_lws = [coef*lw for lw in lws]
	normalized_lws = [torch.cuda.FloatTensor([lw]).clone().detach().requires_grad_(True) for lw in new_lws]
	return normalized_lws


#TRAINING
lgrad_func = nn.L1Loss().to(device) #untuk mengkomputasi lgrad pada paper gradnorm
def train(batches, model, lossf, metricf, config, optimizer, params_lw, optimizer_lw):
	score = {'total_loss': util.AverageMeter(),
			'total_metric': util.AverageMeter(),
			'tot_depth_loss': util.AverageMeter(),
			'tot_depth_metric': util.AverageMeter(),
			'tot_seg_loss': util.AverageMeter(),
			'tot_seg_metric': util.AverageMeter(),
			'tot_lidseg_loss': util.AverageMeter(),
			'tot_lidseg_metric': util.AverageMeter(),
			'tot_bir_loss': util.AverageMeter(),
			'tot_bir_metric': util.AverageMeter()}
	model.train()

	prog_bar = tqdm(total=len(batches))
	#training....
	total_batch = len(batches)
	batch_ke = 1
	for batch_X, batch_Y_true, _ in batches:
		for i in range(len(config['input'])):
			batch_X[i] = batch_X[i].to(device)
		for i in range(len(config['task'])):
			batch_Y_true[i] = batch_Y_true[i].to(device)

		#forward propagation....
		batch_Y_pred = model(batch_X)

		#DEPTH
		tot_depth_loss = params_lw[0] * lossf[0](batch_Y_pred[0], batch_Y_true[0])
		tot_depth_loss = tot_depth_loss + (params_lw[0] * lossf[0](batch_Y_pred[1], batch_Y_true[1]))
		tot_depth_loss = tot_depth_loss + (params_lw[0] * lossf[0](batch_Y_pred[2], batch_Y_true[2]))
		tot_depth_loss = (tot_depth_loss + (params_lw[0] * lossf[0](batch_Y_pred[3], batch_Y_true[3]))) / 4
		tot_depth_metric = metricf[0](batch_Y_pred[0], batch_Y_true[0])
		tot_depth_metric = tot_depth_metric + metricf[0](batch_Y_pred[1], batch_Y_true[1])
		tot_depth_metric = tot_depth_metric + metricf[0](batch_Y_pred[2], batch_Y_true[2])
		tot_depth_metric = (tot_depth_metric + metricf[0](batch_Y_pred[3], batch_Y_true[3])) / 4  

		#SEG
		tot_seg_loss = params_lw[1] * lossf[1](batch_Y_pred[4], batch_Y_true[4])
		tot_seg_loss = tot_seg_loss + (params_lw[1] * lossf[1](batch_Y_pred[5], batch_Y_true[5]))
		tot_seg_loss = tot_seg_loss + (params_lw[1] * lossf[1](batch_Y_pred[6], batch_Y_true[6]))
		tot_seg_loss = (tot_seg_loss + (params_lw[1] * lossf[1](batch_Y_pred[7], batch_Y_true[7]))) / 4 
		tot_seg_metric = metricf[1](batch_Y_pred[4], batch_Y_true[4])
		tot_seg_metric = tot_seg_metric + metricf[1](batch_Y_pred[5], batch_Y_true[5])
		tot_seg_metric = tot_seg_metric + metricf[1](batch_Y_pred[6], batch_Y_true[6])
		tot_seg_metric = (tot_seg_metric + metricf[1](batch_Y_pred[7], batch_Y_true[7])) / 4

		#LIDSEG
		tot_lidseg_loss = params_lw[2] * lossf[2](batch_Y_pred[8], batch_Y_true[8])
		tot_lidseg_metric = metricf[2](batch_Y_pred[8], batch_Y_true[8])

		#BIRDVIEW
		tot_bir_loss = params_lw[3] * lossf[3](batch_Y_pred[9], batch_Y_true[9])
		tot_bir_metric = metricf[3](batch_Y_pred[9], batch_Y_true[9])

		#TOTAL LOSS
		total_loss = tot_depth_loss + tot_seg_loss + tot_lidseg_loss + tot_bir_loss
		total_metric = tot_depth_metric + (1 - tot_seg_metric) + (1 - tot_lidseg_metric) + (1 - tot_bir_metric)

		optimizer.zero_grad()

		if batch_ke == 1:
			total_loss.backward()
			#FIRST LOSS
			tot_depth_loss_0 = torch.clone(tot_depth_loss)
			tot_seg_loss_0 = torch.clone(tot_seg_loss)
			tot_lidseg_loss_0 = torch.clone(tot_lidseg_loss)
			tot_bir_loss_0 = torch.clone(tot_bir_loss)

		elif 1 < batch_ke < total_batch:
			total_loss.backward() 

		elif batch_ke == total_batch: 
			if config['adaptive_lw']:
				optimizer_lw.zero_grad()
				total_loss.backward(retain_graph=True) 

				params = list(filter(lambda p: p.requires_grad, model.parameters()))
				G1R = torch.autograd.grad(tot_depth_loss, params[config['bottleneck'][0]], retain_graph=True, create_graph=True)
				G1 = torch.norm(G1R[0], keepdim=True)
				G2R = torch.autograd.grad(tot_seg_loss, params[config['bottleneck'][0]], retain_graph=True, create_graph=True)
				G2 = torch.norm(G2R[0], keepdim=True)
				G3R = torch.autograd.grad(tot_lidseg_loss, params[config['bottleneck'][0]], retain_graph=True, create_graph=True)
				G3 = torch.norm(G3R[0], keepdim=True)
				G4R = torch.autograd.grad(tot_bir_loss, params[config['bottleneck'][1]], retain_graph=True, create_graph=True)
				G4 = torch.norm(G4R[0], keepdim=True)
				G_avg = (G1+G2+G3+G4) / len(config['loss_weights'])

				tot_depth_loss_hat = tot_depth_loss / tot_depth_loss_0
				tot_seg_loss_hat = tot_seg_loss / tot_seg_loss_0
				tot_lidseg_loss_hat = tot_lidseg_loss / tot_lidseg_loss_0
				tot_bir_loss_hat = tot_bir_loss / tot_bir_loss_0
				tot_loss_hat_avg = (tot_depth_loss_hat + tot_seg_loss_hat + tot_lidseg_loss_hat + tot_bir_loss_hat) / len(config['loss_weights'])

				inv_rate_dep = tot_depth_loss_hat / tot_loss_hat_avg
				inv_rate_seg = tot_seg_loss_hat / tot_loss_hat_avg
				inv_rate_lidseg = tot_lidseg_loss_hat / tot_loss_hat_avg
				inv_rate_bir = tot_bir_loss_hat / tot_loss_hat_avg

				C1 = G_avg*(inv_rate_dep)**1.5
				C2 = G_avg*(inv_rate_seg)**1.5
				C3 = G_avg*(inv_rate_lidseg)**1.5
				C4 = G_avg*(inv_rate_bir)**1.5
				C1 = C1.detach()
				C2 = C2.detach()
				C3 = C3.detach()
				C4 = C4.detach()

				Lgrad = lgrad_func(G1, C1) + lgrad_func(G2, C2) + lgrad_func(G3, C3) + lgrad_func(G4, C4)
				Lgrad.backward()
				optimizer_lw.step()
				new_param_lw = optimizer_lw.param_groups[0]['params']

			else:
				total_loss.backward()
				new_param_lw = 0
			
		optimizer.step() 

		score['total_loss'].update(total_loss.item(), 1) 
		score['total_metric'].update(total_metric.item(), 1) 
		score['tot_depth_loss'].update(tot_depth_loss.item(), 1) 
		score['tot_depth_metric'].update(tot_depth_metric.item(), 1) 
		score['tot_seg_loss'].update(tot_seg_loss.item(), 1) 
		score['tot_seg_metric'].update(tot_seg_metric.item(), 1) 
		score['tot_lidseg_loss'].update(tot_lidseg_loss.item(), 1) 
		score['tot_lidseg_metric'].update(tot_lidseg_metric.item(), 1) 
		score['tot_bir_loss'].update(tot_bir_loss.item(), 1) 
		score['tot_bir_metric'].update(tot_bir_metric.item(), 1) 

		postfix = OrderedDict([('t_total_l', score['total_loss'].avg),
							('t_total_m', score['total_metric'].avg),
							('t_dep_l', score['tot_depth_loss'].avg),
							('t_dep_m', score['tot_depth_metric'].avg),
							('t_seg_l', score['tot_seg_loss'].avg),
							('t_seg_m', score['tot_seg_metric'].avg),
							('t_lidseg_l', score['tot_lidseg_loss'].avg),
							('t_lidseg_m', score['tot_lidseg_metric'].avg),
							('t_bir_l', score['tot_bir_loss'].avg),
							('t_bir_m', score['tot_bir_metric'].avg)])


		prog_bar.set_postfix(postfix)
		prog_bar.update(1)
		batch_ke += 1 

	prog_bar.close()

	#return value
	return postfix, new_param_lw



#VALIDATION
def validate(batches, model, lossf, metricf, config, params_lw):
	score = {'total_loss': util.AverageMeter(),
			'total_metric': util.AverageMeter(),
			'tot_depth_loss': util.AverageMeter(),
			'tot_depth_metric': util.AverageMeter(),
			'tot_seg_loss': util.AverageMeter(),
			'tot_seg_metric': util.AverageMeter(),
			'tot_lidseg_loss': util.AverageMeter(),
			'tot_lidseg_metric': util.AverageMeter(),
			'tot_bir_loss': util.AverageMeter(),
			'tot_bir_metric': util.AverageMeter()}
	model.eval()
	
	with torch.no_grad():
		prog_bar = tqdm(total=len(batches))
		for batch_X, batch_Y_true, _ in batches:
			for i in range(len(config['input'])):
				batch_X[i] = batch_X[i].to(device)
			for i in range(len(config['task'])):
				batch_Y_true[i] = batch_Y_true[i].to(device)

			#forward propagation....
			batch_Y_pred = model(batch_X)

			#DEPTH
			tot_depth_loss = params_lw[0] * lossf[0](batch_Y_pred[0], batch_Y_true[0])
			tot_depth_loss = tot_depth_loss + (params_lw[0] * lossf[0](batch_Y_pred[1], batch_Y_true[1]))
			tot_depth_loss = tot_depth_loss + (params_lw[0] * lossf[0](batch_Y_pred[2], batch_Y_true[2]))
			tot_depth_loss = (tot_depth_loss + (params_lw[0] * lossf[0](batch_Y_pred[3], batch_Y_true[3]))) / 4 
			tot_depth_metric = metricf[0](batch_Y_pred[0], batch_Y_true[0])
			tot_depth_metric = tot_depth_metric + metricf[0](batch_Y_pred[1], batch_Y_true[1])
			tot_depth_metric = tot_depth_metric + metricf[0](batch_Y_pred[2], batch_Y_true[2])
			tot_depth_metric = (tot_depth_metric + metricf[0](batch_Y_pred[3], batch_Y_true[3])) / 4  

			#SEG
			tot_seg_loss = params_lw[1] * lossf[1](batch_Y_pred[4], batch_Y_true[4])
			tot_seg_loss = tot_seg_loss + (params_lw[1] * lossf[1](batch_Y_pred[5], batch_Y_true[5]))
			tot_seg_loss = tot_seg_loss + (params_lw[1] * lossf[1](batch_Y_pred[6], batch_Y_true[6]))
			tot_seg_loss = (tot_seg_loss + (params_lw[1] * lossf[1](batch_Y_pred[7], batch_Y_true[7]))) / 4  
			tot_seg_metric = metricf[1](batch_Y_pred[4], batch_Y_true[4])
			tot_seg_metric = tot_seg_metric + metricf[1](batch_Y_pred[5], batch_Y_true[5])
			tot_seg_metric = tot_seg_metric + metricf[1](batch_Y_pred[6], batch_Y_true[6])
			tot_seg_metric = (tot_seg_metric + metricf[1](batch_Y_pred[7], batch_Y_true[7])) / 4

			#LIDSEG
			tot_lidseg_loss = params_lw[2] * lossf[2](batch_Y_pred[8], batch_Y_true[8])
			tot_lidseg_metric = metricf[2](batch_Y_pred[8], batch_Y_true[8])

			#BIRDVIEW
			tot_bir_loss = params_lw[3] * lossf[3](batch_Y_pred[9], batch_Y_true[9])
			tot_bir_metric = metricf[3](batch_Y_pred[9], batch_Y_true[9])

			#TOTAL LOSS
			total_loss = tot_depth_loss + tot_seg_loss + tot_lidseg_loss + tot_bir_loss
			total_metric = tot_depth_metric + (1 - tot_seg_metric) + (1 - tot_lidseg_metric) + (1 - tot_bir_metric)
			

			#LOG
			score['total_loss'].update(total_loss.item(), 1) 
			score['total_metric'].update(total_metric.item(), 1) 
			score['tot_depth_loss'].update(tot_depth_loss.item(), 1) 
			score['tot_depth_metric'].update(tot_depth_metric.item(), 1) 
			score['tot_seg_loss'].update(tot_seg_loss.item(), 1)
			score['tot_seg_metric'].update(tot_seg_metric.item(), 1) 
			score['tot_lidseg_loss'].update(tot_lidseg_loss.item(), 1) 
			score['tot_lidseg_metric'].update(tot_lidseg_metric.item(), 1) 
			score['tot_bir_loss'].update(tot_bir_loss.item(), 1) 
			score['tot_bir_metric'].update(tot_bir_metric.item(), 1) 

			#update progress bar
			postfix = OrderedDict([('v_total_l', score['total_loss'].avg),
								('v_total_m', score['total_metric'].avg),
								('v_dep_l', score['tot_depth_loss'].avg),
								('v_dep_m', score['tot_depth_metric'].avg),
								('v_seg_l', score['tot_seg_loss'].avg),
								('v_seg_m', score['tot_seg_metric'].avg),
								('v_lidseg_l', score['tot_lidseg_loss'].avg),
								('v_lidseg_m', score['tot_lidseg_metric'].avg),
								('v_bir_l', score['tot_bir_loss'].avg),
								('v_bir_m', score['tot_bir_metric'].avg)])


			prog_bar.set_postfix(postfix)
			prog_bar.update(1)
		prog_bar.close() 

	return postfix



#MAIN FUNCTION
def main():
	#IMPORT MODEL
	if config['arch'] == 'E0': 
		model = arch.E0(in_channel_dim=config['tensor_dim'][1]) 
	elif config['arch'] == 'E1': 
		model = arch.E1(in_channel_dim=config['tensor_dim'][1]) 
	else:
		sys.exit("ERROR, ARCH NOT FOUND............................")
	model.double().to(device) 


	lowest_monitored_score = float('inf')
	stop_count = config['stop_counter']

	#LOSS FUNCTION & METRIC FUNCTION, DE, SS, LS, BEVP
	lossf = [function.HuberLoss().to(device), function.BCEDiceLoss().to(device), function.BCEDiceLoss().to(device), function.BCEDiceLoss().to(device)]
	metricf = [function.L1Loss().to(device), function.IOUScore().to(device), function.IOUScore().to(device), function.IOUScore().to(device)]


	#OPTIMIZER
	params = filter(lambda p: p.requires_grad, model.parameters())
	optima = optim.SGD(params, lr=0.1, momentum=0.9, weight_decay=0.0001)
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optima, mode='min', factor=0.5, patience=4, min_lr=0.00001)

	#LW MGN OPTIMIZER
	if config['adaptive_lw']:
		params_lw = [torch.cuda.FloatTensor([config['loss_weights'][i]]).clone().detach().requires_grad_(True) for i in range(len(config['loss_weights']))]
		optima_lw = optim.SGD(params_lw, lr=0.1) 
		scheduler_lw = optim.lr_scheduler.ReduceLROnPlateau(optima_lw, mode='min', factor=0.5, patience=4, min_lr=0.0001)
	else:
		optima_lw = None
		scheduler_lw = None



	#LOAD DATASET INDEX
	if info['train_idx'] != None:
		train_ids = info['train_idx']
	else:
		if info['val_idx'] != None:
			train_ids = info['val_idx']
		else:
			train_ids = info['test_idx']
	if info_val['val_idx'] != None:
		val_ids = info_val['val_idx']
	else:
		if info_val['test_idx'] != None:
			val_ids = info_val['test_idx']
		else:
			val_ids = info_val['train_idx']	
	#save index
	sizeval = len(val_ids) / (len(val_ids) + len(train_ids))
	total_data = info['n_total'] + info_val['n_total']
	data_idx_dict = {
		'size_val'		: sizeval,
		'train_idx'		: train_ids,
		'val_idx'		: val_ids,
		'n_train'		: len(train_ids),
		'n_val'			: len(val_ids),
		'n_seg_class'	: info['n_seg_class'],
		'n_total_trainval'	: total_data,
		'seg_classes'	: info['seg_classes'],
		'seg_colors'	: info['seg_colors'],
		'n_bird_class'	: info['n_bird_class'],
		'bird_classes'	: info['bird_classes'],
		'bird_colors'	: info['bird_colors'],
	}
	
	#DATA LOADER
	train_dataset = util.datagen(
		file_ids=train_ids,
		config=config,
		input_dir=config['data_dir'][0])
	train_batches = utils.data.DataLoader(train_dataset,
		batch_size=config['tensor_dim'][0],
		shuffle=True,
		num_workers=4,
		drop_last=False)
	val_dataset = util.datagen( 
		file_ids=val_ids,
		config=config,
		input_dir=config['data_dir'][1])
	val_batches = utils.data.DataLoader(val_dataset,
		batch_size=config['tensor_dim'][0],
		shuffle=False,
		num_workers=4,
		drop_last=False)


	#SAVE TRAINING CONFIG AND DATA CONFIG
	with open(config['mod_dir']+'/model_config.yml', 'w') as f:
		yaml.dump(config, f)
	with open(config['mod_dir']+'/data_info.yml', 'w') as d:
		yaml.dump(data_idx_dict, d)


	#TRAINING
	#LOG FILE
	log = OrderedDict([
		('epoch', []),
		('lrate', []),
		('train_total_loss', []),
		('val_total_loss', []),
		('train_total_metric', []),
		('val_total_metric', []),
		('train_depth_loss', []),
		('val_depth_loss', []),
		('train_depth_metric', []),
		('val_depth_metric', []),
		('train_seg_loss', []),
		('val_seg_loss', []),
		('train_seg_metric', []),
		('val_seg_metric', []),
		('train_lidseg_loss', []),
		('val_lidseg_loss', []),
		('train_lidseg_metric', []),
		('val_lidseg_metric', []),
		('train_bir_loss', []),
		('val_bir_loss', []),
		('train_bir_metric', []),
		('val_bir_metric', []),
		('lrate_lw', []),
		('lw_depth', []),
		('lw_seg', []),
		('lw_lidseg', []),
		('lw_bir', []),
		('best_model', []),
		('stop_counter', []),
		('elapsed_time', []),
	])
	
	for epoch in range(99999999):
		print('\n=======---=======---=======Epoch:%.4d=======---=======---=======' % (epoch+1))

		if config['adaptive_lw']:
			curr_lw = optima_lw.param_groups[0]['params']
			lw = np.array([tens.cpu().detach().numpy() for tens in curr_lw])
			lws = np.array([lw[i][0] for i in range(len(lw))])
			current_lr_lw = optima_lw.param_groups[0]['lr']
			print("current lr untuk lw updater: ", current_lr_lw)
		else:
			curr_lw = config['loss_weights']
			lws = config['loss_weights']
			current_lr_lw = 0

		print("current loss weights: ", lws)
		current_lr = optima.param_groups[0]['lr']
		print("current lr untuk training: ", current_lr)

		#training
		start_time = time.time() 
		train_log, new_params_lw = train(batches=train_batches, model=model,
			lossf=lossf, metricf=metricf, config=config, optimizer=optima,
			params_lw=curr_lw, optimizer_lw=optima_lw)
		#validation
		val_log = validate(batches=val_batches, model=model,
			lossf=lossf, metricf=metricf, config=config,
			params_lw=curr_lw)
		if config['adaptive_lw']:
			optima_lw.param_groups[0]['params'] = renormalize_params_lw(new_params_lw)
			scheduler_lw.step(val_log['v_total_m']) 
		#update learning rate
		scheduler.step(val_log['v_total_m']) 
		elapsed_time = time.time() - start_time 


		log['epoch'].append(epoch+1)
		log['lrate'].append(current_lr)
		log['train_total_loss'].append(train_log['t_total_l'])
		log['val_total_loss'].append(val_log['v_total_l'])
		log['train_total_metric'].append(train_log['t_total_m'])
		log['val_total_metric'].append(val_log['v_total_m'])
		log['train_depth_loss'].append(train_log['t_dep_l'])
		log['val_depth_loss'].append(val_log['v_dep_l'])
		log['train_depth_metric'].append(train_log['t_dep_m'])
		log['val_depth_metric'].append(val_log['v_dep_m'])
		log['train_seg_loss'].append(train_log['t_seg_l'])
		log['val_seg_loss'].append(val_log['v_seg_l'])
		log['train_seg_metric'].append(train_log['t_seg_m'])
		log['val_seg_metric'].append(val_log['v_seg_m'])
		log['train_lidseg_loss'].append(train_log['t_lidseg_l'])
		log['val_lidseg_loss'].append(val_log['v_lidseg_l'])
		log['train_lidseg_metric'].append(train_log['t_lidseg_m'])
		log['val_lidseg_metric'].append(val_log['v_lidseg_m'])
		log['train_bir_loss'].append(train_log['t_bir_l'])
		log['val_bir_loss'].append(val_log['v_bir_l'])
		log['train_bir_metric'].append(train_log['t_bir_m'])
		log['val_bir_metric'].append(val_log['v_bir_m'])
		log['lrate_lw'].append(current_lr_lw)
		log['lw_depth'].append(lws[0])
		log['lw_seg'].append(lws[1])
		log['lw_lidseg'].append(lws[2])
		log['lw_bir'].append(lws[3])
		log['elapsed_time'].append(elapsed_time)


		print('| t_total_l: %.4f | t_total_m: %.4f | t_dep_l: %.4f | t_dep_m: %.4f | t_seg_l: %.4f | t_seg_m: %.4f | t_lidseg_l: %.4f | t_lidseg_m: %.4f |t_bir_l: %.4f | t_bir_m: %.4f |'
			% (train_log['t_total_l'], train_log['t_total_m'], train_log['t_dep_l'], train_log['t_dep_m'], train_log['t_seg_l'], train_log['t_seg_m'], train_log['t_lidseg_l'], train_log['t_lidseg_m'], train_log['t_bir_l'], train_log['t_bir_m']))
		print('| v_total_l: %.4f | v_total_m: %.4f | v_dep_l: %.4f | v_dep_m: %.4f | v_seg_l: %.4f | v_seg_m: %.4f | v_lidseg_l: %.4f | v_lidseg_m: %.4f | v_bir_l: %.4f | v_bir_m: %.4f |'
			% (val_log['v_total_l'], val_log['v_total_m'], val_log['v_dep_l'], val_log['v_dep_m'], val_log['v_seg_l'], val_log['v_seg_m'], val_log['v_lidseg_l'], val_log['v_lidseg_m'], val_log['v_bir_l'], val_log['v_bir_m']))
		print('elapsed time: %.4f sec' % (elapsed_time))
		
		
		#save model best only
		if val_log['v_total_m'] < lowest_monitored_score:
			print('v_total_m'+": %.4f < best previous: %.4f" % (val_log['v_total_m'], lowest_monitored_score))
			print("model saved!")
			torch.save(model.state_dict(),config['mod_dir']+'/model_weights.pth')
			lowest_monitored_score = val_log['v_total_m']
			stop_count = config['stop_counter']
			print("stop counter reset: ", config['stop_counter'])
			log['best_model'].append("BEST")
		else:
			stop_count -= 1
			print('v_total_m'+": %.4f >= best previous: %.4f, training stop in %d epoch" % (val_log['v_total_m'], lowest_monitored_score, stop_count))
			print("model not saved!")
			log['best_model'].append("")


		#update stop counter
		log['stop_counter'].append(stop_count)
		#paste to csv file
		pd.DataFrame(log).to_csv(config['mod_dir']+'/model_log.csv', index=False)

		# early stopping 
		if stop_count==0:
			print("EARLY STOPPED")
			break #break for loop
		
		torch.cuda.empty_cache()

if __name__ == "__main__":
	main()






