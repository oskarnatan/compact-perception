import os
import time
import yaml
import cv2
from torch import torch, utils
import numpy as np
import pandas as pd
from collections import OrderedDict

from mylib import arch, util, function
torch.backends.cudnn.benchmark = True
device = torch.device("cuda:0")



trainmap = "Tr1" #Tr12345 Tr1 Tr2
valmap = "Va2" #Va12345 Va1 Va2
mapx = "Te2/"  #Te12345 Te1 Te2 
mod_folder = "E0_alwx/" #E0_slw E0_slwx E0_alwx
mod_dir = "model/perception_"+trainmap+valmap+"/"+mod_folder
save_dir = "prediction/"+mapx
with open("dataset/"+mapx+"data_info.yml", 'r') as f:
	test_info = yaml.load(f)
listfiles = test_info['test_idx']
listfiles.sort() 


#PRINT CONFIGURATION
print("==========================================")
print("MODEL CONFIGURATION:")
with open(mod_dir+"model_config.yml", 'r') as f:
	config = yaml.load(f)
for key in config.keys():
	print('%s: %s' % (key, str(config[key])))
with open(mod_dir+"data_info.yml", 'r') as f:
	data_info = yaml.load(f)


#SAVE DIRECTORY
save_dir = save_dir+mod_folder
os.makedirs(save_dir, exist_ok=True) 


# LOAD ARSITEKTUR DAN WEIGHTS MODEL
model = arch.E0(in_channel_dim=config['tensor_dim'][1])
model.double().to(device) 
model.load_state_dict(torch.load(mod_dir+"model_weights.pth", map_location=device))
model.eval()



inputd = util.datagen( 
	file_ids=listfiles,
	input_dir = "dataset/"+mapx[:2]+"Set/",
	config=config)#,
data = utils.data.DataLoader(inputd,
	batch_size=1, 
	shuffle=False, 
	num_workers=4,
	drop_last=False)


#load loss weights
params_lw = config['loss_weights']
lossf = [function.HuberLoss().to(device), function.BCEDiceLoss().to(device), function.BCEDiceLoss().to(device), function.BCEDiceLoss().to(device)]
metricf = [function.L1Loss().to(device), function.IOUScore().to(device), function.IOUScore().to(device), function.IOUScore().to(device)]
batch = 0
log = OrderedDict([
	('batch', []),
	('forwardpass_time', []),
	('test_total_metric', []),
	('test_depth_metric', []),
	('test_seg_metric', []),
	('test_lidseg_metric', []),
	('test_bir_metric', []),
	('test_total_loss', []),
	('test_depth_loss', []),
	('test_seg_loss', []),
	('test_lidseg_loss', []),
	('test_bir_loss', [])])
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


with torch.no_grad():
	for input_x, batch_Y_true, img_id in data:
		file_name = img_id['img_id']
		for i in range(len(config['input'])):
			input_x[i] = input_x[i].to(device)#cuda()
		for i in range(len(config['task'])):
			batch_Y_true[i] = batch_Y_true[i].to(device)#cuda()
		
		#INFERENCE
		start_time = time.time()
		batch_Y_pred = model(input_x)
		infer_time = time.time() - start_time
			
		#DEPTH
		tot_depth_loss = params_lw[0] * lossf[0](batch_Y_pred[0], batch_Y_true[0])
		tot_depth_loss = tot_depth_loss + (params_lw[0] * lossf[0](batch_Y_pred[1], batch_Y_true[1]))
		tot_depth_loss = tot_depth_loss + (params_lw[0] * lossf[0](batch_Y_pred[2], batch_Y_true[2]))
		tot_depth_loss = (tot_depth_loss + (params_lw[0] * lossf[0](batch_Y_pred[3], batch_Y_true[3]))) / 4 #4 views
		tot_depth_metric = metricf[0](batch_Y_pred[0], batch_Y_true[0])
		tot_depth_metric = tot_depth_metric + metricf[0](batch_Y_pred[1], batch_Y_true[1])
		tot_depth_metric = tot_depth_metric + metricf[0](batch_Y_pred[2], batch_Y_true[2])
		tot_depth_metric = (tot_depth_metric + metricf[0](batch_Y_pred[3], batch_Y_true[3])) / 4  #4 views
		#SEG
		tot_seg_loss = params_lw[1] * lossf[1](batch_Y_pred[4], batch_Y_true[4])
		tot_seg_loss = tot_seg_loss + (params_lw[1] * lossf[1](batch_Y_pred[5], batch_Y_true[5]))
		tot_seg_loss = tot_seg_loss + (params_lw[1] * lossf[1](batch_Y_pred[6], batch_Y_true[6]))
		tot_seg_loss = (tot_seg_loss + (params_lw[1] * lossf[1](batch_Y_pred[7], batch_Y_true[7]))) / 4  #4 views
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
		#ALL
		total_loss = tot_depth_loss + tot_seg_loss + tot_lidseg_loss + tot_bir_loss
		total_metric = tot_depth_metric + (1 - tot_seg_metric) + (1 - tot_lidseg_metric) + (1 - tot_bir_metric)

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
		log['batch'].append(batch)
		log['forwardpass_time'].append(infer_time)
		log['test_total_loss'].append(score['total_loss'].avg)
		log['test_total_metric'].append(score['total_metric'].avg)
		log['test_depth_loss'].append(score['tot_depth_loss'].avg)
		log['test_depth_metric'].append(score['tot_depth_metric'].avg)
		log['test_seg_loss'].append(score['tot_seg_loss'].avg)
		log['test_seg_metric'].append(score['tot_seg_metric'].avg)
		log['test_lidseg_loss'].append(score['tot_lidseg_loss'].avg)
		log['test_lidseg_metric'].append(score['tot_lidseg_metric'].avg)
		log['test_bir_loss'].append(score['tot_bir_loss'].avg)
		log['test_bir_metric'].append(score['tot_bir_metric'].avg)
		batch += 1
			
		#detach tensor
		for i in range(len(config['task'])):
			batch_Y_pred[i] = batch_Y_pred[i].cpu().detach().numpy() 

		#loop batch
		for batch_i in range(1):
			print(file_name[batch_i])
			#LOOP TASK
			for task_i in range(len(config['task'])):
				task_folder = save_dir+"pred_"+config['task'][task_i]+"/"
				os.makedirs(task_folder, exist_ok=True)
				if config['task'][task_i][:3] == 'dep': #depth estimation
					pred_dep = batch_Y_pred[task_i][batch_i].transpose(1,2,0)
					pred_dep = np.round(pred_dep, decimals=2)
					norm_dep = pred_dep * 255.0 #normalization 0 - 255
					cv2.imwrite(task_folder+file_name[batch_i], norm_dep) 
				
				elif config['task'][task_i][:3] == 'seg': #segmentation
					imgx = np.zeros((config['tensor_dim'][2], config['tensor_dim'][3], 3))
					pred_seg = batch_Y_pred[task_i][batch_i]
					inx = np.argmax(pred_seg, axis=0)
					for cmap in data_info['seg_colors']:
						cmap_id = data_info['seg_colors'].index(cmap)
						imgx[np.where(inx == cmap_id)] = cmap
					imgx = util.swap_RGB2BGR(imgx)
					cv2.imwrite(task_folder+file_name[batch_i], imgx) 
		
				elif config['task'][task_i][:3] == 'lid': #lidar segmentation
					imgx = np.zeros((config['tensor_dim'][2], config['tensor_dim'][3], 3))
					pred_seg = batch_Y_pred[task_i][batch_i]
					pred_seg = np.round(pred_seg)
					col_mask = []
					for i in range(data_info['n_seg_class']):
						img_blank = util.blank_frame(configx=config, rgb_color=data_info['seg_colors'][i], target_dim=[config['tensor_dim'][2], config['tensor_dim'][3]])
						col_mask.append(img_blank)
					col_mask = np.array(col_mask)
					#looping class
					for cls_i in range(pred_seg.shape[0]):
						pred_mask = pred_seg[cls_i]
						pred_mask = np.expand_dims(pred_mask, axis=-1) 
						col_mask[cls_i][:,:,0:1] = col_mask[cls_i][:,:,0:1] * pred_mask 
						col_mask[cls_i][:,:,1:2] = col_mask[cls_i][:,:,1:2] * pred_mask 
						col_mask[cls_i][:,:,2:3] = col_mask[cls_i][:,:,2:3] * pred_mask 
						imgx = imgx+col_mask[cls_i]
					imgx = util.swap_RGB2BGR(imgx)
					cv2.imwrite(task_folder+file_name[batch_i], imgx) 		

				elif config['task'][task_i][:3] == 'bir': #birdview
					imgn = np.zeros((config['tensor_dim'][2], config['tensor_dim'][3], 3))
					pred_bird = batch_Y_pred[task_i][batch_i]
					pred_bird = np.round(pred_bird)
					new_mask = []
					for i in range(data_info['n_bird_class']):
						img_blank = util.blank_frame(configx=config, rgb_color=data_info['bird_colors'][i], target_dim=[config['tensor_dim'][2], config['tensor_dim'][3]])
						new_mask.append(img_blank)
					new_mask = np.array(new_mask)
					for cls_i in range(pred_bird.shape[0]):
						pred_mask = pred_bird[cls_i]
						pred_mask = np.expand_dims(pred_mask, axis=-1) 
						new_mask[cls_i][:,:,0:1] = new_mask[cls_i][:,:,0:1] * pred_mask 
						new_mask[cls_i][:,:,1:2] = new_mask[cls_i][:,:,1:2] * pred_mask 
						new_mask[cls_i][:,:,2:3] = new_mask[cls_i][:,:,2:3] * pred_mask 
						imgn = imgn+new_mask[cls_i]
					imgn = util.swap_RGB2BGR(imgn)

					cv2.imwrite(task_folder+file_name[batch_i], imgn) 

		#paste csv file
		pd.DataFrame(log).to_csv(save_dir+'/test_performance.csv', index=False)

#AVG
log['batch'].append("avg_total")
log['forwardpass_time'].append(np.mean(log['forwardpass_time']))
log['test_total_loss'].append(np.mean(log['test_total_loss']))
log['test_total_metric'].append(np.mean(log['test_total_metric']))
log['test_depth_loss'].append(np.mean(log['test_depth_loss']))
log['test_depth_metric'].append(np.mean(log['test_depth_metric']))
log['test_seg_loss'].append(np.mean(log['test_seg_loss']))
log['test_seg_metric'].append(np.mean(log['test_seg_metric']))
log['test_lidseg_loss'].append(np.mean(log['test_lidseg_loss']))
log['test_lidseg_metric'].append(np.mean(log['test_lidseg_metric']))
log['test_bir_loss'].append(np.mean(log['test_bir_loss']))
log['test_bir_metric'].append(np.mean(log['test_bir_metric']))

#VARIANCE
log['batch'].append("var_total")
log['forwardpass_time'].append(np.var(log['forwardpass_time'][:-1]))
log['test_total_loss'].append(np.var(log['test_total_loss'][:-1]))
log['test_total_metric'].append(np.var(log['test_total_metric'][:-1]))
log['test_depth_loss'].append(np.var(log['test_depth_loss'][:-1]))
log['test_depth_metric'].append(np.var(log['test_depth_metric'][:-1]))
log['test_seg_loss'].append(np.var(log['test_seg_loss'][:-1]))
log['test_seg_metric'].append(np.var(log['test_seg_metric'][:-1]))
log['test_lidseg_loss'].append(np.var(log['test_lidseg_loss'][:-1]))
log['test_lidseg_metric'].append(np.var(log['test_lidseg_metric'][:-1]))
log['test_bir_loss'].append(np.var(log['test_bir_loss'][:-1]))
log['test_bir_metric'].append(np.var(log['test_bir_metric'][:-1]))

pd.DataFrame(log).to_csv(save_dir+'/test_performance.csv', index=False)

