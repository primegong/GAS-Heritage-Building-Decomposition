
import numpy as np
import os, glob
from lib.read_data.OBJ_PARSER import *


class data_PARSER(OBJ_PARSER):
	def __init__(self, model_file):
		super(data_PARSER,self).__init__(model_file)

	def get_triangles(self):
		faces = self.get_faces()
		vertices = self.get_vertices()
		for index in faces:
			triangle = [vertices[index[0]-1], vertices[index[2]-1], vertices[index[4]-1]]
			triangle = np.array(triangle).astype(np.float32)
			triangle = tuple(map(str, triangle))
			self.triangles.append(triangle)
		return self.triangles, faces, vertices


def read_3D_segments(dirpath, cate = None):
	segments = {}
	
	if not os.path.exists(dirpath):
		print(dirpath)
		return segments
	obj_files = os.listdir(dirpath)
	if cate != None:
		cate = cate.split('_')[-1]
		obj_files = [file for file in obj_files if cate in file]
	
	for obj_file in obj_files:
		if os.path.splitext(obj_file)[-1] =='.obj':

			model_file = os.path.join(dirpath, obj_file)
			parser = data_PARSER(model_file)
			triangles, faces, vertices = parser.get_triangles()
			segments[obj_file] = triangles
		
		# print('triangles number of %s is %d: '% (obj_file, len(triangles)))
	return segments



def evaluate_semantic(gt_segments, pred_segments, threshold = 0.5):
	
	pred_num = len(pred_segments.keys())
	gt_num = len(gt_segments.keys())

	print('pred_num:', pred_num)
	print('gt_num:', gt_num)


	total_precision = -np.ones((1, len(category)))
	total_recall = -np.ones((1, len(category)))
	total_iou  = -np.ones((1, len(category)))


	for obj in pred_segments.keys():

		ID = int(obj.split('_')[0])

		print('\n----------------------------------------------')

		print('obj_name:', obj)
		if obj not in gt_segments.keys() or obj not in pred_segments.keys():
			continue

		pred_triangles = pred_segments[obj]
		gt_triangles = gt_segments[obj]
	
	
		pred_triangles_num = len(pred_triangles)
		gt_triangles_num = len(gt_triangles)
		print(pred_triangles_num, gt_triangles_num)

		intersection = len(set(pred_triangles).intersection(set(gt_triangles)))
		union = len(set(pred_triangles).union(set(gt_triangles)))


		precision = float(intersection/pred_triangles_num)
		recall = float(intersection/gt_triangles_num)
		iou = float(intersection/union)

		print('intersection:', intersection)
		print('union:', union)
		print('precision：', precision)
		print('recall：',  recall)
		print('iou：', iou)

		total_precision[0, ID] = precision
		total_recall[0, ID] = recall
		total_iou[0, ID] = iou


	print('****************************************')
 
	
	return total_precision, total_recall, total_iou


	

def evaluate_instance(gt_segments, pred_segments, threshold = 0.5):

	pred_num = len(pred_segments.keys())
	gt_num = len(gt_segments.keys())

	flag = np.zeros(gt_num)
	gt_names = list(gt_segments.keys())


	# pixel_level_precision, pixel_level_recall = 0.0, 0.0
	instance_tp = 0.0
	mIoU = []
	matches = []

	for pred_name, pred_segs in pred_segments.items():

		max_iou = 0
		match_gt_name = 0
		match_intersection = 0
		match_union = 0
		
		for gt_name, gt_segs in gt_segments.items():

			intersection = len(set(pred_segs).intersection(set(gt_segs)))
			union = len(set(pred_segs).union(set(gt_segs)))

			iou = intersection*1.0/union

			if iou>max_iou:
				max_iou = iou
				match_gt_name = gt_name
				match_intersection = intersection
				match_union = union
			
		if max_iou>=threshold:
			instance_tp +=1
			ind = gt_names.index(match_gt_name)
			flag[ind] = 1

			mIoU.append(max_iou)
			matches.append((pred_name, match_gt_name))

	# instance_level_precision = instance_tp*1.0/pred_num
	# instance_level_recall = np.sum(flag)*1.0 / gt_num
	# mIoU = np.array(mIoU)
	# mIoU = np.mean(mIoU)

	recall_num = np.sum(flag)*1.0

	left_pred = set(pred_segments) - set(np.unique(matches))
	left_gt = set(gt_segments) - set(np.unique(matches))
	if len(left_gt) and len(left_pred):
		temp = []
		for left_p in left_pred:
			for left_g in left_gt:
				intersection = len(set(left_p).intersection(set(left_g)))
				union = len(set(left_p).union(set(left_g)))

				iou = intersection*1.0/union
				temp.append((left_g, left_p,iou))
		temp.sort(key = lambda  k:k[-1], reverse = True)
	
	
	return instance_tp, recall_num, mIoU, matches


def do_semantic_evaluation(category, gt_dirpath, pred_dirpath):
	model_file = os.listdir(gt_dirpath)
	num_model = len(model_file)

	total_precision = -np.ones((1, len(category)))
	total_recall = -np.ones((1, len(category)))
	total_iou  = -np.ones((1, len(category)))

	model_file=['Tile_+023_+024_noground', 'Tile_+023_+026_2_noground', 'Tile_+023_+026_4', 'Tile_+026_+028_1', 'Tile_+026_+028_7']
	for model in model_file:

		print('\ncurrent model file: ' + model  + '.....................')

		gt_segments = read_3D_segments(gt_dirpath + model)
		pred_segments = read_3D_segments(pred_dirpath +'/sem_results/' + model)

		precision, recall, iou = evaluate_semantic(gt_segments, pred_segments)
		
		total_precision  = np.vstack((total_precision, precision))
		total_recall  = np.vstack((total_recall, recall))
		total_iou = np.vstack((total_iou, iou))

	for i, cate in enumerate(category):
		if i==0:
			continue
		print('----------------------------------------------')

		cur_precision = total_precision[:,i]
		inds = np.where(cur_precision!=-1)[0]
		cur_precision = np.sum(cur_precision[inds])/len(inds)*1.0
		print('precision of ' + cate + ':' ,  cur_precision)

		cur_recall = total_recall[:,i]
		inds = np.where(cur_recall!=-1)[0]
		cur_recall = np.sum(cur_recall[inds])/len(inds)*1.0
		print('recall of ' + cate + ':' ,  cur_recall)

		cur_iou = total_iou[:,i]
		inds = np.where(cur_iou!=-1)[0]
		cur_iou = np.sum(cur_iou[inds])/len(inds)*1.0
		print('iou of ' + cate + ':',   cur_iou)
	print('****************************************\n')
 



def do_instance_evaluation(category, gt_dirpath, pred_dirpath):


	model_file = os.listdir(gt_dirpath)
	model_file=['Tile_+023_+024_noground', 'Tile_+023_+026_2_noground', 'Tile_+023_+026_4', 'Tile_+026_+028_1', 'Tile_+026_+028_7']
	num_model = len(model_file)
	
	for i, cate in enumerate(category):
		if i==0:
			continue
		print('\n----------------------------------------------' + cate)

		total_instance_tp, total_recall_num, mean_iou = 0.0, 0.0, []
		total_pred_num, total_gt_num = 0.0, 0.0
		
		for model in model_file:

			print('current model file: ' + model  + '.....................')

			gt_segments = read_3D_segments(gt_dirpath + model + '/instance', cate)

			if cate[:2] in ['1_','2_','4_']:
				pred_segments = read_3D_segments(pred_dirpath + 'instance_result/' + model + '/' + cate + '/instance', cate)
			elif cate[:2] in ['15']:
				pred_segments = read_3D_segments(pred_dirpath + 'instance_result/' + model + '/2_hip/instance', cate)
			elif cate[:2] in ['16']:
				pred_segments = read_3D_segments(pred_dirpath + 'instance_result/' + model + '/4_wall/instance', cate)
			elif cate[:2] in ['12','13','14']:
				pred_segments = read_3D_segments(pred_dirpath + 'instance_result/' + model + '/hierarchy_structures', cate)
			else:
				pred_segments = read_3D_segments(pred_dirpath + 'instance_result/' + model + '/' + cate , cate)


			if cate[:2] in ['12','13','14']:
				print(cate, pred_dirpath + 'instance_result/' + model + '/hierarchy_structures', pred_segments.keys(), gt_segments.keys())
			pred_num = len(pred_segments.keys())
			gt_num = len(gt_segments.keys())

			total_pred_num += pred_num
			total_gt_num += gt_num

			if gt_num ==0:
				##不存在
				print('gt_num of '+ cate + '==0')
				continue
			else:
				if pred_num == 0:
					print('pred_num of ' + cate + '==0')
					instance_tp = 0.0
					recall_num = 0.0
				else:
					instance_tp, recall_num, mIOU, matches = evaluate_instance(gt_segments, pred_segments)
					total_instance_tp += instance_tp
					total_recall_num += recall_num
					mean_iou.extend(mIOU)

		if total_gt_num ==0:
			continue
		if total_pred_num==0:
			continue
		mean_precision = total_instance_tp*1.0/total_pred_num
		mean_recall = total_recall_num/ total_gt_num
		mean_iou = np.array(mean_iou)
		mean_iou = np.mean(mean_iou)

		# mean_precision = mean_precision/float(num_model)
		# mean_recall = mean_recall/float(num_model)
		# mean_iou = mean_iou/float(num_model)


		print('instance_level_precision of ' + cate + ':' ,  mean_precision)
		print('instance_level_recall of ' + cate + ':' ,  mean_recall)
		print('mIOU of ' + cate + ':' ,  mean_iou)
	print('****************************************\n')

	
			




if __name__ == "__main__":
	global category
	category = ['_ignore_', '1_roofplane', '2_hip',
	'3_chimney', '4_wall', '5_window', '6_stone carving',
	'7_door', '8_glass house', '9_balcony', '10_pediment',
	'11_railing','12_gabled roof', '13_hip roof','14_pyramidal roof','15_main ridge','16_gable']

	gt_dirpath = './truth/'
	pred_dirpath = './predict/'

	# do_semantic_evaluation(category, gt_dirpath, pred_dirpath)
# 
	do_instance_evaluation(category, gt_dirpath, pred_dirpath)



	