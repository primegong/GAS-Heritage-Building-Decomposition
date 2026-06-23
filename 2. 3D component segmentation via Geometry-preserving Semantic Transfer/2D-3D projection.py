import os
import numpy as np
import colorsys,random
from read_data.OBJ_PARSER import *
from read_data.XML_PARSER import *
from read_data.MASK_PARSER import *
from projection_2D_3D.projection import *
from Trianglulation.triangulation import *

from displayInstances.display import display_instances
from cluster.region_growing import regionGrowing
from boundary.boundary import generate_boundary

classes = open('labels_simple.txt','r').readlines()
classes = [cl.strip() for cl in classes]

pixel_size={
	'roofplane': 60,
	'hip':60,
	'chimney':60,
	'wall':100,
	'window':10,
	'stone carving':30,
	'door':30,
	'glass house':60,
	'balcony':60
}



def write_boundary(boundaryEdges, outputPath, componentName):
	with open(os.path.join(outputPath, componentName + '.obj'),'w') as f:
		exp = 0.0000001
		for edge in boundaryEdges:
			[p1, p2] = edge	
			str1  = 'v '+ str(p1[0]) + ' ' + str(p1[1]) + ' ' + str(p1[2]) + '\n'
			str2  = 'v '+ str(p2[0]) + ' ' + str(p2[1]) + ' ' + str(p2[2]) + '\n'
			str3  = 'v '+ str(float(p1[0]) +exp) + ' ' + str(float(p1[1])+exp) + ' ' + str(float(p1[2])+exp) + '\n'
			content = ''.join([str1, str2,str3])
			f.writelines(content)
		faces = ['f '  + str(i*3+1) + ' ' +  str(i*3+2) + ' ' + str(i*3+3) + '\n' for i in range(len(boundaryEdges))]
		f.writelines(faces)


		

def point_to_str(point):
	try:
		point = [str(format(p,'.8f')) for p in point]
	except:
		pass

	p_str=' '.join(point)
	return p_str


def tria_to_str(triaList):
	'''一个三角形一行'''
	triaList_str = []
	for tria in triaList:
		line = [point_to_str(point) for point in tria]
		triaList_str.append(line)
	return np.array(triaList_str)



def generate_colors(N = 10):
	HSV_tuples = [(x*1.0/N, 0.7, 0.7) for x in range(N)]
	RGB_tuples = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))
	random.shuffle(RGB_tuples)
	return RGB_tuples


def generate_sub_regions(height, width,region_size,overlap):
	stride = region_size - int(region_size*overlap)
	ynum = int(np.round(height/float(stride)) + 1)
	xnum = int(np.round(width/float(stride)) + 1)

	if height >0  and height < region_size:
		y_num = 1
	if width>0 and width<region_size:
		xnum = 1
	sub_regions = {}
	for y in range(ynum):
		ystart = y*stride
		yend = min(ystart+region_size-1, height-1)
				
		for x in range(xnum):
			
			xstart = x*stride
			xend = min(xstart+region_size-1, width-1)
			sub_regions[str(y) + '_'+str(x)] = [xstart, ystart, xend, yend]

			sub_width = xend-xstart + 1
			if sub_width<region_size:
				### 到边缘了
				break

		sub_height = yend-ystart + 1
		if sub_height<region_size:
			### 到边缘了
			break
	return sub_regions


def image_concat(segments_path, view_img_path, model_name, region_size=512, overlap=0.2):
	### 存放大图影像和拼好的mask
	if not os.path.exists(view_img_path):
		os.makedirs(view_img_path)

	##### 收集视角信息
	mask_files = glob.glob(segments_path + '/' + model_name + '*.png')
	view_images = {}
	for file in mask_files:
		fname = os.path.splitext(os.path.split(file)[-1])[0]
		orientation ='_'.join(fname.split('_')[:-2])
		print(orientation)
		if orientation not in view_images:
			view_images[orientation] = []
		view_images[orientation].append(fname)

	#### 拼成大图
	for view, mask_list in view_images.items():
		savefig = os.path.join(view_img_path, view +'_vis.png') 
		if os.path.exists(savefig):
			continue


		ori_img = os.path.join(view_img_path, view +'.png')
		if not os.path.exists(ori_img):
			continue

		image = cv2.imread(ori_img) ## 读取原图
		H, W, C = image.shape

		### 原图为0.01的分辨率，按照1024切割
		### 训练和测试中，将所有图像降采样一半，分辨率变为0.02，切割大小为512，
		### 因此拼接的时候，先将原图降采样，然后按照降采样后的大小拼接
		image = cv2.resize(image,[int(W/2),int(H/2)])
		# cv2.imwrite(ori_img.replace('ori_view_image', 'ori_view_image_0.02'), image)
		H, W, C = image.shape

		bImage = Image.new('L',(W,H))
		sub_regions = generate_sub_regions(H, W, region_size,overlap)

		for sub_id, [xstart, ystart, xend, yend] in sub_regions.items():
			[y,x] = sub_id.split('_')
			y = int(y)
			x = int(x)
			maskfile = os.path.join(segments_path, view+ '_' + str(y)+ '_' + str(x) + '.png')
			if not os.path.exists(maskfile):
				continue
			mask = Image.open(maskfile)		
			mask_restore = mask.crop((xstart, ystart,xend, yend))

			x_left = 0 if x==0 else x*(1-overlap)*region_size 
			y_top = 0 if y==0 else y*(1-overlap)*region_size
			location = (int(x_left), int(y_top))
			# print(y,x,location)
			bImage.paste(mask, location)

		bImage.save(os.path.join(view_img_path, view +'_mask.png'))
		# # 上颜色叠加显示
		lbl = np.array(bImage).astype(np.int64)
		display_instances(image, lbl, classes, savefig)
	

def detect_islands(mesh_labels, triangles, faces):
	islands = [] ## 存储离群 三角形的 索引 
	labels = np.unique(mesh_labels)
	for label in labels:
		if label <=0:
			continue

		cls_name = str(classes[int(label)])
		trias_index = np.where(mesh_labels==label)[0]

		cur_trias = triangles[trias_index,:,:]
		cur_faces = faces[trias_index,:]
		clusters = regionGrowing(cur_trias, cur_faces, method ='connectivity', mode = 'edge')

		for j, cluster in enumerate(clusters):
			if len(cluster)<pixel_size[cls_name]:
				islands.extend(trias_index[cluster]) ### 后面还要根据连通性分组，这里直接用extend存在一起
				mesh_labels[trias_index[cluster]] = -1 ###重置为-1

	### 小trick，可以提升效果，放置island 类别错误。方法为： 将套圈的island合并为一个整体
	islands_tria_indexes = np.unique(islands)
	integrated_islands = []
	if len(islands_tria_indexes):
		is_trias = triangles[islands_tria_indexes,:,:]
		is_faces = faces[islands_tria_indexes,:]
		clusters = regionGrowing(is_trias, is_faces, method = 'connectivity', mode = 'point')
		integrated_islands = [islands_tria_indexes[cluster] for cluster in clusters]
		integrated_islands = sorted(integrated_islands, key = lambda x:len(x), reverse=True)
	else:
		integrated_islands = []


	return integrated_islands

def find_trias_with_common_edge(boundaryEdges, btrias_index, faces, mesh_labels):
	##### 查找共边的有语义信息的三角形
	adjacency_trias = []
	for [sID, eID] in boundaryEdges:

		row_ind1 = np.where(faces == sID)[0].tolist()
		row_ind2 = np.where(faces == eID)[0].tolist()

		### common edge
		# find = list(set(row_ind1).intersection(set(row_ind2)))
		# find = list(set(find).difference(set(btrias_index)))	

		### common point
		find = row_ind1 + row_ind2
		find = list(set(find).difference(set(btrias_index)))

		if len(find):
			f = find[0]
			if mesh_labels[f] >0: 
				#### 如果是感兴趣目标，取出类别
				adjacency_trias.append(f)
	return adjacency_trias




def main(model_path, segments_path, view_img_path, save_folder):

	if not os.path.exists(model_path):
		raise(model_path + " is not exists, please check.")
	if not os.path.exists(save_folder):
		os.makedirs(save_folder)

	
	
	model_list = glob.glob(os.path.join(model_path,'*.obj')) # 3D model file name
	model_list = [os.path.split(model)[-1] for model in model_list]
	print(model_list)

	model_list = ['Tile_+010_+004_1.obj', 'Tile_+023_+026_4.obj','Tile_+026_+028_1.obj','Tile_+026_+028_8.obj']
	for model in model_list:

		print('\n ........................current model is: ' + model + '.............................')
		model_name, model_ext = os.path.splitext(model)	


		image_concat(segments_path, view_img_path, model_name) ### 合并mask 为大图
		

		# stage 1: read obj data
		model_file = os.path.join(model_path, model)
		obj_parser = OBJ_PARSER(model_file)
		triangles, faces, vertices = obj_parser.get_triangles()
		if faces.shape[1]==6:
			faces = faces[:,[0,2,4]]
		normals = obj_parser.get_normals(triangles)
		print('reading model done .....')

        #### 1.5 利用bird eye view 的 mask preditions 对跨区的大三角形进行局部三角剖分
		do_triangulation = True
		if do_triangulation:
			print('do_triangulation.....')
			view_masks = glob.glob(view_img_path + '/' + model_name + '*_mask.png')
			M = len(view_masks)
			for view_ind in range(M): #大图 view image 
				maskfile = view_masks[view_ind]
				if "X-Y_bird" not in maskfile:
					continue
				triangles = triangulation(triangles, maskfile, normals, pixel_resolution=0.02)
				normals = obj_parser.get_normals(triangles) ###更新normals
		
				
				# trias_index = np.arange(0,len(triangles)) ## all trias
				# write_into_obj(trias_index, model_path + '/'+ model_name + '_triangulation_'+ str(view_ind) + '.obj', triangles)
			trias_index = np.arange(0,len(triangles)) ## all trias
			write_into_obj(trias_index, model_path + '/'+ model_name + '.obj', triangles)
			print('triangulation done.....')
									
			
		# stage 2: 2D-3D projection
		view_masks = glob.glob(view_img_path + '/' + model_name + '*_mask.png')
		M = len(view_masks)
		N = len(triangles)
		multiview_labels = np.zeros((N,M)) #记录不同视角的分割结果
		for view_ind in range(M): #大图 view image 
			maskfile = view_masks[view_ind]
			print(maskfile)
			labels = projection(triangles, maskfile, normals, pixel_resolution=0.02)
			multiview_labels[:, view_ind] = labels.transpose()
		print('projection done .....')

		if multiview_labels.shape[0]==0:
			continue
		

		###stage 3: 对多视角投影结果 进行 众数投票
		mesh_labels = np.zeros(N).astype(np.int64)
		for v in range(N):
			### 取出可视结果进行众数投票！！！！！！！
			row = multiview_labels[v,:]
			vis_ind = np.where(row>0)[0]

			final_label = 0
			if len(vis_ind)==0:
				continue
			elif len(vis_ind) == 1:
				final_label = int(row[vis_ind[0]])
			else:
				vis_label = row[vis_ind].astype(np.int64)
				count = np.bincount(vis_label)
				final_label = int(np.argmax(count))
	
			mesh_labels[v] = final_label


		#islands 优化前: 按照类别输出
		obj_save_folder = os.path.join(save_folder, model_name)
		if not os.path.exists(obj_save_folder):
			os.makedirs(obj_save_folder)

		labels = np.unique(mesh_labels)
		for label in labels:
			if label <=0:
				continue
			trias_index = np.where(mesh_labels==label)[0]
			cls_name = str(classes[int(label)])
			write_into_obj(trias_index, obj_save_folder + '/'+ str(label) + '_' + cls_name + '_before.obj', triangles)
	
			

		#stage 4: 消除噪声：通过欧式聚类分组，找出数量少的 island,将其合并为周边邻近类别 radius = 0.5
		obj_parser = OBJ_PARSER(model_file)
		faces = obj_parser.get_faces()
		islands = detect_islands(mesh_labels, triangles, faces)
		##### 然后查找island周边三角形，并统计语义类别
		n = 1
		for trias_index in islands:
			# write_into_obj(trias_index, './','island_' + str(n), triangles)
			n+=1

			### isLand 边界线及边界三角形提取
			this_face = faces[trias_index,:]
			boundaryEdges, btrias_index_ = generate_boundary(this_face)
			btrias_index = trias_index[btrias_index_]

			##### 索引转换成具体的坐标值
			bLines = [[vertices[start-1], vertices[end-1]] for (start, end) in boundaryEdges]
			bTrias = [triangles[ind,:,:] for ind in btrias_index]

			#### 输出boundaryEdges
			# outputPath = './'
			# write_boundary(bLines, outputPath,'bound')

			adjacency_trias = find_trias_with_common_edge(boundaryEdges, btrias_index, faces, mesh_labels)
			if len(adjacency_trias):
				neighborLabel = mesh_labels[adjacency_trias]
				count = np.bincount(neighborLabel)
				final_label = int(np.argmax(count))
				mesh_labels[trias_index] = final_label

				

		#stage 5 : 按照类别输出语义分割结果
		obj_save_folder = os.path.join(save_folder, model_name)
		if not os.path.exists(obj_save_folder):
			os.makedirs(obj_save_folder)

		labels = np.unique(mesh_labels)
		for label in labels:
			if label <=0:
				continue
			trias_index = np.where(mesh_labels==label)[0]
			cls_name = str(classes[int(label)])
			write_into_obj(trias_index, obj_save_folder + '/'+ str(label) + '_' + cls_name + '.obj', triangles)
		
		print('Finish projection.....\n')
			


if __name__ == "__main__":
	model_path = '3D Models'  # 3D model folder
	segments_path = '2D SemSegments' # path for segmentation results in oblique image.
	view_img_path = 'Multiview images'# path for original big view images and big segmentation mask.
	
	# folder to save the results.
	save_folder = './sem_results'
	

	main(model_path, segments_path, view_img_path, save_folder) 
