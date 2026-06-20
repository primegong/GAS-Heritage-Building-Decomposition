# import _init_paths
import os, math, cv2, random, colorsys
import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Polygon
from lib.read_data.OBJ_PARSER import OBJ_PARSER
from PIL import Image
from PIL import ImageDraw
import os
import numpy as np
if not hasattr(np, "bool"):
    np.bool = np.bool_
from labelme import utils




planes = {
'X-Y_bird': [0,0,1], 
'X-Z_frontup':[0,-1,1], 
'X-Z_backup':[0,1,1], 
'Y-Z_rightup':[1,0,1], 
'Y-Z_leftup':[-1,0,1],
'X-Z_front':[0,-1,0], 
'X-Z_back':[0,1,0],
'Y-Z_right':[1,0,0], 
'Y-Z_left':[-1,0,0], 
'front_leftup':[-1,-1,1], 
'front_rightup':[1,-1,1], 
'back_leftup':[-1,1,1],
'back_rightup':[1,1,1]
}

rotate_angle = {'X-Y_bird':0,
				'X-Z_frontup':0,
				'X-Z_backup': 180,
				'Y-Z_rightup': 90,
				'Y-Z_leftup': -90,
				'X-Z_front':0,
				'X-Z_back': 180,
				'Y-Z_right': 90,
				'Y-Z_left': -90,
				'front_leftup':-45, 
				'front_rightup': 45,  
				'back_rightup': 135, 
				'back_leftup': -135}

def compute_tri_iou(tria1, tria2):
	"""
	Intersection over union between two shapely polygons.
	"""
	polygon_points1 = np.array(tria1).reshape(3, 2)
	poly1 = Polygon(polygon_points1).convex_hull
	polygon_points2 = np.array(tria2).reshape(3, 2)
	poly2 = Polygon(polygon_points2).convex_hull
	# union_poly = np.concatenate((polygon_points1, polygon_points2))
	if not poly1.intersects(poly2):  # this test is fast and can accelerate calculation
		iou = 0.0
	else:
		try:
			inter_area = poly1.intersection(poly2).area
			union_area = poly1.area + poly2.area - inter_area
			# union_area = MultiPoint(union_poly).convex_hull.area
			if union_area == 0 or poly1.area==0 or poly2.area==0:
				return 1.0
			# print(union_area, poly1.area, poly2.area, list2)

			##### 考虑三角形之间可能存在包含关系，因此取最大交集指标
			overlap1 = inter_area/poly1.area
			overlap2 = inter_area/poly2.area
			iou = float(inter_area) / union_area
			iou = max(iou, max(overlap1,overlap2))
		except shapely.geos.TopologicalError:
			print('shapely.geos.TopologicalError occured, iou set to 0')
			iou = 0.0
	return iou

def bbox_2d(project_tria2D):
	'''project_tria2D = [n,3,2]'''
	xy_min = np.min(project_tria2D, axis = 1)
	xy_max = np.max(project_tria2D, axis = 1)
	bbox = np.hstack((xy_min, xy_max))
	return bbox

def compute_bbox_iou(target_box, bbox_2d):
	x_left = np.maximum(target_box[0], bbox_2d[:,0])
	y_top = np.maximum(target_box[1], bbox_2d[:,1])
	x_right = np.minimum(target_box[2], bbox_2d[:,2])
	y_bottom = np.minimum(target_box[3], bbox_2d[:,3])

	delta_x = x_right - x_left
	delta_y = y_bottom - y_top

	ind1 = np.where(delta_x >0)[0]
	ind2 = np.where(delta_y >0)[0]
	overlap_ind = list(set(ind1).intersection(set(ind2)))
	sign = np.zeros(len(bbox_2d))
	sign[overlap_ind] = 1

	area1 = (target_box[2] - target_box[0])*(target_box[3] - target_box[1])
	area2 = (bbox_2d[:,2] - bbox_2d[:,0])*(bbox_2d[:,3] - bbox_2d[:,1])
	overlap = delta_y * delta_x
	iou = sign * overlap/(area1+area2-overlap)

	return iou

		
	
		

def get_normal(tria):
	tria = np.array(tria)
	v11 = tria[1, :] - tria[0, :]
	v22 = tria[2, :] - tria[0, :]
	normal = np.cross(v11, v22)
	normal_unit = normal / np.linalg.norm(normal)
	return normal_unit


def cos_angle(v1, v2):
	v1 = v1/np.linalg.norm(v1)
	v2 = v2/np.linalg.norm(v2)
	theta = np.dot(v1, v2)
	theta = min(theta, 1)
	theta = max(theta, -1)
	return math.degrees(math.acos(theta))


def proj_xyz(plane, origin, point):
	[A, B, C] = plane
	[x,y,z] = point
	D = -A * origin[0] -B * origin[1] -C* origin[2]
	t = (A*float(x) + B*float(y) + C*float(z) + D)/ (A*A + B*B + C*C)
	x_p = float(x)-A*t
	y_p = float(y)-B*t
	z_p = float(z)-C*t
	return np.array([x_p, y_p, z_p])


def project_to_plane(triaTexture, orientation):
	normals = [get_normal(tria) for tria in triaTexture[:,:,:3]]
	normals = np.array(normals)

	planeNorm = planes[orientation]
	print('plane: ' + orientation)

	angle = math.radians(rotate_angle[orientation]) #### 弧度啊！！！
	rotationMatrix_zaxis = np.array([[math.cos(angle), math.sin(angle), 0], [-math.sin(angle), math.cos(angle), 0], [0, 0, 1]])
	rotationMatrix_xaxis = np.array([[1, 0, 0], [0, math.cos(-np.pi/4), math.sin(-np.pi/4)], [0, -math.sin(-45), math.cos(-np.pi/4)]])
	
	trias = triaTexture[:,:,:3]
	project_tria2D = []
	distances = []
	visable_inds = []
	

	origin = trias.mean(1).mean(0).reshape(-1,1)
	for i, tria in enumerate(trias):
		normal = normals[i]
		if cos_angle(normal, planeNorm)>90: 
			###不可视
			continue
		visable_inds.append(i)
		tria_2D = [] ###记录平面中三角形的的二维坐标
		h_mean = 0

		for [x,y,z] in tria:
			### 三维垂足坐标
			projected_xyz = proj_xyz(planeNorm, origin, [x,y,z])
			### 计算二维图像坐标系下的点坐标
			xyz_rotate = np.dot(rotationMatrix_zaxis, (projected_xyz.reshape(-1,1)-origin)) + origin
			if 'up' in orientation:
				xyz_rotate = np.dot(rotationMatrix_xaxis, xyz_rotate-origin) + origin
			
			if orientation == 'X-Y_bird':
				tria_2D.append([xyz_rotate[0,0], xyz_rotate[1,0]])
			else:
				tria_2D.append([xyz_rotate[0,0], xyz_rotate[2,0]])

			[x_p, y_p, z_p] = projected_xyz
			d_height = float(math.sqrt((float(x)-x_p)*(float(x)-x_p) + (float(y)-y_p)*(float(y)-y_p) + (float(z)-z_p)*(float(z)-z_p))) 
			h_mean += d_height

		project_tria2D.append(tria_2D)
		distances.append(h_mean/3.0)
	

	#### 找出重叠的三角形，并保留前景的那个
	#### 对于bird 视角和带有‘up'的视角,前景是heigh大的那个；对于前后左右四个方向，前景是与视角法向量夹角小于90度的那个

	sign = np.ones(len(project_tria2D)) ##### 1表示前景； -1 表示背景，即被抑制的三角形
	bbox_2D = bbox_2d(project_tria2D)
	vis_triaTexture = triaTexture[visable_inds,:,:]
	vis_gravity = vis_triaTexture[:,:,:3].mean(1)
	thresh = 0.02
	for i, box in enumerate(bbox_2D):
		if sign[i] == -1:
			#### 不考虑背景三角形
			continue
		
		#### 计算bbox iou
		bbox_iou = compute_bbox_iou(box, bbox_2D)
		find = np.where(bbox_iou>thresh)[0]
		#### 去除自身
		find = list(set(find).difference(set([i])))
		if len(find):
			for f in find:
				#### 不考虑背景三角形
				if sign[f] == -1:
					continue
				#### 进一步计算tria_iou
				tria_iou = compute_tri_iou(project_tria2D[i], project_tria2D[f])
				if tria_iou>thresh:

					if orientation == 'X-Y_bird' or 'up' in orientation:
						#### 根据z值判断前后景
						height_diff = vis_gravity[i,2] - vis_gravity[f,2]
						# print(height_diff)
						if height_diff>0.3:
							sign[f] = -1
						elif height_diff<-0.3:
							sign[i] = -1
					else:

						### 根据朝向判断前后景：与法向量的夹角小于90，则i为前景，否则f为前景
						vector  = vis_gravity[i,:] - vis_gravity[f,:]
						dis = np.sqrt(np.sum(vector**2))
						if dis <thresh:
							### 距离大于一定是才认定是前后景关系
							continue
						degree = cos_angle(planeNorm,vector)
						if degree<90:
							sign[f] = -1
						else:
							sign[i] = -1

	
	return np.float32(project_tria2D), distances, visable_inds, sign


def generate_proj_image(this_triaTexture, tria_pxy, color_image, this_figures, model_file, buildName, obj_parser):
	# [blank_h, blank_w] = blank_image.shape
	# color_image = cv2.merge([blank_image]*3)
	[blank_h, blank_w,c] = color_image.shape
	unique_figures = list(set(this_figures))

	for figName in unique_figures:
		texure_file = os.path.split(model_file)[0] + '/' + figName + '.png'
		textureImg = cv2.imread(texure_file)
		th, tw, tc = textureImg.shape

		ind = np.where(this_figures == figName)[0]
		if len(ind) == 0:
			continue
		tria_uv_list = this_triaTexture[ind,:,:]
		pixel_xy = tria_pxy[ind,:,:]

		texure_xy = []
		for tria in tria_uv_list[:,:,3:5]:
			tria_txy = []

			for [u,v] in tria:
				txy = obj_parser.vts2textureXY([u,v], tw, th)
				tria_txy.append(txy)
			texure_xy.append(tria_txy)

		for i in range(len(tria_uv_list)):
			tria_pXY = np.float32(pixel_xy[i])
			tria_tXY = np.float32(texure_xy[i])
			M = cv2.getAffineTransform(tria_tXY, tria_pXY)
			dst = cv2.warpAffine(textureImg, M, (blank_w, blank_h))

			mask = Image.new('RGB', (blank_w, blank_h), (0,0,0))
			draw = ImageDraw.Draw(mask)
			polygon = [tuple(point) for point in tria_pXY]
			draw.polygon(polygon, fill = (255,255, 255), outline = (255,255, 255))
			mask = np.array(mask)
			for i in range(3):
				color_image[:,:,i] = np.where(mask[:,:,i] == 255, dst[:,:,i], color_image[:,:,i])

	return color_image


def plane2D_xy(triaTexture, textureFigure_index, orientation):
	if len(triaTexture)<30:
		return

	### 找出可视三角形，并计算投影坐标，
	project_tria2D, distances, visable_inds, sign = project_to_plane(triaTexture, orientation)

	print(len(project_tria2D), len(visable_inds),'................')
	visable_triaTexture = triaTexture[visable_inds,:,:] 
	visable_figures = textureFigure_index[visable_inds]
	##生成blank images
	blank_image, tria_pxy = cal_blank_image(project_tria2D, pixel_resolution = 0.01)
	# write_Trias_uv(visable_inds, 'test', triaTexture, textureFigure_index, buildName)
	return visable_triaTexture, tria_pxy, blank_image, visable_figures, visable_inds, sign


def cal_blank_image(proj_pxy, pixel_resolution = 0.02):
	### 栅格化后的三角形图像坐标
	x_min = min(proj_pxy[:,:,0].flatten())
	y_min = min(proj_pxy[:,:,1].flatten())
	y_max = max(proj_pxy[:,:,1].flatten())
	tria_pxy_list = []
	for tria in proj_pxy:
		px = np.round((tria[:,0]-x_min)/pixel_resolution)
		py = np.round((tria[:,1]-y_min)/pixel_resolution)
		tria_pxy = np.vstack((px, py)).transpose()
		tria_pxy_list.append(tria_pxy)
	tria_pxy_list = np.array(tria_pxy_list)
	img_width = int(max(tria_pxy_list[:,:,0].flatten()) - min(tria_pxy_list[:,:,0].flatten()) + 1)
	img_height = int(max(tria_pxy_list[:,:,1].flatten()) - min(tria_pxy_list[:,:,1].flatten()) + 1)

	###以左上角为（0，0）
	tria_pxy_list[:,:,1] = img_height-1 - tria_pxy_list[:,:,1]

	blank_image = np.zeros((img_height, img_width), dtype=np.uint8)
	return blank_image, np.int64(tria_pxy_list)


def main(model_path, output):
	tiles = os.listdir(model_path)
	for tileModel in tiles:
		print('\n ........................current model is: ' + tileModel + '.............................')
		files = os.listdir(os.path.join(model_path, tileModel))
		objfiles = [file for file in files if os.path.splitext(file)[-1] =='.obj']
		
		for file in objfiles:
			[buildName, ext] = os.path.splitext(file)

			### output path
			saveImages = os.path.join(output, tileModel, buildName)
			if not os.path.exists(saveImages):
				os.makedirs(saveImages)

			###read obj
			model_file = os.path.join(model_path, tileModel, file)
			obj_parser = OBJ_PARSER(model_file)
			triaTexture, fuseFace, textureFigure_index, trias_str_list = obj_parser.get_triangles_with_texture()
			print('buildName:', buildName, '     num triangles:', len(triaTexture))

			for orientation in planes:
				save_file = saveImages + '/' + buildName  + '_' + orientation+'.png'
				if os.path.exists(save_file):
					continue

				this_triaTexture, tria_pxy, blank_image, this_figures, visable_inds, sign = plane2D_xy(triaTexture, textureFigure_index, orientation)

				#### 仿射变换帖纹理
				print('Texture Mapping...')
				fg = np.where(sign>0)[0]
				bg = np.where(sign<0)[0]
				color_image = cv2.merge([blank_image]*3)
				for ind in [bg,fg]:
					color_image = generate_proj_image(this_triaTexture[ind,:,:], tria_pxy[ind,:,:], color_image, this_figures[ind], model_file, buildName, obj_parser)
				
				cv2.imwrite(save_file, color_image)

		


		

if __name__ == "__main__":

	# 模型文件
	model_path = '3D Models'
	output = 'Projective_IMAGES'
	main(model_path, output)

	


