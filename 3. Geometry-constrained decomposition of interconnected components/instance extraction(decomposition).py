import os, glob,math, cv2
import numpy as np
from cluster.region_growing import regionGrowing
from cluster.meanShift import meanShift
from read_data.OBJ_PARSER import OBJ_PARSER
from sklearn.neighbors import KDTree
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from basic_feature import *

pixel_size={
	'roofplane': 100,
	'hip':60,
	'chimney':60,
	'wall':100,
	'window':10,
	'stone carving':30,
	'door':30,
	'glass house':60,
	'balcony':60
}

def assign_islands_to_neighbor(islands, triangles, output_ins):
	
	#### 就近分配聚类遗留的islands
	flag = -1 * np.ones(len(islands))
	for i, island in enumerate(islands):
		is_gravity = triangles[island,:,:].mean(0)
		
		max_dis = 10000.0
		result = -1
		for j, output in enumerate(output_ins):
			ins_gravity = triangles[output,:,:].mean(1)
			tree = KDTree(ins_gravity)
			dist, ind = tree.query([is_gravity],k=1)
			dist = dist[0][0]
			if dist<max_dis:
				max_dis = dist
				result = j
		flag[i] = result

	for f, island in zip(flag, islands):
		if f!= -1:
			output_ins[int(f)].append(island)
	return output_ins, flag




def write_into_obj(trais_index, obj_save_dir, triangles):

	all_vertice = []
	for ind in trais_index:
		triangles_3D = triangles[ind,:,:]
		# print(triangles_3D)
		for vertice in triangles_3D:
			vertice_str = 'v '+str(vertice[0]) + ' ' + str(vertice[1]) + ' ' + str(vertice[2]) + '\n'
			all_vertice.append(vertice_str)


	all_vertice_set = list(set(all_vertice))
	all_vertice_set.sort(key = all_vertice.index)
	with open(obj_save_dir,'w') as f:

		f.writelines(all_vertice_set)

		vertice_indexes = [all_vertice_set.index(vertice_str) + 1 for vertice_str in all_vertice] #### 顶点编号从1开始
		vertice_indexes = np.array(vertice_indexes).reshape(-1,3)
		vertice_indexes = ['f ' + str(vi[0]) + ' ' + str(vi[1]) + ' ' + str(vi[2]) + '\n' for vi in vertice_indexes]
		f.writelines(vertice_indexes)
	
	f.close()




def instances_connectivity(model_path, save_folder):
	'''
	对所有类先根据连通性分组

	'''
	islands = [] #### 存储连通性分组后产生的“孤岛”

	files = os.listdir(model_path)

	for file in files:
		[name, ext] = os.path.splitext(file)
		[label, cls_name] = name.split('_')
		model_file = os.path.join(model_path, file)

		obj_parser = OBJ_PARSER(model_file)
		triangles, faces, vertices = obj_parser.get_triangles()
		faces = faces[:,[0,2,4]]
	
		obj_save_folder = os.path.join(save_folder, name)
		if not os.path.exists(obj_save_folder):
			os.makedirs(obj_save_folder)

		#### 前景统一按照连通性和密度进行分组
		print('processing: ' + file)
		clusters = regionGrowing(triangles, faces)


		output_ins=[]
		islands = []
		for j, cluster in enumerate(clusters):
			cluster = cluster.tolist()

			### 三角形数量判断
			if len(cluster)>pixel_size[cls_name]:
				output_ins.append(cluster)
			else:
				islands.extend(cluster)

		output_ins, flag = assign_islands_to_neighbor(islands, triangles, output_ins)
		
		### 输出实例
		for j, ins_index in enumerate(output_ins):
			write_into_obj(ins_index, obj_save_folder  + '/' + cls_name + '_' + str(j) + '.obj', triangles)




def neigh_normals(normals, triangles):
	new_normals = []
	gravity = triangles.mean(1)
	tree = KDTree(gravity)
	for g in gravity:
		ind = tree.query_radius([g],r=0.2)
		# print(ind,'/////')
		mean_normal = normals[ind[0],:].mean(0)
		new_normals.append(mean_normal)
	return np.array(new_normals)



def isjoint(tPoints, sPoints, densityTresh = 1.0):
	'''距离和阶跃检测'''

	'''水平距离小于1, 同时高差小于0.1, 则邻接'''


	flag = False
	sTree = KDTree(sPoints[:,:])

	#### 统计小于1 的点数
	HdisM = np.zeros(len(tPoints))
	diff_H = np.zeros(len(tPoints))
	count = 0
	for i, tp in enumerate(tPoints):
		(dis, ind) = sTree.query([tp[:]])

		HdisM[i] = np.sqrt(np.sum((tp[:2]-sPoints[ind[0][0],:2])**2))
		height = abs(tp[-1] - sPoints[ind[0][0],-1])
		diff_H[i] = height

	ind = np.where(HdisM<densityTresh)[0]
	count = len(ind)
	# print(ind)


	if count>0:
		#### 邻近且上下高差小于0.1，不存在阶跃，则相邻
		delta_H  = min(diff_H[ind])
		if delta_H<0.1:
			flag = True
	return flag


def combine_adjacency(clusters, triangles):
	final_clusters = []
	flag = np.zeros(len(clusters))
	
	for i, index_i in enumerate(clusters):
		if flag[i] ==1:
			continue
		
		Triangles_i = triangles[index_i,:,:]
		tPoints = Triangles_i.reshape(-1,3)

		flag[i] = 1
		seedList = [tPoints]
		result = index_i
		while len(seedList):
			tPoints = seedList.pop(0)

			for j, index_j in enumerate(clusters):
				if i==j or flag[j] != 0:
					continue
				
				Triangles_j = triangles[index_j,:,:]
				sPoints = Triangles_j.reshape(-1,3)

				if isjoint(tPoints, sPoints):
					flag[j] = 1
					seedList.append(sPoints)
					result.extend(index_j)

		final_clusters.append(result)
	return final_clusters


def is_convex(normal_i, normal_j, edgeVector):
	alpha_i = math.degrees(math.acos(np.dot(normal_i, edgeVector)))
	alpha_j = math.degrees(math.acos(np.dot(normal_j, edgeVector)))
	delta = abs(alpha_i - alpha_j)  ##### 绝对值！！！！！
	threshold = 0.1

	convex = False	
	if delta>0 and delta>threshold:
		convex=True
	return convex
		


def combine_convex(clusters, triangles, normals):
	final_clusters = []
	flag = np.zeros(len(clusters))
	
	for i, index_i in enumerate(clusters):
		if flag[i] ==1:
			continue
		
		Triangles_i = triangles[index_i,:,:]
		tPoints = Triangles_i.reshape(-1,3)
		normal_i =  normals[i,:]

		flag[i] = 1
		seedList = [tPoints]
		result = index_i
		while len(seedList):
			tPoints = seedList.pop(0)

			for j, index_j in enumerate(clusters):
				if i==j or flag[j] != 0:
					continue
				
				Triangles_j = triangles[index_j,:,:]
				sPoints = Triangles_j.reshape(-1,3)
				normal_j =  normals[j,:]
				
				if isjoint(tPoints, sPoints):
					edgeVector = tPoints.mean(0) - sPoints.mean(0)
					edgeVector = edgeVector/np.linalg.norm(edgeVector)
					tag = is_convex(normal_i, normal_j,edgeVector)
					if tag:
						flag[j] = 1
						seedList.append(sPoints)
						result.extend(index_j)

		final_clusters.append(result)
	return final_clusters


def cos_angle(v1, v2):
	dot_product = np.dot(v1,v2)
	norm_v1 = norm(v1)
	norm_v2 = norm(v2)
	cos_theta = dot_product/(norm_v1 * norm_v2)

	# 计算夹角（弧度）
	angle_radians = np.arccos(np.clip(cos_theta, -1, 1))
	
	# 转换为角度
	angle_degrees = np.degrees(angle_radians)
	print(f"两个向量的夹角为: {angle_degrees} 度")
	return angle_degrees

def roof_decomposition(save_folder):

	### 针对单个屋顶进行 坡面分解：先基于法向量聚类，得到多个破碎的坡面，然后结合屋顶的向下凹陷结构，合并凹性连接的破碎破面

	model_path = save_folder +  '/1_roofplane'
	abs_files = glob.glob(model_path + '/*.obj')


	obj_save_folder = os.path.join(model_path,'instance')
	if not os.path.exists(obj_save_folder):
		os.makedirs(obj_save_folder)

	k=0
	for abs_file in abs_files:
		print(abs_file)

		[abs_path, file] = os.path.split(abs_file)
		[name, ext] = os.path.splitext(file)
		cls_name = name.split('_')[0]

		model_file = os.path.join(model_path, file)

		obj_parser = OBJ_PARSER(model_file)
		triangles, faces, vertices = obj_parser.get_triangles()
		normals = obj_parser.get_normals(triangles)
		faces = faces[:,[0,2,4]]


		output_ins=[]
		if len(triangles)<200:
			output_ins = [np.arange(len(triangles))]
		else:
			##### 法向量聚类
			normals  = obj_parser.get_normals(triangles)
			ng_normals = neigh_normals(normals, triangles)
			clusters = meanShift(ng_normals[:,:2],quantile_value = 0.2) #### 二维法向量

			process_island = True
			if not process_island:
				output_ins = clusters
			else: 
				### 孤岛处理：先依据连通性分组，然后统计三角形数量，小于阈值的为孤岛，最后将孤岛分到最近的坡面组中。 
				### 如果不处理孤岛，直接就是： output_ins = clusters       
				output_ins, islands = [],[]
				# print(len(clusters))
				for j, cluster in enumerate(clusters):
					
					### 对每一个cluster, 依据进行分组
					disjoint_clusters = regionGrowing(triangles[cluster,:,:], faces[cluster,:])

					planeGroup = []
			 
					for d, disjoint in enumerate(disjoint_clusters):
						final_cluster = cluster[disjoint].tolist()
						#### 三角形数量判断 以及 法向量判断
						norm = ng_normals[final_cluster,:].mean(0)
						angle = cos_angle(np.array([0,0,1]), norm)
						if len(final_cluster) < pixel_size[cls_name] or angle<10:
							islands.extend(final_cluster)
						else:
							planeGroup.append(final_cluster)
						  
					if len(planeGroup) == 0:
						continue
					
					#### 考虑到有的坡面距离很近，所以对邻近坡面进行合并（最近距离小于阈值，且不存在阶跃高差，则认为是邻接，分为一组）
					planeGroup = combine_adjacency(planeGroup, triangles)

					
					output_ins.extend(planeGroup)
				#### 将孤岛分配到最近的组
				output_ins, flag = assign_islands_to_neighbor(islands, triangles, output_ins)
				
				##### 将凹陷且邻接的坡面进行合并
				output_ins = combine_convex(output_ins, triangles, normals)

		### 输出实例
		output_ins = sorted(output_ins, key = lambda x:len(x), reverse=True)
		for ins_index in output_ins:
			write_into_obj(ins_index, obj_save_folder  + '/' + 'roofplane_'+str(k)+ '.obj', triangles)
			k+=1



def compute_tri_iou(list1, list2):
	"""
	Intersection over union between two shapely polygons.
	"""
	polygon_points1 = np.array(list1).reshape(-1, 2)
	poly1 = Polygon(polygon_points1).convex_hull
	polygon_points2 = np.array(list2).reshape(-1, 2)
	poly2 = Polygon(polygon_points2).convex_hull
	union_poly = np.concatenate((polygon_points1, polygon_points2))
	
	if not poly1.intersects(poly2):  # this test is fast and can accelerate calculation
		iou = 0
		print('not intersection')
	else:
		try:
			inter_area = poly1.intersection(poly2).area
			union_area = poly1.area + poly2.area - inter_area
			# union_area = MultiPoint(union_poly).convex_hull.area
			if union_area == 0:
				return 1
			
			iou1 = float(inter_area) / float(union_area)

			if min(poly1.area, poly2.area) > 0:
				iou2 = float(inter_area) / float(min(poly1.area, poly2.area))
			else:
				iou2 = 1
			
			iou = max(iou1, iou2)
	
		except shapely.geos.TopologicalError:
			iou = 0
	
	return iou 


def combine_vertical_planes(clusters, triangles):
	final_clusters = []
	flag = np.zeros(len(clusters))
	for i, index_i in enumerate(clusters):
		if flag[i] ==1:
			continue
		
		Triangles_i = triangles[index_i,:,:]
		tPoints = Triangles_i.reshape(-1,3)
		thull = ConvexHull(tPoints[:,:2])
		thull_ind = thull.vertices.tolist()#要闭合必须再回到起点[0]
		thull = tPoints[thull_ind,:2]

		for j, index_j in enumerate(clusters):
			if j<=i or flag[j] == 1:
				continue
			
			Triangles_j = triangles[index_j,:,:]
			sPoints = Triangles_j.reshape(-1,3)
			shull = ConvexHull(sPoints[:,:2])
			shull_ind = shull.vertices.tolist()#要闭合必须再回到起点[0]
			shull = sPoints[shull_ind,:2]

			if compute_tri_iou(thull, shull)>=0.3:
				flag[i] = 1
				flag[j] = 1
				g = index_i + index_j
				final_clusters.append(g)

		if flag[i] == 0:
			final_clusters.append(index_i)
	
	return final_clusters




def facade_decomposition(save_folder):
	model_path = save_folder +  '/4_wall'
	abs_files = glob.glob(model_path + '/*.obj') ### 连通性分组结果


	obj_save_folder = os.path.join(model_path,'instance')
	if not os.path.exists(obj_save_folder):
		os.makedirs(obj_save_folder)


	k = 0
	for abs_file in abs_files:
		[abs_path, file] = os.path.split(abs_file)
		[name, ext] = os.path.splitext(file)
		cls_name = name.split('_')[0]

		model_file = os.path.join(model_path, file)
		obj_parser = OBJ_PARSER(model_file)
		triangles, faces, vertices = obj_parser.get_triangles()
		faces = faces[:,[0,2,4]]


		normals  = obj_parser.get_normals(triangles)
		clusters = meanShift(normals[:,:2], quantile_value = 0.1) ####使用二维法向量聚类可避免上下立面被分成上下两部分

		output_ins=[]
		islands = []
		for j, cluster in enumerate(clusters):

			disjoint_clusters = regionGrowing(triangles[cluster,:,:], faces[cluster,:]) 

			G = []
			for d, disjoint in enumerate(disjoint_clusters):
				
				final_cluster = cluster[disjoint].tolist()
				
				if len(final_cluster)>pixel_size[cls_name]:
					G.append(final_cluster)
				else:
					islands.extend(final_cluster)
			
			##### 将共面的上下两个立面合并：用二维凸包交集是否大于0.5判断
			# group_G = combine_vertical_planes(G, triangles)
			output_ins.extend(G)


		output_ins, flag = assign_islands_to_neighbor(islands, triangles, output_ins)

		
		### 输出实例
		for ins_index in output_ins:
			write_into_obj(ins_index, obj_save_folder  + '/wall_'+str(k)+ '.obj', triangles)
			k+=1

		



def find_optimal_horizontal_plane(triangles, faces):

    gravity = triangles.mean(1)

    zmin = gravity[:,2].min()
    zmax = gravity[:,2].max()

    heights = np.linspace(zmax, zmin, 50) #在（zmin, zmax）范围内生成一系列平面

    best_num = -1
    best_z = None

    for z_cut in heights:

        lower = np.where(gravity[:,2] < z_cut)[0] ## 如果平面下方三角形数量不足10个，说明是噪声

        if len(lower) < 10:
            continue

        clusters = regionGrowing(
            triangles[lower,:],
            faces[lower,:]
        ) # 平面下方应该是多个不连通的垂脊，所以用连通性进行区域生成，看看到底有几组

        num_group = len(clusters)  # 切割后，平面下方的组数

        if num_group > best_num:  #记录最大组数
            best_num = num_group
            best_z = z_cut  #返回组数最大的  并且 最高的那个平面
		elif num_group == best_num:
			best_z = max(best_z, z_cut)


    return best_z  


def split_up_down_ridges(lower_clusters, triangles, gravity, ridgeWidth=0.4):
    final_clusters = []

    for cl in lower_clusters:

        cl = np.array(cl)

        cur_gravity = gravity[cl,:]

        # P
        peak = cur_gravity[
            np.argmax(cur_gravity[:,2])
        ]

        # B
        d = np.linalg.norm(
            cur_gravity[:,:2]-peak[:2],
            axis=1
        )

        bottom = cur_gravity[
            np.argmax(d)
        ]

        # F

        PB = bottom[:2]-peak[:2]

        PB_norm = np.linalg.norm(PB)

        if PB_norm < 1e-6:
            final_clusters.append(cl.tolist())
            continue

        distances = []

        for p in cur_gravity:

            v = p[:2]-peak[:2]

            dist = abs(
                PB[0]*v[1]
                -
                PB[1]*v[0]
            ) / PB_norm

            distances.append(dist)

        distances = np.array(distances)

        farthest_idx = np.argmax(distances)

        F = cur_gravity[farthest_idx]

        # PF

        PF = F[:2]-peak[:2]

        PF_norm = np.linalg.norm(PF)

        if PF_norm < 1e-6:
            final_clusters.append(cl.tolist())
            continue

        upper_ridge = []
        lower_ridge = []

        for idx,p in zip(cl, cur_gravity):

            v = p[:2]-peak[:2]

            dist = abs(
                PF[0]*v[1]
                -
                PF[1]*v[0]
            ) / PF_norm

            if dist < ridgeWidth:
                upper_ridge.append(idx)

            else:
                lower_ridge.append(idx)

        if len(upper_ridge)>10:
            final_clusters.append(upper_ridge)

        if len(lower_ridge)>10:
            final_clusters.append(lower_ridge)

    return final_clusters



def detect_main_ridge(upper_triangles):
    points = upper_triangles.reshape(-1,3)
    xy = points[:,:2]
	if len(xy) < 4:
		return False

    hull = ConvexHull(xy)

    hull_pts = xy[hull.vertices]

    rect = cv2.minAreaRect(
        hull_pts.astype(np.float32)
    )

    w,h = rect[1]

    if min(w,h) < 1e-6:
        return False

    ratio = max(w,h)/min(w,h)

    return ratio > 3.0

def hip_decompostion(save_folder):
	model_path = save_folder +  '/2_hip'
	abs_files = glob.glob(model_path + '/*.obj')

	obj_save_folder = os.path.join(model_path,'instance')
	if not os.path.exists(obj_save_folder):
		os.makedirs(obj_save_folder)

	objID = 0
	mainRidgeID = 0
	
	for abs_file in abs_files:

		final_clusters = []

		[abs_path, file] = os.path.split(abs_file)
		[name, ext] = os.path.splitext(file)
		cls_name = name.split('_')[0]

		model_file = os.path.join(model_path, file)
		obj_parser = OBJ_PARSER(model_file)
		triangles, faces, vertices = obj_parser.get_triangles()
		faces = faces[:,[0,2,4]]
		assert triangles.shape[0] == faces.shape[0]
		

		# Step 1 上层主脊提取 : 寻找满足条件（切割后组数最多）的最高的平面
		PH = find_optimal_horizontal_plane(triangles, faces	)
		gravity = triangles.mean(1)
		upper = np.where(gravity[:,2] >= PH)[0]
		lower = np.where(gravity[:,2] < PH)[0]
		
		has_main_ridge = detect_main_ridge(triangles[upper,:,:])
		if has_main_ridge: # 情况1：存在正脊（庑殿、歇山顶）
			#如果满足长方形，则可认为upper 为mian ridge，输出保存
			write_into_obj(upper, obj_save_folder  + '/' + cls_name + '_mianRidge' + str(mainRidgeID) + '.obj',  triangles)   
			mainRidgeID += 1		
			
			#  Step 2 下层垂脊提取
			lower_clusters = regionGrowing(triangles[lower,:,:], faces[lower,:])
			mapped_clusters = []
			for cl in lower_clusters:
				mapped_clusters.append(
					lower[np.array(cl)].tolist()
				)


			# #### 3. 如果是w形状的屋脊（这种情况很少），上下颠倒，反过来区域生长
			# print('un down...')
			# new_clusters=[]
			# for cl in clusters:
			# 	cl=np.array(cl)
			# 	disjoint_clusters = regionGrowing(triangles[cl,:,:], faces[cl,:])
			# 	for d, disjoint in enumerate(disjoint_clusters):
			# 		inds = cl[disjoint]
			# 		_, flag_ = cycle(triangles[inds,:,:], faces[inds,:], 1, [], False, rate = 0.5)
			# 		group_ids = set(flag_)
			# 		for g in group_ids:
			# 			ids = np.where(flag_==g)[0]
			# 			new_clusters.append(np.array(inds)[ids])


			### 4. 判断是否存在拐点，如果存在则拆解为上下两条短的垂脊
			final_clusters = split_up_down_ridges(mapped_clusters, triangles, gravity)
			
		else:  # 情况2：不存在正脊（攒尖、悬山）
			lower_clusters = regionGrowing(triangles[lower,:,:], faces[lower,:])     #得到多个ridge

			## 计算每组中心，以便于后续对upper就近分配
			output_ins = []
			centers = []
			for cl in lower_clusters:
				inds = lower[cl]
				output_ins.append(
					inds.tolist()
				)

				centers.append(
					gravity[inds,:].mean(0)
				)

			#  对upper就近分配
			for idx in upper:
				g = gravity[idx,:]
				dists = [np.linalg.norm(g[:2]-c[:2]) for c in centers]
				nearest = np.argmin(dists)
				final_clusters[nearest].append(idx)


			
        # 后处理：噪声检测及控制
		final_output_ins=[]
		islands = []
		for cluster in final_clusters:
			if len(cluster)==0:
				continue
			disjoint_clusters = regionGrowing(triangles[cluster,:,:], faces[cluster,:])
			for d, disjoint in enumerate(disjoint_clusters):
				inds = cluster[disjoint].tolist()
				
				#### 三角形数量判断 
				if len(inds) > 10:
					final_output_ins.append(inds)
				else:
					islands.extend(inds)
		final_output_ins, flag_ = assign_islands_to_neighbor(islands, triangles, final_output_ins)



		#### 5. 输出分解结果
		sorted_output_ins = sorted(final_output_ins, key = lambda x:len(x), reverse = True)
		for cluster in sorted_output_ins:
			write_into_obj(cluster, obj_save_folder  + '/' + cls_name + '_' + str(objID) + '.obj', triangles)
			objID+=1





def main(model_path, save_folder):
	##### stage 1: 前景实例提取
	# folder to save foreground instances.
	if not os.path.exists(save_folder):
		os.makedirs(save_folder)

	####  除了屋顶区域的组件，立面上的组件，比如窗户、阳台那些大多是不相邻的，离散的，因此可以先用空间连通性将这些目标实例提取出来	
	###   而且一个建筑物有时候会有多个屋顶高低错落，所以连通性分析也有助于区分同一建筑的不同屋顶
	instances_connectivity(model_path, save_folder)


	# #### stage 2: 依据组件本身特性（坡面凹陷结构，屋脊对称性）进行实例级的组件提取
	roof_decomposition(save_folder)
	hip_decompostion(save_folder)

	facade_decomposition(save_folder)

	

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

	


if __name__ == "__main__":
	sem_path = 'sem_results'  # 3D model folder   
	save_folder = 'instance_result' 
	model_list = os.listdir(sem_path)
	model_list = ['Tile_+023_+026_3']
	for modelName in model_list:
		print('\n ........................current model is: ' + modelName + '.............................')

		main(sem_path + '/' + modelName, save_folder + '/' + modelName) 
