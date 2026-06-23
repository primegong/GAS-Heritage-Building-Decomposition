import copy, math
import numpy as np
from sklearn.neighbors import KDTree
from shapely.geometry import Polygon
from functools import reduce
import numpy as np

#### tria feature
def line_distance(line1, line2):
	point1=np.array(line1).mean(0)
	point2=np.array(line2).mean(0)
	return np.sqrt(np.sum(point1-point2)**2)


def get_normal(tria):
	tria = np.array(tria)
	v11 = tria[1, :] - tria[0, :]
	v22 = tria[2, :] - tria[0, :]
	normal = np.cross(v11, v22)
	normal_unit = normal / np.linalg.norm(normal)
	return normal_unit


def triasCommonPoints(tria1, tria2):
	tria1 = tria1.tolist()
	tria2 = tria2.tolist()

	edgePoint = []
	p3 = []
	p4 = copy.deepcopy(tria2)
	
	for i, p in enumerate(tria1):
		try:
			ind = tria2.index(p)
			edgePoint.append(tria2[ind])
			p4.remove(tria2[ind])
		except:
			p3 = p
	if len(edgePoint) != 3:
		p4 = p4[0]

	return edgePoint, p3, p4

 
def point2plane_distance(point, tria):
	"""
	return:点到面的距离
	"""
	Ax, By, Cz, D = define_plane(tria)
	mod_d = Ax * point[0] + By * point[1] + Cz * point[2] + D
	mod_area = np.sqrt(np.sum(np.square([Ax, By, Cz])))
	d = abs(mod_d) / mod_area
	return d


def get_dihedralAngel(tria1, tria2):
	'''平面二面角theta 范围在[0,90]'''

	normal1 = get_normal(tria1)
	normal2 = get_normal(tria2)
	theta = np.dot(normal1, normal2)

	alpha = []
	belta = []
	gamma = []

	return theta, alpha, belta, gamma

def define_plane(tria):
	"""
	法向量    ：n={A,B,C}
	空间上某点：p={x0,y0,z0}
	点法式方程：A(x-x0)+B(y-y0)+C(z-z0)=Ax+By+Cz-(Ax0+By0+Cz0)
	https://wenku.baidu.com/view/12b44129af45b307e87197e1.html
	:param point1:
	:param point2:
	:param point3:
	:param point4:
	:return:（Ax, By, Cz, D）代表：Ax + By + Cz + D = 0
	"""

	[point1, point2, point3] = tria
	point1 = np.asarray(point1)
	point2 = np.asarray(point2)
	point3 = np.asarray(point3)
	AB = np.asmatrix(point2 - point1)
	AC = np.asmatrix(point3 - point1)
	N = np.cross(AB, AC)  # 向量叉乘，求法向量
	# Ax+By+Cz
	Ax = N[0, 0]
	By = N[0, 1]
	Cz = N[0, 2]
	D = -(Ax * point1[0] + By * point1[1] + Cz * point1[2])
	return Ax, By, Cz, D
 

def cos_angle(v1, v2):
	v1 = v1/np.linalg.norm(v1)
	v2 = v2/np.linalg.norm(v2)
	theta = np.dot(v1, v2)
	theta = min(theta, 1)
	theta = max(theta, -1)
	return math.degrees(math.acos(theta))


def concave_convex(tria1, tria2):
	''' 平面凹凸性：
	凹凸关系通过 CC（Extended Convexity Criterion） 和 SC （Sanity criterion）判据来进行判断。
	其中 CC 利用相邻两片中心连线向量与法向量夹角来判断两片是凹是凸。显然，如果图中a1>a2则为凹，反之则为凸。
	'''

	convexity = -1
	gravity1 = tria1.mean(0)
	gravity2 = tria2.mean(0)
	edgeVector = gravity1 - gravity2
	edgeVector = edgeVector / np.linalg.norm(edgeVector)
	normal1 = get_normal(tria1)
	normal2 = get_normal(tria2)
	alpha1 = math.degrees(math.acos(np.dot(normal1, edgeVector)))
	alpha2 = math.degrees(math.acos(np.dot(normal2, edgeVector)))

	'''考虑到测量噪声等因素，需要在实际使用过程中引入门限值（a1需要比a2大出一定量）来滤出较小的凹凸误判。'''

	delta = alpha1-alpha2 
	threshold = 15

	if delta<0 and abs(delta)>threshold:
		'''凸'''
		convexity = 1
	if abs(delta)<threshold:
		convexity = 0
	if delta>0 and abs(delta)>threshold:
		convexity = -1

	return convexity

def concave_convex_neigh(fi, fj, fuseFace):
	###扩展到邻域平面，计算凹凸性
	convexity = 0
	third_point_i = list(set(fuseFace[fi]).difference(set([start, end])))
	third_point_j = list(set(fuseFace[fj]).difference(set([start, end])))
	edgeVector = vertices[third_point_i,:][0] - vertices[third_point_j,:][0]
	edgeVector = edgeVector/np.linalg.norm(edgeVector)

	neigh_i = np.where(fuseFace==third_point_i)[0]
	neigh_j = np.where(fuseFace==third_point_j)[0]

	### mean normal
	normal_i = normals[fi,:] if len(neigh_i) == 0 else normals[neigh_i,:].mean(0)
	normal_i = normal_i/np.linalg.norm(normal_i)
	normal_j = normals[fj,:] if len(neigh_j) == 0 else normals[neigh_j,:].mean(0)
	normal_j = normal_j/np.linalg.norm(normal_j)

	alpha_i = math.degrees(math.acos(np.dot(normal_i, edgeVector)))
	alpha_j = math.degrees(math.acos(np.dot(normal_j, edgeVector)))
	delta = alpha_i - alpha_j 
	threshold = 15
	if delta<0 and abs(delta)>threshold:
		'''凸'''
		convexity = 1
	if abs(delta)<threshold:
		convexity = 0
	if delta>0 and abs(delta)>threshold:
		convexity = -1

	return convexity



#### pcd feature 
def get_Radius(pcd, point_num):
	tree = KDTree(pcd)
	neighbor_R = []
	for target in pcd:
		(dists, indexes) = tree.query([target], point_num)
		mean_dis = np.array(dists).mean()
		neighbor_R.append(mean_dis)
	sorted_R = np.array(sorted(neighbor_R))
	median_localization = int(len(neighbor_R)/2)
	R = sorted_R[median_localization]
	return R, tree, neighbor_R


def get_density(pcd, radius):
	tree = KDTree(data = pcd)
	density = []
	for target in pcd:
		(inds, dists) = tree.query_radius([target], radius)
		density.append(len(inds))
	return density

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

def generate_boundary(faces):
	# faces = np.array([[1,2,3],[3,2,4],[5,2,1]])
	faces = np.array(faces)
	faces = faces### python 索引 从 0 开始
	numFace = faces.shape[0]
	edgeStart = faces.flatten() 
	edgeEnd = faces[:,[1,2,0]].flatten() 
	edges = np.vstack((edgeStart, edgeEnd)) #[2,n]
	faceId = np.tile(np.arange(numFace),(3,1)).transpose().flatten().tolist()

	boundaryEdges = []
	boundaryFaces = []
	numEdges= edges.shape[1]
	for i in range(numEdges):
		curEdge = edges[:,i].tolist()
		# print(curEdge)
		ind1 = np.where(edges[1,:] == curEdge[0])[0]
		ind2 = np.where(edges[0,:] == curEdge[1])[0]
		index = list(set(ind1).intersection(set(ind2)))
		if len(index) == 0:
			boundaryEdges.append(curEdge)
			boundaryFaces.append([i])
	
	boundaryEdges=np.array(boundaryEdges)
	boundaryFaces = np.array(boundaryFaces)
	return boundaryEdges, boundaryFaces


def getCommonEdges(faces):
    '''
    input:
        faces:[nx3]
    return:
        commonEdges: indexes of vertices in common edges, [nx2]
        facePairs:  triangle pairs where they locate in ,[nx2]
    '''

    # faces = np.array([[1,2,3],[3,2,4],[5,2,1]])
    faces = np.array(faces)### python 索引 从 0 开始
    numFace = faces.shape[0]
    edgeStart = faces.flatten() 
    edgeEnd = faces[:,[1,2,0]].flatten() 
    edges = np.vstack((edgeStart, edgeEnd)) #[2,n]
    faceId = np.tile(np.arange(numFace),(3,1)).transpose().flatten().tolist()

    commonEdges = []
    facePires = []
    numEdges= edges.shape[1]
    for i in range(numEdges):
        curEdge = edges[:,i].tolist()
        # print(curEdge)
        ind1 = np.where(edges[1,:] == curEdge[0])[0]
        ind2 = np.where(edges[0,:] == curEdge[1])[0]
        index = list(set(ind1).intersection(set(ind2)))
        if len(index):
            commonEdges.append(curEdge)
            facePires.append([faceId[i], faceId[index[0]]])
    
    commonEdges=np.array(commonEdges)
    facePires = np.array(facePires)

    ###### 去除冗余:edgeStart<edgeEnd ################
    directioned = np.where(commonEdges[:,0]<commonEdges[:,1])[0]
    commonEdges = commonEdges[directioned,:]
    facePires = facePires[directioned,:]


    return commonEdges, facePires


def get_bbox3D(trias):
	xs = trias[:,:,0]
	ys = trias[:,:,1]
	zs = trias[:,:,2]
	x_min = min(xs.flatten())
	y_min = min(ys.flatten())
	z_min = min(zs.flatten())
	x_max = max(xs.flatten())
	y_max = max(ys.flatten())
	z_max = max(zs.flatten())
	bbox3D = [x_min, y_min, z_min, x_max, y_max, z_max]
	return bbox3D


def get_bbox_overlap(range_i, range_j):
	overlap_x_min = max(range_i[0], range_j[0])
	overlap_y_min = max(range_i[1], range_j[1])
	overlap_z_min = max(range_i[2], range_j[2])
	overlap_x_max = min(range_i[3], range_j[3])
	overlap_y_max = min(range_i[4], range_j[4])
	overlap_z_max = min(range_i[5], range_j[5])
	
	delta_x = overlap_x_max - overlap_x_min 
	delta_y = overlap_y_max - overlap_y_min
	delta_z = overlap_z_max - overlap_z_min


	bbox_overlap = delta_x * delta_y *delta_z

	bbox_i = np.array(range_i).reshape((2,3))
	area_i = reduce(lambda x,y : x*y, bbox_i[1,:]-bbox_i[0,:])
	bbox_j = np.array(range_j).reshape((2,3))
	area_j = reduce(lambda x,y : x*y, bbox_j[1,:]-bbox_j[0,:])
	overlap_ratio = bbox_overlap/min(area_i, area_j)


	return (delta_x, delta_y, delta_z), overlap_ratio


 
def getAreaOfPolygonbyVector(points):
	# 基于向量叉乘计算多边形面积
	area = 0
	if(len(points)<3):
		raise Exception("error")

	for i in range(0,len(points)-1):
		p1 = points[i]
		p2 = points[i + 1]

		triArea = (p1[0]*p2[1] - p2[0]*p1[1])/2
		area += triArea

	fn=(points[-1][0]*points[0][1]-points[0][0]*points[-1][1])/2
	return abs(area+fn)



def L2(p1, p2):
    return np.sqrt(np.sum((p1-p2)**2))