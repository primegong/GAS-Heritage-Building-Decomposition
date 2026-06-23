import numpy as np
import math
from sklearn.neighbors import KDTree
from collections import Counter


##### 区域生长算法 ####
def regionGrowing(triangles, faces, method = 'connectivity', mode = 'edge'):
    seedMark = np.zeros(len(faces)) ###标记每个三角形的group id
    left = np.where(seedMark == 0)[0]
    
    gravity = triangles.mean(1)
    tree = KDTree(gravity)

    group_id = 1
    while len(left)>0: 
        regionGrow(seedMark, gravity, tree, faces, group_id, method,mode)
        left = np.where(seedMark == 0)[0]
        group_id+=1

    ###整理成grouplist
    group_ids = set(seedMark)
    clusters = []
    for k in group_ids:
        find = np.where(seedMark==k)[0]
        clusters.append(find)
    clusters = sorted(clusters, key = lambda x:len(x), reverse = True)
    return clusters



def regionGrow(seedMark, gravity, tree, faces, group_id, method, mode):
    #### 以未分类的第一个为种子点
    unclassified = np.where(seedMark == 0)[0]
    seedList = [unclassified[0]]
 
    
    while(len(seedList)>0):
        
        #弹出种子点序列的第一个点作为生长点,并标记为group_id
        seed = seedList.pop(0)
        seedMark[seed] = group_id
        
        ##查找连通
        find = []
        if method == 'connectivity':
            find = eval(method)(faces[seed,:], faces, mode)

        if method == 'dist_Euclidean':
            find = eval(method)(gravity[seed,:], tree)
       

        for f in find:
            if seedMark[f] == 0: ##避免重复
                ### 标记为 group id 组
                seedMark[f] = group_id

                ##判断是否满足种子点条件
                seedList.append(f)

                # if isSeed(gravity[f,:], gravity[unclassified,:]):
                #     seedList.append(f)

               

def isSeed(target, search_field):
    seed = False
    dists = L2(target, search_field)
    neigh_r = np.where(dists<=0.3)[0]
    if len(neigh_r) > 5:
        seed = True
    return seed

def L2(p1, p2):
    np1 = len(np.array(p1).shape)
    np2=  len(np.array(p2).shape)

    if np1 == np2 ==1:
        dist = np.sqrt(np.sum((p1-p2)**2))

    elif np1!= 1:
        num = np.array(p1).shape[0]
        p2 = np.tile(p2, (num, 1))
        dist = np.sqrt(np.sum((p1-p2)**2, axis = 1))

    elif np2!= 1:
        num = np.array(p2).shape[0]
        p1 = np.tile(p1, (num, 1))
        dist = np.sqrt(np.sum((p1-p2)**2, axis = 1))

    return dist


########################################
###### 一些生长条件/方法
########################################


def connectivity(seed, faces, mode = 'edge'):
    ### 三角形连通性 
    [v1, v2, v3] = seed
    find1 = np.where(faces==v1)[0].tolist() 
    find2 = np.where(faces==v2)[0].tolist() 
    find3 = np.where(faces==v3)[0].tolist() 

    union = find1 + find2 + find3
    find = union
    if mode =='edge':
        count = np.bincount(union)
        find = np.where(count==2)[0] ###共边，边连通
    if mode =='point':
        find = np.unique(union).tolist()  ### 共点通，点连
    return find



def dist_Euclidean(seed, tree, radius = 0.5):
    neighbors = tree.query_radius([seed], r = radius,)[0]
    return neighbors


def connectivity_radius(seed, features):
    find = []

    # ##计算垂直距离:
    a = np.arange(len(features))
    left = list(set(a).difference((set(find))))
    if len(left):
        source = [features[seed,0:2], features[seed,2:4]]
        for k in left:
            target = [features[k,0:2], features[k,2:4]]
            
            min_dis = 10000
            edge = []
            for si in source:
                for tj in target:
                    dis = L2(si, tj)
                    if dis<min_dis:
                        min_dis = dis
                        edge = [si, tj]

            v1 = source[1] - source[0]
            v2 = edge[1] -edge[0]
            cos_theta = np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
            sin_theta = np.sqrt(1-cos_theta**2)
            hirachy_dis = abs(min_dis*cos_theta)
            vertical_dis = abs(min_dis*sin_theta)
            if vertical_dis < 0.05 and hirachy_dis<1:
                find.append(k) 
    return find



def connectivity_for_lines(seed, features):
    [start, end] = features[seed]
    find1_s = np.where(features==start)[0] 
    find1_e = np.where(features==end)[0]
    find = list(set(find1_s).union(set(find1_e)))
    return find

# def orientation(seed, features):
#     ###计算方向向量
#     midPoints = features[:,:3]
#     orientationVector = features[:,3:]
#     tree = KDTree(midPoints)
#     neigh_ind = tree.query_radius([midPoints[seed]], 0.2)
#     neigh_ind = neigh_ind[0]
#     find = []
#     if len(neigh_ind):
#         source = orientationVector[seed,:]
#         for ind in neigh_ind:
#             target = orientationVector[ind,:]
#             cos_angle = np.dot(source, target)
#             cos_angle = min(cos_angle, 1)
#             cos_angle = max(cos_angle, -1)
#             degree = abs(math.degrees(math.acos(cos_angle)))
#             if degree<10:
#                 find.append(ind)
#     return find





def orientation(seed, features):
    source_start = features[seed,0:2]
    source_end = features[seed,2:4]
    source = features[seed,4:6]
    find = []
    for i in range(len(features)):
        target = features[i,4:6]
        cos_angle = np.dot(source, target)
        cos_angle = min(cos_angle, 1)
        cos_angle = max(cos_angle, -1)
        degree = abs(math.degrees(math.acos(cos_angle)))
        if degree<1:
            find.append(i)
            # ##计算距离：
            # target_start = features[i,0:2]
            # target_end = features[i,2:4]
            # dis1 = L2(source_start, target_start)
            # dis2 = L2(source_start, target_end)
            # dis3 = L2(source_end, target_start)
            # dis4 = L2(source_end, target_end)

            # distance = min([dis1, dis2, dis3, dis4])
            # if distance < 0.1:
            #     find.append(i)            
    return find
