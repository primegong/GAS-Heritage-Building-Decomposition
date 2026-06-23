from scipy import spatial
import numpy as np 


'''main function: Cluster_MultiOutput
	input : datalist
	output: grouped datalist and their indexes in original datalist'''








def cluster(datalist, R, neighbor_R):
	sorted_inds = sorted(range(len(neighbor_R)), key=lambda k: neighbor_R[k])

	tree = spatial.KDTree(data = datalist)

	########  2 step: cluster ######################## 
	grouped_Trias = []
	target_used = []
	used = []
	index_OfTrias = []
	for j in sorted_inds:
		target = datalist[j]
		if tuple(target) in used:
			continue
		if j == len(datalist)-1:
			break

		group = [tuple(target)]
		index = [j]
		while True:
			
			inds = tree.query_ball_point(target, r = R)
			index = list(set(index).union(set(inds)))

			results = datalist[inds,:].tolist()
			results = list(map(tuple,results))
			results = list(set(results))
			
			group = list(set(results).union(set(group)))
			


			target_used.append((target[0], target[1], target[2]))
			if len(results)>1:
				#### find edge points ##############
				edge_points = convex_hull(results)
				inner_points = list(set(results).difference(set(edge_points)))
				target_used.extend(inner_points)
		
			
			#next:
			diff = list(set(group).difference(set(target_used)))
			if len(diff):
				target = diff[0]    
			else:
				break 
				             
		used = used + group
		group = list(map(list,group))
		grouped_Trias.append(group)
		index_OfTrias.append(index)
	return grouped_Trias, index_OfTrias