import numpy as np



def generate_boundary(faces):
	# faces = np.array([[1,2,3],[3,2,4],[5,2,1]])
	faces = np.array(faces)
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
			boundaryFaces.append(faceId[i])
	
	boundaryEdges=np.array(boundaryEdges)
	boundaryFaces = np.array(boundaryFaces) #### 边缘线所在三角形ID
	
	return boundaryEdges, boundaryFaces