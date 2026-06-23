import numpy as np


class OBJ_PARSER:
	def __init__(self, model_file):
		self.model_file = model_file
		self.vertices = []
		self.faces = []
		self.triangles = []

	def read_data(self, case = 0):
		try:
			f_generator = open(self.model_file,'r')
		except Exception as e:
			print(e)
		else:
			for index, line in enumerate(f_generator):

				if case == 0:
					pass
				if case == 1:   
					if line[:2] == 'v ':
						try:
							v,x,y,z = line.strip().split(' ')
						except:
							v,x,y,z,_,_,_ = line.strip().split(' ')
						self.vertices.append([x,y,z])
				if case == 2:
					if line[:2] == 'vt':
						vt,u,v = line.strip().split(' ')
						self.vts.append([u,v])
				if case == 3:
					if line[:2] == 'f ':
						f,_index1,_index2,_index3 = line.strip().split(' ')
						try:
							coord_index1, vt_index1 = _index1.split('/')
							coord_index2, vt_index2 = _index2.split('/')
							coord_index3, vt_index3 = _index3.split('/')
							self.faces.append([int(coord_index1), int(vt_index1), int(coord_index2), int(vt_index2), int(coord_index3), int(vt_index3)])
						except:
							coord_index1 = _index1
							coord_index2 = _index2
							coord_index3 = _index3
							self.faces.append([int(coord_index1), int(0), int(coord_index2), int(0), int(coord_index3), int(0)])


						
	def get_faces(self):
		# print('reading faces from obj file...')
		self.read_data(case = 3)
		return np.array(self.faces)

	def get_vertices(self):
		# print('reading vertices from obj files...')
		self.read_data(case = 1)
		# print('total vertices: ', len(self.vertices))
		return np.array(self.vertices).astype(np.float64)

	def get_vts(self):
		self.read_data(case = 2)
		return self.vts

	def get_triangles(self):
		faces = self.get_faces()
		vertices = self.get_vertices()
		for index in faces:
			triangle = [vertices[index[0]-1], vertices[index[2]-1], vertices[index[4]-1]]
			self.triangles.append(triangle)
		self.triangles = np.array(self.triangles).astype(np.float32)
		# print('total triangles: ',len(self.triangles))
		return self.triangles, faces, vertices


	def get_normals(self, trias):
		'''trias:  a list of triangles, 
		[N, 3, 3]

		return:
		normals: N x3
		'''

		v11 = trias[:,1, :] - trias[:,0, :]
		v22 = trias[:,2, :] - trias[:,0, :]
		normals = np.cross(v11, v22)
		normals_unit = normals / np.tile(np.linalg.norm(normals, axis = 1),(3,1)).transpose()
		self.normals = normals_unit
		return normals_unit
	
	def getCommonEdges(self, faces):
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


