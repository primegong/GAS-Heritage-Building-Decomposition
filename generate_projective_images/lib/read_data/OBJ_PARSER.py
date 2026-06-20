import numpy as np
import os
import cv2


class OBJ_PARSER:
	def __init__(self, model_file):
		self.model_file = model_file
		self.vertices = []
		self.faces = {}
		self.triangles = []
		self.vts = []
		self.triaTexture = []

	def read_data(self, case = 0):
		try:
			f_generator = open(self.model_file,'r')
		except Exception as e:
			print(e)
		else:
			textureID = 0
			self.faces = {'0':[]}
			modelName = os.path.split(self.model_file)[-1][:-4]
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

					if 'g Polygonal_Model_' in line and 'Triangles_' in line:
						textureID = line.strip().split('Triangles_')[-1]
						textureFigure =  modelName + 'Image' + textureID
	
						if textureFigure not in self.faces.keys():
							self.faces[textureFigure] = []
					elif 'usemtl m' in line:
						textureID = line.strip().split('usemtl m')[-1]
						# textureID =  textureFigure.split('_')[-1]
						textureFigure = modelName + 'Image' + textureID
						if textureFigure not in self.faces.keys():
							self.faces[textureFigure] = []
					elif line[:2] == 'f ':
						f,_index1,_index2,_index3 = line.strip().split(' ')
			
						try:
							coord_index1, vt_index1 = _index1.split('/')
							coord_index2, vt_index2 = _index2.split('/')
							coord_index3, vt_index3 = _index3.split('/')
							#### 要求三点不共线，去除直线三角形
							if len(np.unique([coord_index1, coord_index2, coord_index3])) == 3:	
								textureFigure = list(self.faces.keys())[-1]
								self.faces[textureFigure].append([int(coord_index1)-1, int(vt_index1)-1, int(coord_index2)-1, int(vt_index2)-1, int(coord_index3)-1, int(vt_index3)-1])
						except:
							coord_index1 = _index1
							coord_index2 = _index2
							coord_index3 = _index3
							if len(np.unique([coord_index1, coord_index2, coord_index3])) == 3:
								textureFigure = list(self.faces.keys())[-1]
								self.faces[textureFigure].append([int(coord_index1)-1, int(0), int(coord_index2)-1, int(0), int(coord_index3)-1, int(0)])
				

						
	def get_faces(self):
		# print('reading faces from obj file...')
		self.read_data(case = 3)
		return self.faces

	def get_vertices(self):
		# print('reading vertices from obj files...')
		if self.vertices == []:
			self.read_data(case = 1)
		# print('total vertices: ', len(self.vertices))
		return self.vertices, np.array(self.vertices).astype(np.float64)

	def get_vts(self):
		if self.vts == []:
			self.read_data(case = 2)
		return np.array(self.vts).astype(np.float64)

	def get_triangles(self):
		faces = self.get_faces()
		vertices_str, vertices = self.get_vertices()
		triangles_str = []
		for textureFigure, curfaces in faces.items():
			for index in curfaces:
				triangle = [vertices[index[0]], vertices[index[2]], vertices[index[4]]]
				self.triangles.append(triangle)
				tria_str = [vertices_str[index[0]], vertices_str[index[2]], vertices_str[index[4]]]
				triangles_str.append(tria_str)
		self.triangles = np.array(self.triangles).astype(np.float32)
		# print('total triangles: ',len(self.triangles))
		return self.triangles, faces, vertices, np.array(triangles_str)

	def get_normals(self):
		if self.triangles == []:
			self.get_triangles()
		normals = []
		for triangle in self.triangles:
			v11 = triangle[1, :] - triangle[0, :]
			v22 = triangle[2, :] - triangle[0, :]
			normal = np.cross(v11, v22)
			normal_unit = normal / np.linalg.norm(normal)
			normals.append(normal_unit)
		self.normals = np.array(normals)
		return self.normals

	def vts2textureXY(self, uv, width, height):
		tXY = [int(uv[0]*(width-1)), (height-1) - int(uv[1]*(height-1))]
		return tXY


	def get_triangles_with_texture(self, rgb = True):
		vertices_str, vertices = self.get_vertices()
		faces = self.get_faces()
		if rgb:
			vts = self.get_vts()
		

		textureFigure_index = []
		fuseFace = []
		trias_str_list = []
		for textureFigure, curFaces in faces.items():
			if textureFigure=='0' or 'Image0' in textureFigure:
				continue

			if rgb:
				texture_file = os.path.split(self.model_file)[0] + '/' + textureFigure.replace(' ','_') + '.png'
				print(texture_file)
				img = cv2.imread(texture_file)
				height, width, channel = img.shape
			
			for curfaceInd in curFaces:
				curfaceInd = np.array(curfaceInd).reshape(-1,2)
				tria_xyzuvrgb = []
				tria_xyz_str = []
				for (vi, vtj) in curfaceInd:
					xyz = vertices[vi]
					xyz_str = vertices_str[vi]
					if rgb:
						uv = vts[vtj]
						tXY = self.vts2textureXY(uv, width, height)
						rgb_value = img[tXY[1], tXY[0],:]
						tria_xyzuvrgb.append([xyz[0], xyz[1], xyz[2], uv[0], uv[1], rgb_value[0], rgb_value[1],rgb_value[2]])
						tria_xyz_str.append(xyz_str)
					else:
						tria_xyzuvrgb.append([xyz[0], xyz[1], xyz[2]])
						tria_xyz_str.append(xyz_str)
    					
						
				self.triaTexture.append(tria_xyzuvrgb)
				textureFigure_index.append(textureFigure)
				fuseFace.append(curfaceInd[:,0].transpose())
				trias_str_list.append(tria_xyz_str)
				
		self.triaTexture = np.array(self.triaTexture)
		fuseFace = np.array(fuseFace)
		textureFigure_index = np.array(textureFigure_index)
		return self.triaTexture, fuseFace, textureFigure_index, trias_str_list


	def change_texture(self, objectiveTexture):
		pass



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


