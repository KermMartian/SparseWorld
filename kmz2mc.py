#!/usr/bin/python
import os
import sys
sys.path.append("./TopoMC/pymclevel") #"../pymclevel/")
sys.path.append("./TopoMC") #"../pymclevel/")
import mclevel
import collada
import numpy as np
from numpy import array
from xml.dom import minidom
import zipfile
import yaml
import math
import mcBlockData
import nbt
import time					# For progress timing
import argparse

epsilon = 1e-5

class Tri2Voxel:

	def __init__(self,model):
		self.offset = array([0,0,0])	# Offset applied to incoming tris
		self.scale  = array([1,1,1])	# Scaling applied to incoming tris
		self.voffset= array([0,0,0])	# Offset of the origin voxel's vertex
		self.arr3d_id = array([])		# Nice big array :P
		self.arr3d_dt = array([])		# Nice big array :P
		self.model = model
		self.voxchg = 0
		self.pxscan = 0

	def geom_prep(self,mins,maxs):
		self.offset = 1-array([mins[0],mins[1],mins[2]])
		self.arrdim = (np.ceil((maxs[0]-mins[0])*self.scale[0]+2), \
		               np.ceil((maxs[1]-mins[1])*self.scale[1]+2), \
		               np.ceil((maxs[2]-mins[2])*self.scale[2]+2))
		print("Reserving %d x %d x %d array..." % (self.arrdim[0], self.arrdim[1], self.arrdim[2]))
		self.arr3d_id = np.zeros(self.arrdim)
		self.arr3d_dt = np.zeros(self.arrdim)

		self.tvoffset = array([self.offset,]*3)
		self.tvscale  = array([self.scale,]*3)

	''' Convert a triangle in 3-space
	''  into voxels. arr3d is a reference
	''  to a 3D array, triangle is the triangle
	''  being converted, and offset and scale are
	''  used to scale model coordinates to voxel
	''  coordinates. Offset is applied BEFORE scale.
	'''
	def geom_tri2voxel(self,triangle):

		tv = triangle.vertices
		tt = triangle.texcoords
		try:
			tm = self.model.materials[triangle.material+"ID"]
			ti = tm.effect.params[0].image
		except:
			tm = None
			ti = None

		if tv.shape != (3,3):
			print("Warning: Bad triangle shape:")
			print(tv.shape)
			return

		# Step 1: Offset the triangle in 3-space
		# Step 2: Scale the triangle in 3-space
		tv = tv + self.tvoffset
		tv = tv * self.tvscale

		# Step 3: Find the length of each side, compute normal
		indices=np.arange(3)
		oboe = (indices+1)%3
		# First get surface normal and edge lengths! :)
		# e = tv[oboe] - tv[indices] # <- cool way to do it
		e = array([tv[oboe[i]] - tv[i] for i in indices]) # <- boring way

		#L1=np.linalg.norm(e1)
		#L2=np.linalg.norm(e2)
		#L3=np.linalg.norm(e3)
		L = np.apply_along_axis(np.linalg.norm,-1,e)

		snorm = np.cross(e[1],e[0])
		slen  = np.linalg.norm(snorm)
		if slen == 0:
			print("Discarding triangle with point normal")
			return
		snorm = snorm/slen

		# Step 4: Do some sort of preprocessing to find tm/ti/tt coordinates
		txc = None
		if tt and ti:
			txs	= ti.uintarray.shape
			txc = np.dot(tt[0],array([[txs[1],0],[0,txs[0]]]))

		# Step 5: Iterate over this triangle
		Linc = 1 /L
		Lspan = np.ceil(L)
		Lspaces = [np.linspace(0,1,1+Lspan[i]) for i in indices]
		bary_coords = np.zeros(3)

		omitaxis = self.geom_findnonplanar(snorm)			# For texturing
		majax = 1 if omitaxis == 0 else 0
		smjax = 2 if omitaxis != 2 else 1
		minax = omitaxis

		for i,j in np.vstack((indices,oboe)).transpose():
			for ca in Lspaces[i]:
				for cb in Lspaces[j]:
					if ca+cb > 1:
						break
					bary_coords.fill(1 - ca - cb)
					bary_coords[i] = ca
					bary_coords[j] = cb
					# here we will need to have calculated matrices
					# to take us from barycentric coords to 3-space
					# Cast ray must take two args, src and dir
					# and moves from src to src+dir
					cart_coords = self.geom_makecart(tv,bary_coords)
					voxels = self.castraythroughvoxels(cart_coords,snorm) 

					for voxel in [voxels]:
						a, b, c = self.geom_vc2cs(voxel)
						if self.arr3d_id[a, b, c] != 0 or a < 0 or b < 0 or c < 0:
							# Avoid replacing voxels. Todo: majority vector rules
							# Also avoid wrapping up to the ceiling
							continue;
						voxel[omitaxis] = None
						x, y, z = voxel
						mat, dat = self.geom_mat(tv,x,y,z,ti,txc,majax,smjax,minax)

						try:
							self.arr3d_id[a,b,c] = mat
							self.arr3d_dt[a,b,c] = dat
							self.voxchg += 1
						except IndexError:
							pass #print("Warning: couldn't index (%d,%d,%d) in output matrices" % (a,b,c))
	
	def geom_findnonplanar(self,normal):
		for omit in [2, 1, 0]:
			norm = array([0., 0., 0.])
			norm[omit] = 1.
			if epsilon < abs(np.dot(normal,norm)):
				return omit
			
	def castraythroughvoxels(self,origin,direction,radius=1):

		# Cube containing origin point.
		x = origin[0] #np.floor(origin[0]);
		y = origin[1] #np.floor(origin[1]);
		z = origin[2] #np.floor(origin[2]);
		# Break out direction vector.
		dx = direction[0] if abs(direction[0]) != 0 else 0.
		dy = direction[1] if abs(direction[1]) != 0 else 0.
		dz = direction[2] if abs(direction[2]) != 0 else 0.
		# Direction to increment x,y,z when stepping.
		stepX = self.signum(dx);
		stepY = self.signum(dy);
		stepZ = self.signum(dz);
		# See description above. The initial values depend on the fractional
		# part of the origin.
		tMaxX = self.intbound(origin[0], dx);
		tMaxY = self.intbound(origin[1], dy);
		tMaxZ = self.intbound(origin[2], dz);
		# The change in t when taking a step (always positive).
		tDeltaX = stepX/dx if dx != 0 else np.inf;
		tDeltaY = stepY/dy if dy != 0 else np.inf;
		tDeltaZ = stepZ/dz if dz != 0 else np.inf;

		# Avoids an infinite loop.
		if dx == 0 and dy == 0 and dz == 0:
			print("Warning: direction vector cast in zero direction")
			return

		# Rescale from units of 1 cube-edge to units of 'direction' so we can
		# compare with 't'.
		radius /= math.sqrt(dx*dx+dy*dy+dz*dz);

		# Total T and return value
		tSigma = 0
		tDelta = 0
		rval = [0., None]

		while 1:

			# Invoke the callback, unless we are not *yet* within the bounds of the world
			if tDelta > rval[0]:
#				print("Block at %s vec pct %f" % (str(prevpos),tDelta))
				rval = [tDelta, array(prevpos)]

			prevpos = [x,y,z]
			tSigma += tDelta

			# tMaxX stores the t-value at which we cross a cube boundary along the
			# X axis, and similarly for Y and Z. Therefore, choosing the least tMax
			# chooses the closest cube boundary. Only the first case of the -f-o-u-r-
			# three cases has been commented in detail.
			if tMaxX < tMaxY:
				if tMaxX < tMaxZ:
					# Update which cube we are now in.
					x += stepX
					# Adjust tMaxX to the next X-oriented boundary crossing.
					tMaxX += tDeltaX
					# Record the deltaT for the cube we just left
					tDelta = tDeltaX
					if tMaxX > radius:
						break
					continue
			else:
				if tMaxY < tMaxZ:
					if tMaxY > radius:
						break
					y += stepY
					tMaxY += tDeltaY
					tDelta = tDeltaY
					continue

			if tMaxZ > radius:
				break;
			z += stepZ
			tMaxZ += tDeltaZ
			tDelta = tDeltaZ

		tDelta = 1. - tSigma
		
		if tDelta > rval[0]:
#			print("Block at %s vec pct %f" % (str(prevpos),tDelta))
			rval = [tDelta, array(prevpos)]

		return rval[1]

	def intbound(self, s, ds):
		if ds < 0:
			return self.intbound(-s, -ds)
		elif ds == 0:
			return np.inf
		else:
			s = self.modulus(s, 1)
			return (1-s)/ds

	def signum(self, x):
		return 1 if x > 0 else (0 if abs(x) == 0 else -1)

	def modulus(self, value, modulus):
		return (value % modulus + modulus) % modulus
		
	def geom_makecart(self,tri,bary):
		return tri[0]*bary[0]+tri[1]*bary[1]+tri[2]*bary[2]

	def geom_vc2cs(self,coords):
		return np.round(coords - self.voffset)

	def geom_vc2c(self,coord,axis):
		return np.round(coord - self.voffset[axis])

	def geom_eucdist(self,p1,p2):
		s = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2
		return math.sqrt(s)

	def geom_cart2bary(self,tri,x,y,z,minax):
		P = array([x,y,z])
		if minax == 0:
			a = 1
			b = 2
		elif minax == 1:
			a = 0
			b = 2
		else:
			a = 0
			b = 1
		c = minax

		det = (tri[1,b]-tri[2,b])*(tri[0,a]-tri[2,a]) + \
		      (tri[2,a]-tri[1,a])*(tri[0,b]-tri[2,b])

		if det != 0.:		# No incoming div-by-0
			# Compute barycentric coordinates
			l1 = ((tri[1,b]-tri[2,b])*(P[a]-tri[2,a]) + \
				  (tri[2,a]-tri[1,a])*(P[b]-tri[2,b]))/det;
			l2 = ((tri[2,b]-tri[0,b])*(P[a]-tri[2,a]) + \
				  (tri[0,a]-tri[2,a])*(P[b]-tri[2,b]))/det;
			l3 = 1. - l1 - l2

		else:				# Incoming divide-by-zero.
			print("Determinant: Div-by-0 warning %s %s %s" % (str(x),str(y),str(z)))
			return [0., 0., 0., x, y, z]

		rval = [l1, l2, l3, x, y, z]
		rval[3+c] = (l1*tri[0,c]) + (l2*tri[1,c]) + (l3*tri[2,c])

		return rval

	def geom_mat(self,tri,x,y,z,timg,ttex,majax,smjax,minax):

		if ttex == None or timg == None:
			return [1, 0]

		# Step 1: Grab all the points
		p1 = [x, y, z]
		p1[majax] -= 0.5
		p1[smjax] -= 0.5
		p1 = self.geom_cart2bary(tri, p1[0], p1[1], p1[2],minax)
		if p1[0] == 0 and p1[2] == 0 and p1[2] == 0:
			return [1, 0]

		p2 = [x, y, z]
		p2[majax] += 0.5
		p2[smjax] -= 0.5
		p2 = self.geom_cart2bary(tri, p2[0], p2[1], p2[2],minax)
		if p2[0] == 0 and p2[2] == 0 and p2[2] == 0:
			return [1, 0]

		p3 = [x, y, z]
		p3[majax] -= 0.5
		p3[smjax] += 0.5
		p3 = self.geom_cart2bary(tri, p3[0], p3[1], p3[2],minax)
		if p3[0] == 0 and p3[2] == 0 and p3[2] == 0:
			return [1, 0]

		p4 = [x, y, z]
		p4[majax] += 0.5
		p4[smjax] += 0.5
		p4 = self.geom_cart2bary(tri, p4[0], p4[1], p4[2],minax)
		if p4[0] == 0 and p4[2] == 0 and p4[2] == 0:
			return [1, 0]


		# Step 2: Map modified texture coordinates to pixel coords
		p1 = np.dot(p1[0:3],ttex)
		p2 = np.dot(p2[0:3],ttex)
		p3 = np.dot(p3[0:3],ttex)
		p4 = np.dot(p4[0:3],ttex)

		# Step 3: Scan/rasterize over all pixels
		c = 1+abs(max(p2[0]-p1[0],p4[0]-p3[0]))
		d = 1+abs(max(p3[1]-p1[1],p4[1]-p2[1]))

		# Cheap trimming. Appears to be ok.
		c = c if c < 7 else 7
		d = d if d < 7 else 7

		pxcount = 0
		pxsum = [0.]*len(timg.uintarray[0,0])

		for a in np.linspace(0,1,d):
			for b in np.linspace(0,1,c):
				cx = a*p3[0]+(1-a)*p1[0]+a*(p4[0]-p3[0])*b+(1-a)*(p2[0]-p1[0])*b
				cy = a*p3[1]+(1-a)*p1[1]+a*(p4[1]-p3[1])*b+(1-a)*(p2[1]-p1[1])*b
				# Step 3: Grab the pixel there
				pxc = array([self.modulus(cx,timg.uintarray.shape[1]), \
		             self.modulus((timg.uintarray.shape[0] - cy),timg.uintarray.shape[0])])
				if np.isnan(pxc[0]) or np.isnan(pxc[1]):
					continue
				if pxc[1] >= timg.uintarray.shape[0] or pxc[0] >= timg.uintarray.shape[1]:
					continue
				pixel = timg.uintarray[pxc[1],pxc[0]]
				pxcount += 1
				pxsum = pxsum + pixel

		data, damage = [1, 0]
		if pxcount > 0:
			self.pxscan += pxcount
			pixel = pxsum/float(pxcount)
			if len(pixel) >= 4 and pixel[3] < 127:
				if pixel[3] < 1:
					data, damage = [0, 0]		# air
				else:
					data, damage = [20, 0]		# glass
			else:
				data, damage = mcBlockData.nearest(int(pixel[0]),int(pixel[1]),int(pixel[2]))
		return [data, damage]

class ModelRecurse:
	def __init__(self):
		self.abort = False

	def recurse_model(self,model,mode,ind):
		for node in model.scenes[0].nodes:
			ind = self.recurse_dive(node,mode,None,ind)
			if self.abort:
				break;

		self.abort = False
		return ind

	def recurse_dive(self,node,mode,xform,ind):
		xform2 = None

		# Deal with fetching and possibly combining transforms
		if node.transforms:
			xform2 = node.transforms[0].matrix
			if None != xform:
				xform = np.dot(xform,xform2)
			else:
				xform = xform2

		# Deal with the geometry, if it has any
		for child in node.children:
			if isinstance(child,collada.scene.NodeNode):
				ind = self.recurse_dive(child.node,mode,xform,ind)
			elif isinstance(child,collada.scene.Node):
				ind = self.recurse_dive(child,mode,xform,ind)
			elif isinstance(child,collada.scene.GeometryNode):
				ind = self.recurse_geometry(child.geometry,mode,xform,ind)
			else:
				print xform
				print("Found an unknown %s" % (type(child)))

			if self.abort:
				break;
		return ind

	def recurse_geometry(self,node,mode,xform,ind):
		if mode == 'extents':
			mins, maxs = ind
			for triset in node.primitives:
				if not(isinstance(triset,collada.triangleset.TriangleSet)):
					print("Warning: ignoring primitive of type %s" % type(triset))
					continue

				# Apply the transform, if there is one
				if None != xform:
					oshape = triset.vertex.shape
					v = np.ones((oshape[0],1+oshape[1]))
					v[:,:3] = triset.vertex
					v = v.transpose()
					v = np.dot(xform,v)
					v = v.transpose()

				else:
					v = triset.vertex
				
				maxs = array([max(maxs[0],np.max(v[:,0])), \
							  max(maxs[1],np.max(v[:,1])), \
							  max(maxs[2],np.max(v[:,2]))])
				mins = array([min(mins[0],np.min(v[:,0])), \
							  min(mins[1],np.min(v[:,1])), \
							  min(mins[2],np.min(v[:,2]))])

			print("Scanned geometry '%s' for extents" % node.name)
			return [mins, maxs]

		elif mode == 'convert':
			starttime = time.time()
			startvox = ind.voxchg
			for triset in node.primitives:
				if not(isinstance(triset,collada.triangleset.TriangleSet)):
					print("Warning: ignoring primitive of type %s" % type(triset))
					continue

				trilist = list(triset)
				for tri in trilist:
					# Apply the transform, if there is one
					otv = tri.vertices
					if None != xform:
						tv = np.ones((otv.shape[0], 1 + otv.shape[1]))
						tv[:,:3] = otv
						tv = tv.transpose()
						tv = np.dot(xform,tv)
						tv = tv[:3,:]
						tv = tv.transpose()
						tri.vertices = tv
				
					ind.geom_tri2voxel(tri)
					tri.vertices = otv

			print("Converted geometry '%s' in %d s, changed %d voxels" % (node.name,time.time()-starttime,ind.voxchg-startvox))
		else:
			print("Warning: skipping geometry for unknown mode '%s'" % mode)
		return ind
		

def main():

    # parse options and get results
	parser = argparse.ArgumentParser(description='Converts a single building from a Collada file and pastes into a Minecraft world')
	parser.add_argument('--model', required=True, type=str, help='relative or absolute path to .kmz file containing Collada model and assets')
	parser.add_argument('--debug', action='store_true', help='enable debug output')
	parser.add_argument('--world', required=True, type=str, help='path to main folder of a target Minecraft world')
	args = parser.parse_args()

	filename = args.model

	# Determine where to paste into target world
	zipf = zipfile.ZipFile(args.model, 'r')
	kmldata = minidom.parse(zipf.open('doc.kml'))
	zipf = None
	location = kmldata.getElementsByTagName('Location')[0]
	latitude  = float(location.getElementsByTagName('latitude')[0].childNodes[0].data)
	longitude = float(location.getElementsByTagName('longitude')[0].childNodes[0].data)
	altmode   = str(kmldata.getElementsByTagName('altitudeMode')[0].childNodes[0].data)
	altitude  = float(location.getElementsByTagName('altitude')[0].childNodes[0].data)
	kmldata = None
	
	# Get information about the target world
	yamlfile = open(os.path.join(args.world, 'Region.yaml'), 'r')
	yamlfile.readline()				# discard first line
	myRegion = yaml.safe_load(yamlfile)
	yamlfile.close()

	# Compute the world (x,y) for this model
	llextents = myRegion['wgs84extents']['elevation']
	metersPerLat =  (myRegion['tiles']['ymax'] - myRegion['tiles']['ymin']) * myRegion['tilesize']
	metersPerLat /= (llextents['ymax'] - llextents['ymin'])
	metersPerLon =  (myRegion['tiles']['xmax'] - myRegion['tiles']['xmin']) * myRegion['tilesize']
	metersPerLon /= (llextents['xmax'] - llextents['xmin'])
	modelBaseLoc = [ myRegion['tiles']['xmin'] * myRegion['tilesize'] + \
	                 ((longitude - llextents['xmin']) * metersPerLon), \
	                 myRegion['tiles']['ymax'] * myRegion['tilesize'] - \
	                 ((latitude  - llextents['ymin']) * metersPerLat), 0 ]	#Because Minecraft maps are flipped upside-down
	print("Loc: %f,%f => %d,%d within %s" % (latitude, longitude, modelBaseLoc[0], modelBaseLoc[1], str(llextents)))

	# Open the model and determine its extents
	model = collada.Collada(filename, ignore=[collada.DaeUnsupportedError,
	               collada.DaeBrokenRefError])
	maxs = array([-1e99,-1e99,-1e99])
	mins = array([ 1e99, 1e99, 1e99])
	mr = ModelRecurse()
	mins, maxs = mr.recurse_model(model,"extents",[mins,maxs])
	print("Computed model extents: [%f, %f, %f,] to [%f, %f, %f]" % (mins[0], mins[1], mins[2], 
                                                                     maxs[0], maxs[1], maxs[2]))

	# some sort of scaling information
	scale = [.01,.01,.01]
	if model.assetInfo != None and model.assetInfo.unitmeter != None:
		print("This model contains units, %f %s per meter" % (model.assetInfo.unitmeter, model.assetInfo.unitname))
		scale = model.assetInfo.unitmeter
		scale = [scale, scale, scale]

	t2v = Tri2Voxel(model)
	t2v.scale = array(scale)
	t2v.geom_prep(mins,maxs)

	# Use extents and modelBaseLoc to compute the world coordinate that
	# corresponds to the output array's [0,0,0]
	cornerBase = t2v.tvoffset[0] * t2v.tvscale[0]
	modelBaseLoc -= cornerBase
	modelBaseLoc = [round(x) for x in modelBaseLoc]

	# Convert and fix orientation
	mr.recurse_model(model,"convert",t2v)		# Do the conversion!
	t2v.arr3d_id = np.fliplr(t2v.arr3d_id)		# Fix block ID array
	t2v.arr3d_dt = np.fliplr(t2v.arr3d_dt)		# Fix damage val array

	# Print some stats
	ar1  = np.count_nonzero(t2v.arr3d_id)
	ar01 = np.prod(t2v.arrdim)
	print("%d/%d voxels filled (%.2f%% fill level)" % (ar1,ar01,100*ar1/ar01))
	print("t2v reports %d voxels changed" % t2v.voxchg)
	
	# Open MC level for pasting 
	level = mclevel.fromFile(os.path.join(args.world,"level.dat"))

	# Compute world-scaled altitude information
	if altmode == "absolute":
		sealevel = myRegion['sealevel'] if 'sealevel' in myRegion else 64
		modelAltBase = int(altitude * myRegion['vscale'] + sealevel)
	elif altmode == "relativeToGround":
		xbase = int(round(modelBaseLoc[0] + cornerBase[0]))
		zbase = int(round(modelBaseLoc[1] + cornerBase[1]))
		chunk = level.getChunk(int(xbase/16.), int(zbase/16.))
		voxcol = chunk.Blocks[xbase % 16, zbase % 16, :]
		voxtop = [i for i, e in enumerate(voxcol) if e != 0][-1] + 1
		modelAltBase = int(voxtop + modelBaseLoc[2])
		chunk = None
	else:
		print("Error: Unknown altitude mode in KML file.")
		raise IOError
	print("Model base altitude is %d meters (voxels)" % modelAltBase)
	
	# Compute new world height
	worldheight = int(modelAltBase+t2v.arrdim[2])
	worldheight |= worldheight >> 1
	worldheight |= worldheight >> 2
	worldheight |= worldheight >> 4
	worldheight |= worldheight >> 8
	worldheight |= worldheight >> 16
	worldheight += 1

	if worldheight > level.Height:
		print("World height increased from %d to %d" % \
		      (level.Height,worldheight))
		level.Height = worldheight
		level.root_tag["Data"]["worldHeight"] = nbt.TAG_Int(worldheight)
	
	chunksx = [int(np.floor(modelBaseLoc[0]/16.)), \
	           int(np.floor((modelBaseLoc[0]+t2v.arrdim[0])/16.))]
	chunksz = [int(np.floor(modelBaseLoc[1]/16.)), \
	           int(np.floor((modelBaseLoc[1]+t2v.arrdim[1])/16.))]
	for x in xrange(chunksx[0], 1+chunksx[1]):
		for z in xrange(chunksz[0], 1+chunksz[1]):

			chunk = level.getChunk(x,z)
			xmin = max(0,modelBaseLoc[0]-16*x)
			xmax = min(16,t2v.arrdim[0]+16*(chunksx[0]-x))
			zmin = max(0,modelBaseLoc[1]-16*z)
			zmax = min(16,t2v.arrdim[1]+16*(chunksz[0]-z))

			if xmax <= 0 or zmax <= 0:
				continue;

			#print("Copying %d,%d,%d to %d,%d,%d" % (xmin,modelAltBase,zmin,xmax,(modelAltBase+t2v.arrdim[2]),zmax))
			inp = chunk.Blocks[xmin:xmax,zmin:zmax, \
			                   modelAltBase:(modelAltBase+t2v.arrdim[2])]

			# Data first because Blocks must retain its 0s
			ind = chunk.Data[xmin:xmax,zmin:zmax, \
			                 modelAltBase:(modelAltBase+t2v.arrdim[2])]
			chunk.Data[xmin:xmax,zmin:zmax, \
			           modelAltBase:(modelAltBase+t2v.arrdim[2])] = \
			np.where(inp != 0, ind, \
			         t2v.arr3d_dt[(16*(x-chunksx[0])):(16*(x-chunksx[0])+(xmax-xmin)), \
			         (16*(z-chunksz[0])):(16*(z-chunksz[0])+(zmax-zmin)),:])

			# Blocks second.
			chunk.Blocks[xmin:xmax,zmin:zmax, \
			             modelAltBase:(modelAltBase+t2v.arrdim[2])] = \
			np.where(inp != 0, inp, \
			         t2v.arr3d_id[(16*(x-chunksx[0])):(16*(x-chunksx[0])+(xmax-xmin)), \
			         (16*(z-chunksz[0])):(16*(z-chunksz[0])+(zmax-zmin)),:])

			# And mark the chunk.
			chunk.chunkChanged()

	print("Relighting level...")
	level.generateLights()
	print("Saving level...")
	level.saveInPlace()

# Get running!
if __name__ == '__main__':
	main()
