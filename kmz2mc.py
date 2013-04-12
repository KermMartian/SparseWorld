#!/usr/bin/python
import sys
sys.path.append("../pymclevel/")
import mclevel
import collada
import numpy as np
from numpy import array
import math
import mcBlockData
import scipy.optimize as scop

epsilon = 1e-2

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

		# Step 3: Find the length of each side
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

		Linc = 1 /L
		Lspan = np.ceil(L)
		Lspaces = [np.linspace(0,1,1+Lspan[i]) for i in indices]
		bary_coords = np.zeros(3)
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
					self.castraythroughvoxels(cart_coords,snorm) 
	
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

		while 1:

			# Invoke the callback, unless we are not *yet* within the bounds of the world
			if not(x < 0 or y < 0 or z < 0):
				#print("Block at %f %f %f" % (np.floor(x), np.floor(y), np.floor(z)))
				self.voxchg += 1
				self.arr3d_id[x,y,z] = 1
				self.arr3d_dt[x,y,z] = 1

			# tMaxX stores the t-value at which we cross a cube boundary along the
			# X axis, and similarly for Y and Z. Therefore, choosing the least tMax
			# chooses the closest cube boundary. Only the first case of the four
			# has been commented in detail.
			if tMaxX < tMaxY:
				if tMaxX < tMaxZ:
					if tMaxX > radius:
						break
					# Update which cube we are now in.
					x += stepX
					# Adjust tMaxX to the next X-oriented boundary crossing.
					tMaxX += tDeltaX
					# Record the normal vector of the cube face we entered.
					continue
			else:
				if tMaxY < tMaxZ:
					if tMaxY > radius:
						break
					y += stepY
					tMaxY += tDeltaY
					continue

			if tMaxZ > radius:
				break;
			z += stepZ
			tMaxZ += tDeltaZ

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

	def geom_c2vcd(self,coord,axis):
		return round(coord + self.voffset[axis]) - 0.5

	def geom_c2vcu(self,coord,axis):
		return round(coord + self.voffset[axis]) + 0.5

	def geom_vc2c(self,coord,axis):
		return np.round(coord - self.voffset[axis] - 0.5)

	def geom_eucdist(self,p1,p2):
		s = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2
		return math.sqrt(s)

	def geom_cart2bary_elf(self,tri,x,y,z):
		if x == None:
			a = 1
			b = 2
			c = 0
		elif y == None:
			a = 0
			b = 2
			c = 1
		else:
			a = 0
			b = 1
			c = 2

		reindex = array([a,b,c])
		trimatrix = np.matrix(tri[reindex]).transpose()  
		#instead of trimatrix = matrix(tris).transpose()
		P = np.matrix([[x],[y],[z]])[reindex]
		#instead of P =matrix([[x],[y],[z]])

		offset = trimatrix[:,0]
		centered = trimatrix - offset # translate to origin
		theta_x = math.atan2(centered[2,2],centered[1,2])
		rx = np.matrix([[1,0,0],[0,math.cos(-theta_x),-math.sin(-theta_x)],[0,math.sin(-theta_x),math.cos(-theta_x)]])
		intermediate1 = rx*centered
		theta_z = math.atan2(intermediate1[1,2],intermediate1[0,2])
		rz = np.matrix([[math.cos(-theta_z),-math.sin(-theta_z),0],[math.sin(-theta_z),math.cos(-theta_z),0],[0,0,1]])
		intermediate2 = rz*intermediate1
		theta_x2 = math.atan2(intermediate2[2,1],intermediate2[1,1])
		rx2 = np.matrix([[1,0,0],[0,math.cos(-theta_x2),-math.sin(-theta_x2)],[0,math.sin(-theta_x2),math.cos(-theta_x2)]])
		planar = rx2*intermediate2

		P[2] = Unknown("z")
		Pplanar = rx2*rz*rx*(P-offset) 

		# Do something here to set tolerances
		# and take care of rounding error, if it seems like it matters

		T = np.matrix([[(planar[0,0]-planar[0,2]),(planar[0,1]-planar[0,2])],[(planar[1,0]-planar[1,2]),(planar[1,1]-planar[1,2])]])
		det = np.linalg.det(T)
		if det == 0:        # Incoming div-by-0
			print("Determinant: Div-by-0 warning")
			return [0., 0., 0., x, y, z]
		lam12 = np.linalg.inv(T) * (Pplanar[:2]-planar[:2,2])
		lambda1 = lam12[0,0]
		lambda2 = lam12[1,0]
		lambda3 = 1. - lambda1 - lambda2
		rval = [lambda1,lambda2,lambda3,x,y,z]
		rval[3+c] = lambda1*trimatrix[2,0]+lambda2*trimatrix[2,1]+lambda3*trimatrix[2,2]

		# feel free to put better guesses if you have them
		# but it's linear so I think it should be okay?
		solved = (rval[3+c] - P[2,0]).collapse_on('z')
		real_z = -solved.sterm / solved.mterm
		if solved(z=real_z) < epsilon:
			rval[0] = rval[0](z=real_z)
			rval[1] = rval[1](z=real_z)
			rval[2] = rval[2](z=real_z)
			rval[3+c] = rval[3+c](z=real_z)
		else:
			print "Found no solution to equation %s=0" % (str(rval[3+c] - P[2,0]))
			return [0., 0., 0., x, y, z]

		return rval

	def geom_cart2bary(self,tri,x,y,z):
		P = array([x,y,z])
		if x == None:
			a = 1
			b = 2
			c = 0
		elif y == None:
			a = 0
			b = 2
			c = 1
		else:
			a = 0
			b = 1
			c = 2

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

	def geom_bary(self,tri,x,y,z):


		# Get the barycentric coordinates of the center of this quad
		[l1, l2, l3, x, y, z] = self.geom_cart2bary(tri, x, y, z)
		if l1 == 0. and l2 == 0. and l3 == 0.:
			return [x, y, z, False]

		inside = l1>=0 and l1<=1 and l2>=0 and l2<=1 and l3>=0 and l3<=1

		# give back the results
		return [x, y, z, inside]


	def geom_mat(self,tri,x,y,z,timg,ttex,majax,smjax,minax):

		if ttex == None or timg == None:
			return [1, 0]

		# Step 1: Grab all the points
		p1 = [x, y, z]
		p1[majax] -= 0.5
		p1[smjax] -= 0.5
		p1 = self.geom_cart2bary(tri, p1[0], p1[1], p1[2])
		if p1[0] == 0 and p1[2] == 0 and p1[2] == 0:
			return [1, 0]

		p2 = [x, y, z]
		p2[majax] += 0.5
		p2[smjax] -= 0.5
		p2 = self.geom_cart2bary(tri, p2[0], p2[1], p2[2])
		if p2[0] == 0 and p2[2] == 0 and p2[2] == 0:
			return [1, 0]

		p3 = [x, y, z]
		p3[majax] -= 0.5
		p3[smjax] += 0.5
		p3 = self.geom_cart2bary(tri, p3[0], p3[1], p3[2])
		if p3[0] == 0 and p3[2] == 0 and p3[2] == 0:
			return [1, 0]

		p4 = [x, y, z]
		p4[majax] += 0.5
		p4[smjax] += 0.5
		p4 = self.geom_cart2bary(tri, p4[0], p4[1], p4[2])
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
		pxsum = [0, 0, 0]

		for a in np.linspace(0,1,d):
			for b in np.linspace(0,1,c):
				cx = a*p3[0]+(1-a)*p1[0]+a*(p4[0]-p3[0])*b+(1-a)*(p2[0]-p1[0])*b
				cy = a*p3[1]+(1-a)*p1[1]+a*(p4[1]-p3[1])*b+(1-a)*(p2[1]-p1[1])*b
				# Step 3: Grab the pixel there
				pxc = array([cx % timg.uintarray.shape[1], \
		             (timg.uintarray.shape[0] - cy) % timg.uintarray.shape[0]])
				if pxc[1] >= timg.uintarray.shape[0] or pxc[0] >= timg.uintarray.shape[1]:
					continue
				pixel = timg.uintarray[pxc[1],pxc[0]]
				pxcount += 1
				pxsum = [pxsum[0]+pixel[0], pxsum[1]+pixel[1], pxsum[2]+pixel[2]]

		data, damage = [1, 0]
		if pxcount > 0:
			self.pxscan += pxcount
			pixel = [float(pxsum[0])/float(pxcount), \
					 float(pxsum[1])/float(pxcount), \
					 float(pxsum[2])/float(pxcount)]
			data, damage = mcBlockData.nearest(int(pixel[0]),int(pixel[1]),int(pixel[2]))
		return [data, damage]

def main():
	if len(sys.argv) < 2:
		print("Usage: %s <KMZ_file>" % sys.argv[0])
		return

	filename = sys.argv[1]
	mesh = collada.Collada(filename, ignore=[collada.DaeUnsupportedError,
	               collada.DaeBrokenRefError])

	maxs = array([-1e99,-1e99,-1e99])
	mins = array([ 1e99, 1e99, 1e99])
	for geom in mesh.geometries:
		for triset in geom.primitives:
			maxs = array([max(maxs[0],np.max(triset.vertex[:,0])), \
			              max(maxs[1],np.max(triset.vertex[:,1])), \
			              max(maxs[2],np.max(triset.vertex[:,2]))])
			mins = array([min(mins[0],np.min(triset.vertex[:,0])), \
			              min(mins[1],np.min(triset.vertex[:,1])), \
			              min(mins[2],np.min(triset.vertex[:,2]))])

	# Get some sort of scaling information
	scale = [.01,.01,.01]
	if mesh.assetInfo != None and mesh.assetInfo.unitmeter != None:
		print("This model contains units, %f %s per meter" % (mesh.assetInfo.unitmeter, mesh.assetInfo.unitname))
		scale = mesh.assetInfo.unitmeter
		scale = [scale, scale, scale]

	t2v = Tri2Voxel(mesh)
	t2v.scale = array(scale)
	t2v.geom_prep(mins,maxs)

	for geom in mesh.geometries:
		for triset in geom.primitives:
			trilist = list(triset)
			for tri in trilist:
				t2v.geom_tri2voxel(tri)

	# Print some stats
	ar1  = np.count_nonzero(t2v.arr3d_id)
	ar01 = np.prod(t2v.arrdim)
	print("%d/%d voxels filled (%.2f%% fill level)" % (ar1,ar01,100*ar1/ar01))
	print("t2v reports %d voxels changed" % t2v.voxchg)
	
	# Paste into MC level
	
	level = mclevel.fromFile("/home/christopher/.minecraft/saves/Cemetech/level.dat")
	t2v.arr3d_id = np.fliplr(t2v.arr3d_id)
	t2v.arr3d_dt = np.fliplr(t2v.arr3d_dt)
	for x in xrange(0,int(np.ceil(t2v.arrdim[0]/16.))):
		for z in xrange(0,int(np.ceil(t2v.arrdim[1]/16.))):

			chunk = level.getChunk(x,z)
			xmax = min(16,t2v.arrdim[0]-16*x)
			zmax = min(16,t2v.arrdim[1]-16*z)

			chunk.Blocks[0:xmax,0:zmax,34:(34+t2v.arrdim[2])] = \
			      t2v.arr3d_id[(16*x):(16*x+xmax),(16*z):(16*z+zmax),:]
			chunk.Data[0:xmax,0:zmax,34:(34+t2v.arrdim[2])] = \
			      t2v.arr3d_dt[(16*x):(16*x+xmax),(16*z):(16*z+zmax),:]

			chunk.chunkChanged()

	print("Relighting level...")
	level.generateLights()
	print("Saving level...")
	level.saveInPlace()

# Get running!
if __name__ == '__main__':
	main()
