    def add_step(self, particle_hull, support_hull, position=None, height=2):
        """
        Add a step edge to an existing particle and support hull
        Hardcoded for Pt and CeO2 :/
        Step height is always 2 Pt layers
        """
        hull_1 = particle_hull[0]
        hull_2 = particle_hull[1]
        hull_3 = support_hull[0]

        #project points of hull_1 and hull_2 onto the xy plane
        hull_1_points = hull_1.points[:,0:2]
        hull_2_points = hull_2.points[:,0:2]
        particle_outline = np.concatenate((hull_1_points,hull_2_points))
        particle_outline_hull = self.convex_hull(particle_outline)
        #generate a random point within the particle_outline_hull
        #point = self.random_point_in_hull(particle_outline_hull)
        center_of_outline_hull = np.mean(particle_outline_hull.points,axis=0)
        point = np.array([0.01,0.01])
        line = point-center_of_outline_hull
        orthogonal_line = np.array([-line[1],line[0]])
        normalized_orthogonal_line = orthogonal_line/np.linalg.norm(orthogonal_line)
        point_1 = point+normalized_orthogonal_line*100
        point_2 = point-normalized_orthogonal_line*100

        zs, hull_levels = self.split_hull_by_z(hull_1)
        particle_interface_intersections = []
        new_hull_levels_1 = []
        for z, hull_level in zip(zs,hull_levels):
            print(hull_level)
            hull = self.convex_hull(hull_level[:,:2])
            center_of__hull = np.mean(hull.points,axis=0)
            intersection_points = self.get_intersection_points(hull,np.array([point_1,point_2]))
            particle_interface_intersections.append(intersection_points)
            print("intersection points:",intersection_points)


            assert len(intersection_points) == 2 or len(intersection_points) == 0
            
            if len(intersection_points) == 2:
                #remove points from hull_level that are not on the same side of the line as the center of the hull
                hull_level = np.array(self.filter_points(hull_level[:,:2],intersection_points,center_of_outline_hull))
                print("hull_level after filtering:",hull_level)
                plt.figure()
                plt.scatter(hull.points[:,0],hull.points[:,1])
                plt.scatter(hull_level[:,0],hull_level[:,1],c='k')
                plt.scatter(line[0],line[1],c='r')
                plt.scatter(orthogonal_line[0],orthogonal_line[1],c='g')
                plt.scatter(point_1[0],point_1[1])
                plt.scatter(point_2[0],point_2[1])
                plt.scatter(intersection_points[0][0],intersection_points[0][1])
                plt.scatter(intersection_points[1][0],intersection_points[1][1])
                plt.axis('equal')
                plt.show()
                #add intersection points to hull_level
                intersection_points = self.add_z_coordinate(np.array(intersection_points),z)
                hull_level = self.add_z_coordinate(hull_level,z)
                print(hull_level)
                print(intersection_points)
                hull_level = np.concatenate([np.array(hull_level), np.array(intersection_points)], axis=0)

            new_hull_levels_1.append(hull_level)

        zs, hull_levels = self.split_hull_by_z(hull_2)
        new_hull_levels_2 = []
        for z, hull_level in zip(zs,hull_levels):
            hull = self.convex_hull(hull_level[:,:2])
            center_of__hull = np.mean(hull.points,axis=0)
            intersection_points = self.get_intersection_points(hull,np.array([point_1,point_2]))
            particle_interface_intersections.append(intersection_points)

            assert len(intersection_points) == 2 or len(intersection_points) == 0
            
            if len(intersection_points) == 2:
                #remove points from hull_level that are not on the same side of the line as the center of the hull
                hull_level = self.filter_points(hull_level[:,:2],intersection_points,center_of_outline_hull)
                #add intersection points to hull_level
                #add z coordinate to intersection points
                intersection_points = self.add_z_coordinate(np.array(intersection_points),z)
                hull_level = self.add_z_coordinate(np.array(hull_level),z)
                hull_level = np.concatenate([np.array(hull_level), np.array(intersection_points)], axis=0)

            new_hull_levels_2.append(hull_level)
        print("new hull levels 1:", new_hull_levels_1)
        print("new hull levels 2:",new_hull_levels_2)
        print("particle_hull[0].points",particle_hull[0].points)
        print("particle_hull[1].points",particle_hull[1].points)
        #plot particle hull in 3d
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(particle_hull[0].points[:,0], particle_hull[0].points[:,1], particle_hull[0].points[:,2], color='red')
        ax.scatter(particle_hull[1].points[:,0], particle_hull[1].points[:,1], particle_hull[1].points[:,2], color='red')
        ax.view_init(0, 45)
        ax.axis('equal')
        plt.show()

        particle_hull[0] = self.convex_hull(np.concatenate(new_hull_levels_1))
        particle_hull[1] = self.convex_hull(np.concatenate(new_hull_levels_2))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(particle_hull[0].points[:,0], particle_hull[0].points[:,1], particle_hull[0].points[:,2], color='red')
        ax.scatter(particle_hull[1].points[:,0], particle_hull[1].points[:,1], particle_hull[1].points[:,2], color='red')
        ax.view_init(0, 45)
        ax.axis('equal')
        plt.show()

        print("particle_hull[0].points",particle_hull[0].points)
        print("particle_hull[1].points",particle_hull[1].points)
        #zs, hull_levels = self.split_hull_by_z(hull_3)
        #hull_levels_interface = hull_levels[0]
        return particle_hull, support_hull 




# %%
ob_gen = ObjectGenerator()
#particle_hull = ob_gen.particle_hull(4,interface_radii=[10,11,8,7],layer_sample_points=[6,6,6,6],centers=[[0,0],[0,0],[0,0],[0,0]])
particle_hull = ob_gen.particle_hull(5,interface_radii=[10,11,9,8,7],layer_sample_points=[6,6,6,6,6],centers=[[0,0],[0,0],[0,0],[0,0],[0,0]])
#print(particle_hull[0].points)
#for i in range(100):
#    ob_gen.add_step(particle_hull.copy(),particle_hull.copy())
# %%
support_hull = ob_gen.support_hull(8,50,50,particle_interface_points=np.array([[5.69638606, 2.88445904, 0],[0.30238439, 6.99237546, 0],[-4.64750397, 4.79482633, 0],[-5.62285321, -1.09368618, 0],[-1.00215511, -4.9157151, 0],[5.76586847, -0.65974722, 0]]))
print(len(particle_hull),len(support_hull))
ob_gen.visualize_hull_points(support_hull+particle_hull)
ob_gen.mayavi_points(support_hull+particle_hull)
#stepped particle
particle_hull_2, support_hull_2 = ob_gen.add_step(particle_hull.copy(), support_hull) 
# %%
Pt_bulk = ob_gen.Pt_lattice(10,10,10)
#Pt_bulk.z -= 1
CeO2_bulk = ob_gen.Ceria_lattice(15,15,15)
#RANDOMLY DISPLACE BULK ATOMS BEFORE FILTERING


#plot the lattice, color by label
#translate label to color
filtered_Pt=ob_gen.filter_atoms_by_hull(Pt_bulk,particle_hull)
filtered_Pt_2=ob_gen.filter_atoms_by_hull(Pt_bulk,particle_hull_2)
filtered_ceo2 = ob_gen.filter_atoms_by_hull(CeO2_bulk,support_hull)
Pt_bulk.label = Pt_bulk.label.replace({'Ce': 0, 'O': 1, 'Pt': 2})
CeO2_bulk.label = CeO2_bulk.label.replace({'Ce': 0, 'O': 1, 'Pt': 2})
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(CeO2_bulk.x, CeO2_bulk.y, CeO2_bulk.z, c=CeO2_bulk.label)
ax.view_init(0, 45)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.axis('equal')
plt.show()
#%%
#plot filtered pt and filtered ceo2 in the same frame with mayavi
filtered_atoms = pd.concat([filtered_Pt_2,filtered_ceo2])
filtered_atoms_2 = pd.concat([filtered_Pt,filtered_ceo2])
filtered_atoms_2.label = filtered_atoms_2.label.replace({'Ce': 0, 'O': 1, 'Pt': 2})
filtered_atoms.label = filtered_atoms.label.replace({'Ce': 0, 'O': 1, 'Pt': 2})
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(filtered_atoms.x, filtered_atoms.y, filtered_atoms.z, c=filtered_atoms.label)
ax.view_init(0, 45)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.axis('equal')
plt.show()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(filtered_atoms.x, filtered_atoms.y, filtered_atoms.z, c=filtered_atoms.label)
ax.view_init(90,0)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.axis('equal')
plt.show()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(filtered_atoms_2.x, filtered_atoms_2.y, filtered_atoms_2.z, c=filtered_atoms_2.label)
ax.view_init(90,0)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.axis('equal')
plt.show()
#plot particle_hull and particle_hull_2
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(particle_hull[0].points[:,0], particle_hull[0].points[:,1], particle_hull[0].points[:,2], color='red',s=100)
#ax.scatter(particle_hull[1].points[:,0], particle_hull[1].points[:,1], particle_hull[1].points[:,2], color='red')
#ax.scatter(particle_hull_2[0].points[:,0], particle_hull_2[0].points[:,1], particle_hull_2[0].points[:,2], color='blue')
#ax.scatter(particle_hull_2[1].points[:,0], particle_hull_2[1].points[:,1], particle_hull_2[1].points[:,2], color='blue')
#ax.view_init(0, 45)
#ax.set_xlabel('X Label')
#ax.set_ylabel('Y Label')
#ax.set_zlabel('Z Label')
#ax.axis('equal')
#plt.show()


#print number of Pt atoms
print(len(filtered_Pt))
print(len(filtered_Pt_2))

#print z values of Pt atoms and unique z values of hull points



#find the points that are in filtered Pt but not in filtered Pt_2 and print their z coordinate
filtered_Pt_2_points = filtered_Pt_2[['x','y','z']].values
filtered_Pt_points = filtered_Pt[['x','y','z']].values
z_values = []
for point in filtered_Pt_points:
    if not np.any(np.all(point == filtered_Pt_2_points,axis=1)):
        z_values.append(point[2])
print(z_values)

ob_gen.mayavi_atomic_structure(filtered_atoms)


# %%# %%
from ase.cluster import wulff_construction

surfaces = [(1, 0, 0),(1,1,0), (1, 1, 1)]
esurf = [1.86, 1.68, 1.49]   # Surface energies.
lc = 3.9242
size = 200# Number of atoms
atoms = wulff_construction('Pt', surfaces, esurf,
                           size, 'fcc',
                           rounding='above', latticeconstant=lc)
#plot atoms with mayavi
#extract x,y,z coordinates from atoms
x_coords = atoms.positions[:,0]
y_coords = atoms.positions[:,1]
z_coords = atoms.positions[:,2]
particle = pd.DataFrame({'x': x_coords, 'y': y_coords, 'z': z_coords, 'label': ['Pt']*len(x_coords)})
#rotate points around axis
points = np.column_stack((x_coords,y_coords,z_coords))
points = ob_gen.rotate_points_around_axis(points, [-1,-1,0], 0.9553166181245093)
points = ob_gen.rotate_points_around_axis(points, [0,0,1], 0.9553166181245093)
particle = pd.DataFrame({'x': points[:,0], 'y': points[:,1], 'z': points[:,2], 'label': ['Pt']*len(x_coords)})

#if z closer to 0 than epsilon, set z to 0, if z < -epsilon, remove point
epsilon = 0.0001

particle.loc[abs(particle['z']) < epsilon, 'z'] = 0
particle = particle[particle['z'] >= 0]

particle.reset_index(drop=True, inplace=True)
particle['z'] = particle['z'].round(4)
ob_gen.mayavi_atomic_structure(particle)

print(len(particle))
#print number of atoms
#print unique z values
print(np.unique(particle.z))
#print number of atoms at each z
for z in np.unique(atoms.positions[:,2]):
    print(len(atoms.positions[atoms.positions[:,2]==z]))

# %%
ob_gen = ObjectGenerator()
bob = ob_gen.generate_wulff_particle(200,'Pt')
CeO2_bulk_test = ob_gen.Ceria_lattice(15,15,15)

points = bob[['x','y','z']].values
particle_hull  = ob_gen.hull_from_points(points)
particle_hull = ob_gen.expand_hull(particle_hull,0.000000001)

support_hull = ob_gen.support_hull(8,50,50,particle_interface_points=np.array([[5.69638606, 2.88445904, 0],[0.30238439, 6.99237546, 0],[-4.64750397, 4.79482633, 0],[-5.62285321, -1.09368618, 0],[-1.00215511, -4.9157151, 0],[5.76586847, -0.65974722, 0]]))
print(particle_hull[0].points,support_hull)
particle_new, support_new = ob_gen.add_step(particle_hull.copy(),support_hull.copy())
ob_gen.visualize_hull_points(particle_new)
ob_gen.visualize_hull_points(particle_hull)
filtered_Pt=ob_gen.filter_atoms_by_hull(bob,particle_new)
#relaxed_struct = ob_gen.relax_structure(filtered_Pt)
#ob_gen.mayavi_atomic_structure(relaxed_struct)
print("Filtered Pt",filtered_Pt)   
filtered_ceo2 = ob_gen.filter_atoms_by_hull(CeO2_bulk_test,support_new)
filtered_atoms = pd.concat([filtered_Pt,filtered_ceo2])
relaxed_struct = ob_gen.set_interface_spacing(filtered_atoms, 2.1)
relaxed_struct = ob_gen.relax_structure(relaxed_struct)
#ob_gen.mayavi_atomic_structure(relaxed_struct)
filtered_atoms.label = filtered_atoms.label.replace({'Ce': 0, 'O': 1, 'Pt': 2})
ob_gen.mayavi_atomic_structure(filtered_atoms)
# %%
relaxed_struct = ob_gen.remove_overlapping_atoms(relaxed_struct)
relaxed_struct.label = relaxed_struct.label.replace({'Ce': 0, 'O': 1, 'Pt': 2})
ob_gen.mayavi_atomic_structure(relaxed_struct)
ob_gen.generate_ase_cluster()