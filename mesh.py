import numpy as np
import random
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

class polyhedron:
    """
    An element of this class represents a polyhedron.

    Args :
    vertices : list of the vertices (triplets) of the polyhedron
    """
    def __init__(self, vertices):
        self.vertices = np.array(vertices)
        self.projected_vertices = np.array([(x,y) for x,y,z in self.vertices]) #projection of the vertices on the plane z=0

    def __str__(self):
        return "Polyhedron with vertices : " + str(self.vertices)
    
    def project(self) :
        """
        Project the polyhedron on the plane z=0
        """
        self.projected_vertices = [(x,y) for x,y,z in self.vertices]
    
    def rotate(self, Gamma):
        """
        Rotate the polyhedron by the rotation matrix Gamma
        """
        self.vertices = np.dot(Gamma, np.array(self.vertices).T).T
        self.project()

    def y_bounds(self):
        """
        Returns the bounds of the projection in the y direction
        """
        return (min(y for x,y in self.projected_vertices), max(y for x,y in self.projected_vertices))
    
    def random_y(self):
        """
        Returns a random chord of the polyhedron
        """
        y_min, y_max = self.y_bounds()
        y = random.uniform(y_min, y_max)
        return y
    
    def random_chord_length(self, y = None): 
        """
        Returns the length of a random chord of the polyhedron
        """
        if y is None:
            y = self.random_y()
        
        over_y = []
        under_y = []
        for point_2d in self.projected_vertices:
            if point_2d[1] > y:
                over_y.append(point_2d)
            else:
                under_y.append(point_2d)

        #for every pair of points with 1 over y and 1 under, we compute the intersection point
        intersections_x = []
        for point_over in over_y:
            for point_under in under_y:
                x_over, y_over = point_over
                x_under, y_under = point_under
                x_inter = x_under + (y-y_under)*(x_over-x_under)/(y_over-y_under)
                intersections_x.append(x_inter)

        return max(intersections_x) - min(intersections_x)
    
    def plot_proj(self, name, chord_y = None) :
        """
        plots the projected points in MeshPlots/projections/name

        Args :
        name : name of the file
        chord_y : y coordinate of the chord to plot. If None, no chord is plotted
        """
        plt.figure()
        plt.plot(*zip(*self.projected_vertices), 'o')
        n = len(self.projected_vertices)
        for i in range(n):
            for j in range(i+1,n):
                plt.plot(*zip(self.projected_vertices[i],self.projected_vertices[j]), color='b')
        if chord_y is not None :
            plt.axhline(y=chord_y, color='r')
        plt.savefig("MeshPlots/projections/"+name)
        plt.close()

    def plot_3d(self, name) :
        """
        plots the 3D points in MeshPlots/3D/name

        Args :
        name : name of the file
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        n = len(self.vertices)
        for i in range(n):
            for j in range(i+1,n):
                ax.plot(*zip(self.vertices[i],self.vertices[j]), color='b')
        ax.scatter(*zip(*self.vertices))
        plt.savefig("MeshPlots/3D/"+name)
        plt.close()
    
class regular_tetrahedron(polyhedron) :
    """
    An element of this class represents a regular tetrahedron, using the polyhedron class.

    Args :
    r : length of the edges of the tetrahedron
    """
    def __init__(self, r = 1):
        self.r = r
        self.h = r*np.sqrt(3)/2 #height of one of the triangular faces
        self.H = r*np.sqrt(6)/3 #height of the tetrahedron
        self.b = r/(2*np.sqrt(3)) #distance from the center to the center of a face
        self.a = r/np.sqrt(3) #distance from the center to a vertex
        self.vertices = np.array([(-self.r/2,-self.b,0), (self.r/2,-self.b,0), (0,self.a,0), (0,0,self.H)])
        self.projected_vertices = np.array(self.vertices[:,:2])

    def check_edges_length(self):
        """
        Prints the edges length of the tetrahedron
        """
        for i in range(4):
            for j in range(i+1,4):
                print("Edge",i,j,"length :",np.linalg.norm(np.array(self.vertices[i])-np.array(self.vertices[j])))

class cube(polyhedron) :
    """
    An element of this class represents a cube, using the polyhedron class.

    Args :
    r : length of the edges of the cube
    """
    def __init__(self, r = 1):
        self.r = r
        self.vertices = r*np.array([(x,y,z) for x in [-1/2,1/2] for y in [-1/2,1/2] for z in [-1/2,1/2]])
        self.projected_vertices = np.array([(x,y) for x,y,z in self.vertices])

class regular_pyramid(polyhedron) :
    """
    An element of this class represents a regular pyramid (base = square, summit above the center of the square, all edges of the same length), using the polyhedron class.

    Args :
    r : length of the edges of the pyramid
    """
    def __init__(self, r = 1):
        self.r = r
        self.h = r*np.sqrt(2)/2 #height of the pyramid
        self.normalized_h = np.sqrt(2)/2 #height of the pyramid, with r = 1
        self.vertices = r*np.array([(-1/2,-1/2,0), (1/2,-1/2,0), (1/2,1/2,0), (-1/2,1/2,0), (0,0,self.normalized_h)])
        self.projected_vertices = np.array(self.vertices[:,:2])

class regular_triangular_prism(polyhedron) :
    """
    An element of this class represents a regular triangular prism (equilateral triangle base, all edges of the same length) (basically, a tent), using the polyhedron class.

    Args :
    r : length of the edges of the prism
    """
    def __init__(self, r = 1):
        self.r = r
        self.h = r*np.sqrt(3)/2 #height of the prism
        self.normalized_h = np.sqrt(3)/2 #height of the prism, with r = 1
        self.vertices = r*np.array([(-1/2,-1/2,0), (1/2,-1/2,0), (1/2,1/2,0), (-1/2,1/2,0), (-1/2,0,self.normalized_h), (1/2,0,self.normalized_h)])
        self.projected_vertices = np.array(self.vertices[:,:2])


def random_uniform_rotation_matrix() :
    return Rotation.random().as_matrix()

def plot_cumulative_cld_hist(chords, name, bins=100) :
    """
    plots the cumulative cld as an histogram and saves it in MeshPlots/clds/name

    Args :
    polyhedrons : list of polyhedrons
    """
    chords.sort()
    plt.figure()
    plt.hist(chords, bins=bins, cumulative=True, density=True, histtype='step')
    plt.savefig("MeshPlots/clds/"+name)
    plt.close()

def plot_cld_hist(chords, name, bins=100) :
    """
    plots the cld as an histogram and saves it in MeshPlots/clds/name

    Args :
    polyhedrons : list of polyhedrons
    """
    plt.figure()
    plt.hist(chords, bins=bins, density=True)
    plt.savefig("MeshPlots/clds/"+name)
    plt.close()

def get_regular_tetrahedron_list_of_chords(r = 1, N = 10000) :
    """
    Returns the list of N chord lengths of a regular tetrahedron with edges of length r

    Args :
    r : length of the edges of the tetrahedron
    N : number of chords
    """
    tetrahedron = regular_tetrahedron(r)
    list_of_chords = []
    for i in range(N) :
        Gamma = random_uniform_rotation_matrix()
        tetrahedron.rotate(Gamma)
        list_of_chords.append(tetrahedron.random_chord_length())
    return list_of_chords

def get_cube_list_of_chords(r = 1, N = 10000) :
    """
    Returns the list of N chord lengths of a cube with edges of length r

    Args :
    r : length of the edges of the cube
    N : number of chords
    """
    cube_ = cube(r)
    list_of_chords = []
    for i in range(N) :
        Gamma = random_uniform_rotation_matrix()
        cube_.rotate(Gamma)
        list_of_chords.append(cube_.random_chord_length())
    return list_of_chords

def get_regular_pyramid_list_of_chords(r = 1, N = 10000) :
    """
    Returns the list of N chord lengths of a regular pyramid with edges of length r

    Args :
    r : length of the edges of the pyramid
    N : number of chords
    """
    pyramid = regular_pyramid(r)
    list_of_chords = []
    for i in range(N) :
        Gamma = random_uniform_rotation_matrix()
        pyramid.rotate(Gamma)
        list_of_chords.append(pyramid.random_chord_length())
    return list_of_chords

def get_regular_triangular_prism_list_of_chords(r = 1, N = 10000) :
    """
    Returns the list of N chord lengths of a regular triangular prism with edges of length r

    Args :
    r : length of the edges of the prism
    N : number of chords
    """
    prism = regular_triangular_prism(r)
    list_of_chords = []
    for i in range(N) :
        Gamma = random_uniform_rotation_matrix()
        prism.rotate(Gamma)
        list_of_chords.append(prism.random_chord_length())
    return list_of_chords

def get_tetrahedron_cld(N = 524288, bins=4096) :
    chords = get_regular_tetrahedron_list_of_chords(N = N)
    plot_cld_hist(chords, 'cld_tetra', bins=bins)
    plot_cumulative_cld_hist(chords,'cumulative_cld_tetra', bins=bins)

def get_cube_cld(N = 524288, bins=4096) :
    chords = get_cube_list_of_chords(N = N)
    plot_cld_hist(chords, 'cld_cube', bins=bins)
    plot_cumulative_cld_hist(chords,'cumulative_cld_cube', bins=bins)

def get_pyramid_cld(N = 524288, bins=4096) :
    chords = get_regular_pyramid_list_of_chords(N = N)
    plot_cld_hist(chords, 'cld_pyramid', bins=bins)
    plot_cumulative_cld_hist(chords,'cumulative_cld_pyramid', bins=bins)

def get_prism_cld(N = 524288, bins=4096) :
    chords = get_regular_triangular_prism_list_of_chords(N = N)
    plot_cld_hist(chords, 'cld_prism', bins=bins)
    plot_cumulative_cld_hist(chords,'cumulative_cld_prism', bins=bins)

def get_all_cld(N = 524288, bins=4096) :
    get_tetrahedron_cld(N = N, bins=bins)
    get_cube_cld(N = N, bins=bins)
    get_pyramid_cld(N = N, bins=bins)
    get_prism_cld(N = N, bins=bins)

def plot_sample_tetrahedrons() :
    zone1 = []
    zone2 = []
    for i in range(100) :
        tetrahedron = regular_tetrahedron()
        Gamma = random_uniform_rotation_matrix()
        tetrahedron.rotate(Gamma)
        y = tetrahedron.random_y()
        chord_length = tetrahedron.random_chord_length(y)
        if chord_length < 0.5 :
            zone1.append((tetrahedron, y))
        if chord_length > 0.8 :
            zone2.append((tetrahedron, y))
    
    for i,(tetra,y) in enumerate(zone1) :
        tetra.plot_proj('tetra_zone1_'+str(i), y)
        tetra.plot_3d('tetra_zone1_'+str(i))

    for i,(tetra,y) in enumerate(zone2) :
        tetra.plot_proj('tetra_zone2_'+str(i), y)
        tetra.plot_3d('tetra_zone2_'+str(i))

def plot_sample_cubes() :
    zone1 = []
    zone2 = []
    zone3 = []
    for i in range(300) :
        cube_ = cube()
        Gamma = random_uniform_rotation_matrix()
        cube_.rotate(Gamma)
        y = cube_.random_y()
        chord_length = cube_.random_chord_length(y)
        if chord_length < 0.8 :
            zone1.append((cube_, y))
        if (chord_length > 1.1) and (chord_length < 1.3) :
            zone2.append((cube_, y))
        if chord_length > 1.5 :
            zone3.append((cube_, y))
    
    for i,(cube_,y) in enumerate(zone1) :
        cube_.plot_proj('cube_zone1_'+str(i), y)
        cube_.plot_3d('cube_zone1_'+str(i))

    for i,(cube_,y) in enumerate(zone2) :
        cube_.plot_proj('cube_zone2_'+str(i), y)
        cube_.plot_3d('cube_zone2_'+str(i))

    for i,(cube_,y) in enumerate(zone3) :
        cube_.plot_proj('cube_zone3_'+str(i), y)
        cube_.plot_3d('cube_zone3_'+str(i))

get_all_cld(N = 1048576, bins=8192)