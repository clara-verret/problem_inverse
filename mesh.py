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
    
    def random_chord_length(self) :
        """
        Returns the length of a random chord of the polyhedron
        """
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
    
    def plot_proj(self, name) :
        """
        plots the projected points in MeshPlots/projections/name

        Args :
        name : name of the file
        """
        plt.figure()
        plt.plot(*zip(*self.projected_vertices), 'o')
        plt.savefig("MeshPlots/projections/"+name)

    def plot_3d(self, name) :
        """
        plots the 3D points in MeshPlots/3D/name

        Args :
        name : name of the file
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(4):
            for j in range(i+1,4):
                ax.plot(*zip(self.vertices[i],self.vertices[j]), color='b')
        ax.scatter(*zip(*self.vertices))
        plt.savefig("MeshPlots/3D/"+name)
    
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
        self.vertices = np.array([(-1/2,-self.b,0), (1/2,-self.b,0), (0,self.a,0), (0,0,self.H)])
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
        self.vertices = np.array([(x,y,z) for x in [-1/2,1/2] for y in [-1/2,1/2] for z in [-1/2,1/2]])
        self.projected_vertices = np.array([(x,y) for x,y,z in self.vertices])

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

def plot_cld_hist(chords, name, bins=100) :
    """
    plots the cld as an histogram and saves it in MeshPlots/clds/name

    Args :
    polyhedrons : list of polyhedrons
    """
    plt.figure()
    plt.hist(chords, bins=bins, density=True)
    plt.savefig("MeshPlots/clds/"+name)

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

chords = get_regular_tetrahedron_list_of_chords(N = 100000)
plot_cld_hist(chords, 'cld', bins=1000)
plot_cumulative_cld_hist(chords,'cumulative_cld', bins=1000)