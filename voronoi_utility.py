'''Author: Tyler Reddy

The purpose of this Python module is to provide utility code for handling spherical Voronoi Diagrams.'''

import scipy
import scipy.spatial
import numpy

class Voronoi_Sphere_Surface:
    '''Voronoi diagrams on the surface of a sphere.
    
    Parameters
    ----------
    points: array, shape (npoints, 3)
        Coordinates of points to construct a Voronoi diagram on the surface of a sphere
    
    References
    ----------
    
    .. [Caroli] Caroli et al. (2009) INRIA 7004
    
    '''

    def __init__(self,points):
        self.original_point_array = points

    def Delaunay_triangulation_spherical_surface(self):
        '''Delaunay tessellation of the points on the surface of the sphere. This is simply the 3D convex hull of the points. Returns a shape (N,3,3) array of points representing the vertices of the Delaunay triangulation on the sphere (i.e., N three-dimensional triangle vertex arrays).'''
        hull_instance = scipy.spatial.ConvexHull(self.original_point_array)
        list_points_vertices_Delaunay_triangulation = [] 
        for simplex in hull_instance.simplices: #for each simplex (face; presumably a triangle) of the convex hull
            convex_hull_triangular_facet_vertex_coordinates = self.original_point_array[simplex]
            assert convex_hull_triangular_facet_vertex_coordinates.shape == (3,3), "Triangular facet of convex hull should be a triangle in 3D space specified by coordinates in a shape (3,3) numpy array."
            list_points_vertices_Delaunay_triangulation.append(convex_hull_triangular_facet_vertex_coordinates)
        array_points_vertices_Delaunay_triangulation = numpy.array(list_points_vertices_Delaunay_triangulation)
        return array_points_vertices_Delaunay_triangulation




        


