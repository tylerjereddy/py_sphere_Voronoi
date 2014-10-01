'''Author: Tyler Reddy

The purpose of this Python module is to provide utility code for handling spherical Voronoi Diagrams.'''

import scipy
import scipy.spatial
import numpy

class Voronoi_Sphere_Surface(points):
    '''Voronoi diagrams on the surface of a sphere.
    
    Parameters
    ----------
    points: array
        Coordinates of points to construct a Voronoi diagram on the surface of a sphere
    
    
    
    '''

    def __init__(self,points):
        self.original_point_array = points

    def Delaunay_triangulation_spherical_surface(self):
        '''Delaunay tesselation of the points on the surface of the sphere. This is simply the 3D convex hull of the points.'''
        hull_instance = scipy.spatial.ConvexHull(self.original_point_array)
        array_points_vertices_Convex_Hull = self.original_point_array[hull_instance.vertices]




        


