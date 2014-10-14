'''Author: Tyler Reddy

The purpose of this Python module is to provide utility code for handling spherical Voronoi Diagrams.'''

import scipy
import scipy.spatial
import numpy
import numpy.linalg

def produce_triangle_vertex_coordinate_array_Delaunay_sphere(input_point_array):
    '''Return shape (N,3,3) numpy array of the Delaunay triangle vertex coordinates on the surface of the sphere.'''
    hull_instance = scipy.spatial.ConvexHull(input_point_array)
    list_points_vertices_Delaunay_triangulation = [] 
    for simplex in hull_instance.simplices: #for each simplex (face; presumably a triangle) of the convex hull
        convex_hull_triangular_facet_vertex_coordinates = input_point_array[simplex]
        assert convex_hull_triangular_facet_vertex_coordinates.shape == (3,3), "Triangular facet of convex hull should be a triangle in 3D space specified by coordinates in a shape (3,3) numpy array."
        list_points_vertices_Delaunay_triangulation.append(convex_hull_triangular_facet_vertex_coordinates)
    array_points_vertices_Delaunay_triangulation = numpy.array(list_points_vertices_Delaunay_triangulation)
    return array_points_vertices_Delaunay_triangulation

def produce_array_Voronoi_vertices_on_sphere_surface(facet_coordinate_array_Delaunay_triangulation,sphere_radius):
    '''Return shape (N,3) array of coordinates for the vertices of the Voronoi diagram on the sphere surface given a shape (N,3,3) array of Delaunay triangulation vertices.'''
    assert facet_coordinate_array_Delaunay_triangulation.shape[1:] == (3,3), "facet_coordinate_array_Delaunay_triangulation should have shape (N,3,3)."
    #calculate the surface normal of each triangle as the vector cross product of two edges
    list_triangle_facet_normals = []
    for triangle_coord_array in facet_coordinate_array_Delaunay_triangulation:
        facet_normal = numpy.cross(triangle_coord_array[1] - triangle_coord_array[0],triangle_coord_array[2] - triangle_coord_array[0]) 
        facet_normal_magnitude = numpy.linalg.norm(facet_normal) #length of surface normal
        facet_normal_unit_vector = facet_normal / facet_normal_magnitude #vector of length 1 in same direction
        list_triangle_facet_normals.append(facet_normal_unit_vector)

    array_facet_normals = numpy.array(list_triangle_facet_normals) * sphere_radius #adjust for radius of sphere
    array_Voronoi_vertices = array_facet_normals
    assert array_Voronoi_vertices.shape[1] == 3, "The array of Voronoi vertices on the sphere should have shape (N,3)."
    return array_Voronoi_vertices


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
        self.sphere_centroid = numpy.average(self.original_point_array,axis=0)
        self.estimated_sphere_radius = numpy.average(scipy.spatial.distance.cdist(self.original_point_array,self.sphere_centroid[numpy.newaxis,:]))

    def Delaunay_triangulation_spherical_surface(self):
        '''Delaunay tessellation of the points on the surface of the sphere. This is simply the 3D convex hull of the points. Returns a shape (N,3,3) array of points representing the vertices of the Delaunay triangulation on the sphere (i.e., N three-dimensional triangle vertex arrays).'''
        array_points_vertices_Delaunay_triangulation = produce_triangle_vertex_coordinate_array_Delaunay_sphere(self.original_point_array)
        return array_points_vertices_Delaunay_triangulation

    def Voronoi_vertices_spherical_surface(self):
        '''Determine the Voronoi vertices on the surface of the sphere given the vertices of the Delaunay triangulation on the surface of the sphere. Inspired by response here: http://stackoverflow.com/a/22234783'''
        facet_coordinate_array_Delaunay_triangulation = produce_triangle_vertex_coordinate_array_Delaunay_sphere(self.original_point_array)
        array_Voronoi_vertices = produce_array_Voronoi_vertices_on_sphere_surface(facet_coordinate_array_Delaunay_triangulation,self.sphere_radius)
        return array_Voronoi_vertices

    def Voronoi_polygons_spherical_surface(self):
        '''Compute useful dictionary data structure relating to the polygons in the spherical Voronoi diagram and the original data points that they contain.'''
        pass





        


