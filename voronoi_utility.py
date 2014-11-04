'''Author: Tyler Reddy

The purpose of this Python module is to provide utility code for handling spherical Voronoi Diagrams.'''

import scipy
import scipy.spatial
import numpy
import numpy.linalg
import pandas

def convert_cartesian_array_to_spherical_array(coord_array,angle_measure='radians'):
    '''Take shape (N,3) cartesian coord_array and return an array of the same shape in spherical polar form (r, theta, phi). Based on StackOverflow response: http://stackoverflow.com/a/4116899
    use radians for the angles by default, degrees if angle_measure == 'degrees' '''
    spherical_coord_array = numpy.zeros(coord_array.shape)
    xy = coord_array[...,0]**2 + coord_array[...,1]**2
    spherical_coord_array[...,0] = numpy.sqrt(xy + coord_array[...,2]**2)
    spherical_coord_array[...,1] = numpy.arctan2(numpy.sqrt(xy), coord_array[...,2])
    spherical_coord_array[...,2] = numpy.arctan2(coord_array[...,1],coord_array[...,0])
    if angle_measure == 'degrees':
        spherical_coord_array[...,1] = numpy.degrees(spherical_coord_array[...,1])
        spherical_coord_array[...,2] = numpy.degrees(spherical_coord_array[...,2])
    return spherical_coord_array

def convert_spherical_array_to_cartesian_array(spherical_coord_array,angle_measure='radians'):
    '''Take shape (N,3) spherical_coord_array (r,theta,phi) and return an array of the same shape in cartesian coordinate form (x,y,z). Based on the equations provided at: http://en.wikipedia.org/wiki/List_of_common_coordinate_transformations#From_spherical_coordinates
    use radians for the angles by default, degrees if angle_measure == 'degrees' '''
    cartesian_coord_array = numpy.zeros(spherical_coord_array.shape)
    #convert to radians if degrees are used in input (prior to Cartesian conversion process)
    if angle_measure == 'degrees':
        spherical_coord_array[...,1] = numpy.deg2rad(spherical_coord_array[...,1])
        spherical_coord_array[...,2] = numpy.deg2rad(spherical_coord_array[...,2])
    #now the conversion to Cartesian coords
    cartesian_coord_array[...,0] = spherical_coord_array[...,0] * numpy.sin(spherical_coord_array[...,1]) * numpy.cos(spherical_coord_array[...,2])
    cartesian_coord_array[...,1] = spherical_coord_array[...,0] * numpy.sin(spherical_coord_array[...,1]) * numpy.sin(spherical_coord_array[...,2])
    cartesian_coord_array[...,2] = spherical_coord_array[...,0] * numpy.cos(spherical_coord_array[...,1])
    return cartesian_coord_array

def produce_triangle_vertex_coordinate_array_Delaunay_sphere(hull_instance):
    '''Return shape (N,3,3) numpy array of the Delaunay triangle vertex coordinates on the surface of the sphere.'''
    list_points_vertices_Delaunay_triangulation = [] 
    for simplex in hull_instance.simplices: #for each simplex (face; presumably a triangle) of the convex hull
        convex_hull_triangular_facet_vertex_coordinates = hull_instance.points[simplex]
        assert convex_hull_triangular_facet_vertex_coordinates.shape == (3,3), "Triangular facet of convex hull should be a triangle in 3D space specified by coordinates in a shape (3,3) numpy array."
        list_points_vertices_Delaunay_triangulation.append(convex_hull_triangular_facet_vertex_coordinates)
    array_points_vertices_Delaunay_triangulation = numpy.array(list_points_vertices_Delaunay_triangulation)
    return array_points_vertices_Delaunay_triangulation

def produce_array_Voronoi_vertices_on_sphere_surface(facet_coordinate_array_Delaunay_triangulation,sphere_radius,sphere_centroid):
    '''Return shape (N,3) array of coordinates for the vertices of the Voronoi diagram on the sphere surface given a shape (N,3,3) array of Delaunay triangulation vertices.'''
    assert facet_coordinate_array_Delaunay_triangulation.shape[1:] == (3,3), "facet_coordinate_array_Delaunay_triangulation should have shape (N,3,3)."
    #calculate the surface normal of each triangle as the vector cross product of two edges
    list_triangle_facet_normals = []
    for triangle_coord_array in facet_coordinate_array_Delaunay_triangulation:
        facet_normal = numpy.cross(triangle_coord_array[1] - triangle_coord_array[0],triangle_coord_array[2] - triangle_coord_array[0]) 
        facet_normal_magnitude = numpy.linalg.norm(facet_normal) #length of surface normal
        facet_normal_unit_vector = facet_normal / facet_normal_magnitude #vector of length 1 in same direction

        #try to ensure that facet normal faces the correct direction (i.e., out of sphere)
        triangle_centroid = numpy.average(triangle_coord_array,axis=0)
        #the Euclidean distance between the triangle centroid and the facet normal should be smaller than the sphere centroid to facet normal distance, otherwise, need to invert the vector
        triangle_to_normal_distance = scipy.spatial.distance.euclidean(triangle_centroid,facet_normal_unit_vector)
        sphere_centroid_to_normal_distance = scipy.spatial.distance.euclidean(sphere_centroid,facet_normal_unit_vector)
        delta_value = sphere_centroid_to_normal_distance - triangle_to_normal_distance
        if delta_value < -0.1: #need to rotate the vector so that it faces out of the circle
            #print 'delta_value:', delta_value
            facet_normal_unit_vector *= -1 #I seem to get a fair number of degenerate / duplicate Voronoi vertices (is this ok?! will have to filter them out I think ?!)
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
        self.hull_instance = scipy.spatial.ConvexHull(self.original_point_array)

    def Delaunay_triangulation_spherical_surface(self):
        '''Delaunay tessellation of the points on the surface of the sphere. This is simply the 3D convex hull of the points. Returns a shape (N,3,3) array of points representing the vertices of the Delaunay triangulation on the sphere (i.e., N three-dimensional triangle vertex arrays).'''
        array_points_vertices_Delaunay_triangulation = produce_triangle_vertex_coordinate_array_Delaunay_sphere(self.hull_instance)
        return array_points_vertices_Delaunay_triangulation

    def Voronoi_vertices_spherical_surface(self):
        '''Determine the Voronoi vertices on the surface of the sphere given the vertices of the Delaunay triangulation on the surface of the sphere. Inspired by response here: http://stackoverflow.com/a/22234783'''
        facet_coordinate_array_Delaunay_triangulation = produce_triangle_vertex_coordinate_array_Delaunay_sphere(self.hull_instance)
        array_Voronoi_vertices = produce_array_Voronoi_vertices_on_sphere_surface(facet_coordinate_array_Delaunay_triangulation,self.estimated_sphere_radius,self.sphere_centroid)
        assert facet_coordinate_array_Delaunay_triangulation.shape[0] == array_Voronoi_vertices.shape[0], "The number of Delaunay triangles should match the number of Voronoi vertices."
        return array_Voronoi_vertices

    def Voronoi_polygons_spherical_surface(self):
        '''Compute useful dictionary data structure relating to the polygons in the spherical Voronoi diagram and the original data points that they contain.'''
        #generate the array of Voronoi vertices:
        facet_coordinate_array_Delaunay_triangulation = produce_triangle_vertex_coordinate_array_Delaunay_sphere(self.hull_instance)
        array_Voronoi_vertices = produce_array_Voronoi_vertices_on_sphere_surface(facet_coordinate_array_Delaunay_triangulation,self.estimated_sphere_radius,self.sphere_centroid)
        assert facet_coordinate_array_Delaunay_triangulation.shape[0] == array_Voronoi_vertices.shape[0], "The number of Delaunay triangles should match the number of Voronoi vertices."
        #now, the tricky part--building up a useful Voronoi polygon data structure

        #new strategy--I already have the Voronoi vertices and the generators, so work based off a distance matrix between them
        distance_matrix_Voronoi_vertices_to_generators = scipy.spatial.distance.cdist(array_Voronoi_vertices,self.original_point_array)
        #now, each row of the above distance array corresponds to a single Voronoi vertex, which each column of that row representing the distance to the respective generator point
        #if we iterate through each of the rows and determine the indices of the minimum distances, we obtain the indices of the generators for which that voronoi vertex is a polygon vertex 
        generator_Voronoi_region_dictionary = {} #store the indices of the generators for which a given Voronoi vertex is also a polygon vertex
        for Voronoi_point_index, Voronoi_point_distance_array in enumerate(distance_matrix_Voronoi_vertices_to_generators):
            Voronoi_point_distance_array = numpy.around(Voronoi_point_distance_array,decimals=10)
            indices_of_generators_for_which_this_Voronoi_point_is_a_polygon_vertex = numpy.where(Voronoi_point_distance_array == Voronoi_point_distance_array.min())[0]
            assert indices_of_generators_for_which_this_Voronoi_point_is_a_polygon_vertex.size >= 3, "By definition, a Voronoi vertex must be equidistant to at least 3 generators."
            generator_Voronoi_region_dictionary[Voronoi_point_index] = indices_of_generators_for_which_this_Voronoi_point_is_a_polygon_vertex #so dictionary looks like 0: array(12,17,27), ...

        #now, go through the above dictionary and collect the Voronoi point indices forming the polygon for each generator index
        dictionary_Voronoi_point_indices_for_each_generator = {}
        for Voronoi_point_index, indices_of_generators_for_which_this_Voronoi_point_is_a_polygon_vertex in generator_Voronoi_region_dictionary.iteritems():
            for generator_index in indices_of_generators_for_which_this_Voronoi_point_is_a_polygon_vertex:
                if generator_index in dictionary_Voronoi_point_indices_for_each_generator:
                    list_Voronoi_indices = dictionary_Voronoi_point_indices_for_each_generator[generator_index] 
                    list_Voronoi_indices.append(Voronoi_point_index)
                    dictionary_Voronoi_point_indices_for_each_generator[generator_index] = list_Voronoi_indices
                else: #initialize the list of Voronoi indices for that generator key
                    dictionary_Voronoi_point_indices_for_each_generator[generator_index] = [Voronoi_point_index]
        #so this dictionary should have format: {generator_index: [list_of_Voronoi_indices_forming_polygon_vertices]}

        #now, I want to sort the polygon vertices in a consistent, non-intersecting fashion
        dictionary_sorted_Voronoi_point_coordinates_for_each_generator = {}
        for generator_index, list_unsorted_Voronoi_region_vertices in dictionary_Voronoi_point_indices_for_each_generator.iteritems():
            current_array_Voronoi_vertices = array_Voronoi_vertices[list_unsorted_Voronoi_region_vertices]
            if current_array_Voronoi_vertices.shape[0] > 3:
                polygon_hull_object = scipy.spatial.ConvexHull(current_array_Voronoi_vertices[...,:2]) #trying to project to 2D for edge ordering, and then restore to 3D after
                point_indices_ordered_vertex_array = polygon_hull_object.vertices
                current_array_Voronoi_vertices = current_array_Voronoi_vertices[point_indices_ordered_vertex_array]
            assert current_array_Voronoi_vertices.shape[0] >= 3, "All generators should be within Voronoi regions (polygons with at least 3 vertices)."
            dictionary_sorted_Voronoi_point_coordinates_for_each_generator[generator_index] = current_array_Voronoi_vertices

        return (generator_Voronoi_region_dictionary, dictionary_sorted_Voronoi_point_coordinates_for_each_generator)

        


