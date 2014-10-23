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
        facet_index_counter = 0
        edge_dictionary = {} #store the coordinates of all Voronoi edges in the system, classified by the Voronoi vertex shared by all the edges
        for Delaunay_facet_neighbour_index_array in self.hull_instance.neighbors:
            voronoi_vertex_coordinate_corresponding_to_current_Delaunay_facet = array_Voronoi_vertices[facet_index_counter]
            list_Voronoi_edges_including_current_facet = []
            for index_of_neighbouring_facet in Delaunay_facet_neighbour_index_array:
                if numpy.allclose(voronoi_vertex_coordinate_corresponding_to_current_Delaunay_facet, array_Voronoi_vertices[index_of_neighbouring_facet],atol=1e-7): 
                    continue #avoid vertex-vertex 0-length edges
                current_Voronoi_edge_array = numpy.array([voronoi_vertex_coordinate_corresponding_to_current_Delaunay_facet, array_Voronoi_vertices[index_of_neighbouring_facet]])
                list_Voronoi_edges_including_current_facet.append(current_Voronoi_edge_array)
            edge_dictionary[facet_index_counter] = numpy.array(list_Voronoi_edges_including_current_facet)
            facet_index_counter += 1
        #so, edge_dictionary should looking something like this for a given sample entry: {0: [ [Voronoi_edge_coord_1, Voronoi_edge_coord_2], [Voronoi_edge_coord_1, Voronoi_edge_coord_3], [Voronoi_edge_coord_1,Voronoi_edge_coord_4]], ...}
        #also, I'm anticipating a lot of possible duplication of edges in edge_dictionary (i.e., edges shared with neighbours)
        #furthermore, these would only be straight-line edges, rather than the desired great-circle edges, although the vertices are of course the same in both cases [so, it's really just a plotting issue I think]
        #return edge_dictionary #temporary return for testing purposes
        
        #the Voronoi edges contained in edge_dictionary look reasonable by visual inspection, but we don't have the edges organized into polygons surrounding a single data point just yet

        #I want to calculate the midpoints of all Voronoi edges to simplify downstream data structuring (the vertices can be equidistant to multiple generators and are more confusing to work with here I think)
        edge_midpoint_dictionary = {} 
        for Voronoi_vertex_number, array_Voronoi_edges_associated_with_this_vertex in edge_dictionary.iteritems():
            array_edge_midpoints_associated_with_this_vertex = numpy.average(array_Voronoi_edges_associated_with_this_vertex,axis=1)
            edge_midpoint_dictionary[Voronoi_vertex_number] = array_edge_midpoints_associated_with_this_vertex
        
        #find and store the indices of the generator (original data) points that are closest to each Voronoi edge midpoint (some will have multiple closest points)
        dictionary_closest_generator_indices_per_edge_midpoint = {}
        for Voronoi_vertex_number, array_edge_midpoints_associated_with_this_vertex in edge_midpoint_dictionary.iteritems():
            distance_matrix_Voronoi_edge_midpoints_to_original_data_points = scipy.spatial.distance.cdist(array_edge_midpoints_associated_with_this_vertex,self.original_point_array)
            #print 'Voronoi_vertex_number:', Voronoi_vertex_number
            #print 'argmin of distance_matrix_Voronoi_edge_midpoints_to_original_data_points:', numpy.argmin(distance_matrix_Voronoi_edge_midpoints_to_original_data_points,axis=1)
            array_indices_for_generators_closest_to_edge_midpoints = numpy.argmin(distance_matrix_Voronoi_edge_midpoints_to_original_data_points,axis=1) #each row (midpoint) should have a set of indices for the closest original data points
            dictionary_closest_generator_indices_per_edge_midpoint[Voronoi_vertex_number] = array_indices_for_generators_closest_to_edge_midpoints
        #a representative element of the above dictionary looks like this: {0: array([104,114]),...}

        #now I want to reorganize the data structure such that I capture the Voronoi vertices that surround each generator (original data) point
        #for any given generator, this should include all Voronoi edge midpoints that include that generator amongst their 'closest'
        dictionary_generator_Voronoi_polygons = {}
        for generator_index, generator_coordinate in enumerate(self.original_point_array):
            #print 'generator_index:', generator_index
            list_voronoi_vertices_current_generator = []
            for Voronoi_vertex_number, array_indices_for_generators_closest_to_edge_midpoints in dictionary_closest_generator_indices_per_edge_midpoint.iteritems():
                if generator_index in array_indices_for_generators_closest_to_edge_midpoints:
                    indices_of_edges_bounding_current_generator = numpy.where(array_indices_for_generators_closest_to_edge_midpoints == generator_index)
                    #print 'indices_of_edges_bounding_current_generator:', indices_of_edges_bounding_current_generator
                    vertices_of_edges_bounding_current_generator = edge_dictionary[Voronoi_vertex_number][indices_of_edges_bounding_current_generator]
                    for vertex in vertices_of_edges_bounding_current_generator:
                        list_voronoi_vertices_current_generator.extend(vertex)
            #try using pandas to remove duplicate rows (vertices) before storing in dictionary
            array_Voronoi_vertices = numpy.around(numpy.array(list_voronoi_vertices_current_generator),decimals=8) #rounding decimals as pre-processing for duplicate removal
            df = pandas.DataFrame(array_Voronoi_vertices)
            #print 'df shape before:', df.shape
            df.drop_duplicates(inplace=True)
            #print 'df shape after deduplication:', df.shape
            array_Voronoi_vertices = df.values #convert back to numpy array after dropping duplicates

            #now, I want to sort the polygon vertices in a consistent, non-intersecting fashion
            if array_Voronoi_vertices.shape[0] > 3:
                polygon_hull_object = scipy.spatial.ConvexHull(array_Voronoi_vertices[...,:2]) #trying to project to 2D for edge ordering, and then restore to 3D after
                point_indices_ordered_vertex_array = polygon_hull_object.vertices
                array_Voronoi_vertices = array_Voronoi_vertices[point_indices_ordered_vertex_array]

            dictionary_generator_Voronoi_polygons[generator_index] = {'generator_coordinate':generator_coordinate,'voronoi_polygon_vertices':array_Voronoi_vertices}


        return (edge_dictionary,edge_midpoint_dictionary,dictionary_closest_generator_indices_per_edge_midpoint,dictionary_generator_Voronoi_polygons) #temporary test return





        


