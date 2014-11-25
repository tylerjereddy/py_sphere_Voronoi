'''Author: Tyler Reddy

The purpose of this Python module is to provide utility code for handling spherical Voronoi Diagrams.'''

import scipy
if int(scipy.__version__.split('.')[1]) < 13:
    raise ImportError('Module requires version of scipy module >= 0.13.0')
import scipy.spatial
import numpy
import numpy.linalg
import pandas
import math
import numpy.random

def generate_random_array_spherical_generators(num_generators,sphere_radius,prng_object):
    '''Generate and return an array of shape (num_generators,3) random points on a sphere of given radius.
    Based on: Muller, M. E. "A Note on a Method for Generating Points Uniformly on N-Dimensional Spheres." Comm. Assoc. Comput. Mach. 2, 19-20, Apr. 1959.
    And see: http://mathworld.wolfram.com/SpherePointPicking.html
    prng_object is a pseudo random number generator object from numpy used for pinning down the random seed (so that pathological random states can be avoided as needed).
    see: http://stackoverflow.com/a/5837352/2942522'''
    #generate Gaussian random variables:
    array_random_Gaussian_data = prng_object.normal(loc=0.0,scale=1.0,size=(num_generators,3)) 
    #I'm not sure if this is the proper approach, but try projecting to the sphere_radius (may want to check above literature more closely):
    spherical_polar_data = convert_cartesian_array_to_spherical_array(array_random_Gaussian_data)
    spherical_polar_data[...,0] = sphere_radius
    cartesian_random_points = convert_spherical_array_to_cartesian_array(spherical_polar_data)
    #filter out any duplicate generators:
    df_random_points = pandas.DataFrame(cartesian_random_points)
    df_random_points_no_duplicates = df_random_points.drop_duplicates()
    array_random_spherical_generators = df_random_points_no_duplicates.as_matrix()
    return array_random_spherical_generators




def estimate_surface_area_spherical_polygon_JPL(array_ordered_Voronoi_polygon_vertices,sphere_radius):
    '''Estimate the area of a spherical polygon using the method proposed in a JPL documnet: http://trs-new.jpl.nasa.gov/dspace/bitstream/2014/41271/1/07-0286.pdf
    My attempts at implementing the exact solution for spherical polygon surface area have been very problematic so I'm trying this instead.
    I think this may only apply to polygons that do not contain either pole of the sphere.'''
    current_vertex_index = 0
    surface_area_spherical_polygon = 0
    num_vertices_in_Voronoi_polygon = array_ordered_Voronoi_polygon_vertices.shape[0] #the number of rows == number of vertices in polygon
    while current_vertex_index < num_vertices_in_Voronoi_polygon:
        #print '-------------'
        #print 'current_vertex_index:', current_vertex_index
        if current_vertex_index == 0:
            previous_vertex_index = num_vertices_in_Voronoi_polygon - 1
        else:
            previous_vertex_index = current_vertex_index - 1
        if current_vertex_index == num_vertices_in_Voronoi_polygon - 1:
            next_vertex_index = 0
        else:
            next_vertex_index = current_vertex_index + 1

        print 'previous_vertex_index:', previous_vertex_index
        print 'current_vertex_index:', current_vertex_index
        print 'next_vertex_index:', next_vertex_index

        previous_vertex_coordinate = array_ordered_Voronoi_polygon_vertices[previous_vertex_index]
        current_vertex_coordinate = array_ordered_Voronoi_polygon_vertices[current_vertex_index]
        next_vertex_coordinate = array_ordered_Voronoi_polygon_vertices[next_vertex_index]

        previous_vertex_spherical_polar_coordinates = convert_cartesian_array_to_spherical_array(previous_vertex_coordinate)
        current_vertex_spherical_polar_coordinates = convert_cartesian_array_to_spherical_array(current_vertex_coordinate)
        next_vertex_spherical_polar_coordinates = convert_cartesian_array_to_spherical_array(next_vertex_coordinate)
        
        next_vertex_theta = next_vertex_spherical_polar_coordinates[1]
        next_vertex_phi = next_vertex_spherical_polar_coordinates[2]
        current_vertex_theta = current_vertex_spherical_polar_coordinates[1]
        current_vertex_phi = current_vertex_spherical_polar_coordinates[2]
        previous_vertex_theta = previous_vertex_spherical_polar_coordinates[1]
        previous_vertex_phi = previous_vertex_spherical_polar_coordinates[2]
        
        #looks like my phi == their phi; my theta == their lambda
        delta_lambda = (next_vertex_theta - previous_vertex_theta)
        sine_value = math.sin(current_vertex_phi)
        print 'delta_lambda:', delta_lambda
        print 'sine_value:', sine_value
        edge_contribution_to_surface_area_of_polygon =  delta_lambda * sine_value
        print 'edge_contribution_to_surface_area_of_polygon:', edge_contribution_to_surface_area_of_polygon

        surface_area_spherical_polygon += edge_contribution_to_surface_area_of_polygon

        current_vertex_index += 1

    surface_area_spherical_polygon = surface_area_spherical_polygon * (-(sphere_radius**2) / 2.)
    assert surface_area_spherical_polygon > 0, "Surface areas of spherical polygons should be > 0 but got: {SA}".format(SA=surface_area_spherical_polygon)
    return surface_area_spherical_polygon

def filter_polygon_vertex_coordinates_for_extreme_proximity(array_ordered_Voronoi_polygon_vertices,sphere_radius):
    '''Merge (take the midpoint of) polygon vertices that are judged to be extremely close together and return the filtered polygon vertex array. The purpose is to alleviate numerical complications that may arise during surface area calculations involving polygons with ultra-close / nearly coplanar vertices.'''
    while 1:
        distance_matrix = scipy.spatial.distance.cdist(array_ordered_Voronoi_polygon_vertices,array_ordered_Voronoi_polygon_vertices,'euclidean')
        maximum_euclidean_distance_between_any_vertices = numpy.amax(distance_matrix)
        vertex_merge_threshold = 0.02 #merge any vertices that are separated by less than 1% of the longest inter-vertex distance (may have to play with this value a bit)
        threshold_assessment_matrix = distance_matrix / maximum_euclidean_distance_between_any_vertices
        row_indices_that_violate_threshold, column_indices_that_violate_threshold = numpy.where((threshold_assessment_matrix < vertex_merge_threshold) & (threshold_assessment_matrix > 0))
        if len(row_indices_that_violate_threshold) > 0 and len(column_indices_that_violate_threshold) > 0:
            for row, column in zip(row_indices_that_violate_threshold,column_indices_that_violate_threshold):
                if not row==column: #ignore diagonal values
                    first_violating_vertex_index = row
                    associated_vertex_index = column
                    new_vertex_at_midpoint = ( array_ordered_Voronoi_polygon_vertices[row] + array_ordered_Voronoi_polygon_vertices[column] ) / 2.0
                    spherical_polar_coords_new_vertex = convert_cartesian_array_to_spherical_array(new_vertex_at_midpoint)
                    spherical_polar_coords_new_vertex[0] = 1.0 #project back to surface of sphere
                    new_vertex_at_midpoint = convert_spherical_array_to_cartesian_array(spherical_polar_coords_new_vertex)
                    array_ordered_Voronoi_polygon_vertices[row] = new_vertex_at_midpoint
                    array_ordered_Voronoi_polygon_vertices = numpy.delete(array_ordered_Voronoi_polygon_vertices,column,0)
                    break 
        else: break #no more violating vertices
    return array_ordered_Voronoi_polygon_vertices


def calculate_surface_area_of_planar_polygon_in_3D_space(array_ordered_Voronoi_polygon_vertices):
    '''Based largely on: http://stackoverflow.com/a/12653810
    Use this function when spherical polygon surface area calculation fails (i.e., lots of nearly-coplanar vertices and negative surface area).'''
    #unit normal vector of plane defined by points a, b, and c
    def unit_normal(a, b, c):
        x = numpy.linalg.det([[1,a[1],a[2]],
             [1,b[1],b[2]],
             [1,c[1],c[2]]])
        y = numpy.linalg.det([[a[0],1,a[2]],
             [b[0],1,b[2]],
             [c[0],1,c[2]]])
        z = numpy.linalg.det([[a[0],a[1],1],
             [b[0],b[1],1],
             [c[0],c[1],1]])
        magnitude = (x**2 + y**2 + z**2)**.5
        return (x/magnitude, y/magnitude, z/magnitude)

    #area of polygon poly
    def poly_area(poly):
        '''Accepts a list of xyz tuples.'''
        assert len(poly) >= 3, "Not a polygon (< 3 vertices)."
        total = [0, 0, 0]
        N = len(poly)
        for i in range(N):
            vi1 = poly[i]
            vi2 = poly[(i+1) % N]
            prod = numpy.cross(vi1, vi2)
            total[0] += prod[0]
            total[1] += prod[1]
            total[2] += prod[2]
        result = numpy.dot(total, unit_normal(poly[0], poly[1], poly[2]))
        return abs(result/2)

    list_vertices = [] #need a list of tuples for above function
    for coord in array_ordered_Voronoi_polygon_vertices:
        list_vertices.append(tuple(coord))
    planar_polygon_surface_area = poly_area(list_vertices)
    return planar_polygon_surface_area

def calculate_surface_area_of_a_spherical_Voronoi_polygon(array_ordered_Voronoi_polygon_vertices,sphere_radius):
    '''Calculate the surface area of a polygon on the surface of a sphere. Based on equation provided here: http://mathworld.wolfram.com/SphericalPolygon.html'''
    array_ordered_Voronoi_polygon_vertices = filter_polygon_vertex_coordinates_for_extreme_proximity(array_ordered_Voronoi_polygon_vertices,sphere_radius) #filter vertices for extreme proximity
    theta = calculate_and_sum_up_inner_sphere_surface_angles_Voronoi_polygon(array_ordered_Voronoi_polygon_vertices,sphere_radius)
    n = array_ordered_Voronoi_polygon_vertices.shape[0]
    surface_area_Voronoi_polygon_on_sphere_surface = (theta - ((n - 2) * math.pi)) * (sphere_radius ** 2)
    if surface_area_Voronoi_polygon_on_sphere_surface <= 0: #the spherical polygon surface area could not be calculated properly (in the limit of large numbers of vertices and when vertices that are nearly co-planar are present this seems to be a problem)
        surface_area_Voronoi_polygon_on_sphere_surface = calculate_surface_area_of_planar_polygon_in_3D_space(array_ordered_Voronoi_polygon_vertices) #just use planar area as estimate, which may be quite appropriate with many of the failure /edge cases in any case
    return surface_area_Voronoi_polygon_on_sphere_surface

def calculate_and_sum_up_inner_sphere_surface_angles_Voronoi_polygon(array_ordered_Voronoi_polygon_vertices,sphere_radius):
    '''Takes an array of ordered Voronoi polygon vertices (for a single generator) and calculates the sum of the inner angles on the sphere surface. The resulting value is theta in the equation provided here: http://mathworld.wolfram.com/SphericalPolygon.html '''
    num_vertices_in_Voronoi_polygon = array_ordered_Voronoi_polygon_vertices.shape[0] #the number of rows == number of vertices in polygon
    #two edges (great circle arcs actually) per vertex are needed to calculate tangent vectors / inner angle at that vertex
    current_vertex_index = 0
    list_Voronoi_poygon_angles_radians = []
    while current_vertex_index < num_vertices_in_Voronoi_polygon:
        #print '-------------'
        #print 'current_vertex_index:', current_vertex_index
        current_vertex_coordinate = array_ordered_Voronoi_polygon_vertices[current_vertex_index]
        if current_vertex_index == 0:
            previous_vertex_index = num_vertices_in_Voronoi_polygon - 1
        else:
            previous_vertex_index = current_vertex_index - 1
        if current_vertex_index == num_vertices_in_Voronoi_polygon - 1:
            next_vertex_index = 0
        else:
            next_vertex_index = current_vertex_index + 1
        #try using the law of cosines to produce the angle at the current vertex (basically using a subtriangle, which is a common strategy anyway)
        current_vertex = array_ordered_Voronoi_polygon_vertices[current_vertex_index] 
        previous_vertex = array_ordered_Voronoi_polygon_vertices[previous_vertex_index]
        next_vertex = array_ordered_Voronoi_polygon_vertices[next_vertex_index] 
        #print 'subtriangle vertex coords:', previous_vertex,current_vertex,next_vertex
        #produce a,b,c for law of cosines using spherical distance (http://mathworld.wolfram.com/SphericalDistance.html)
        a = math.acos(numpy.dot(current_vertex,next_vertex))
        b = math.acos(numpy.dot(next_vertex,previous_vertex))
        c = math.acos(numpy.dot(previous_vertex,current_vertex))
        #print 'a,b,c side lengths on subtriangle:', a, b, c
        #try outputting straight-line Euclidean distances for debugging comparison, as I think there are some suspect edge lengths coming out from this
        a_euclid = scipy.spatial.distance.euclidean(current_vertex,next_vertex)
        b_euclid = scipy.spatial.distance.euclidean(next_vertex,previous_vertex)
        c_euclid = scipy.spatial.distance.euclidean(previous_vertex,current_vertex)
        #print 'Euclidean edge lengths (debug):', a_euclid,b_euclid,c_euclid
        current_vertex_inner_angle_on_sphere_surface = math.acos((math.cos(b) - math.cos(a)*math.cos(c)) / (math.sin(a)*math.sin(c)))
        #print 'current vertex inner angle (degrees):', math.degrees(current_vertex_inner_angle_on_sphere_surface)

        list_Voronoi_poygon_angles_radians.append(current_vertex_inner_angle_on_sphere_surface)

        current_vertex_index += 1

    theta = numpy.sum(numpy.array(list_Voronoi_poygon_angles_radians))

    return theta 

def calculate_derivative_great_circle_arc_specified_point(edge_coordinates, sphere_radius):
    '''Inspired loosely by http://glowingpython.blogspot.co.uk/2013/02/visualizing-tangent.html
        Basic idea is to calculate the derivative of the great circle arc (spanning over the edge_coordinates) at a specified coordinate, as this will be important in the calculation of spherical polygon surface areas. Would be convenient to return a vector for the derivative line.'''
    #now based on http://stackoverflow.com/a/1342706:
    derivative_estimate_vector = numpy.cross(edge_coordinates[1],numpy.cross(edge_coordinates[0],edge_coordinates[1]))
    return derivative_estimate_vector #when calculating the angle between these in another function, would probably want to translate the derivative vertex to the origin

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
    sphere_radius: float
        Radius of the sphere (providing radius is more accurate than forcing an estimate). Default None (force estimation).
    sphere_center_origin_offset_vector: array, shape (3,)
        A 1D numpy array that can be subtracted from the generators (original data points) to translate the center of the sphere back to the origin. Default None assumes already centered at origin.
    
    References
    ----------
    
    .. [Caroli] Caroli et al. (2009) INRIA 7004
    
    '''

    def __init__(self,points,sphere_radius=None,sphere_center_origin_offset_vector=None):
        if numpy.all(sphere_center_origin_offset_vector):
            self.original_point_array = points - sphere_center_origin_offset_vector #translate generator data such that sphere center is at origin
        else:
            self.original_point_array = points
        self.sphere_centroid = numpy.average(self.original_point_array,axis=0)
        if not sphere_radius:
            self.estimated_sphere_radius = numpy.average(scipy.spatial.distance.cdist(self.original_point_array,self.sphere_centroid[numpy.newaxis,:]))
        else: 
            self.estimated_sphere_radius = sphere_radius #if the radius of the sphere is known, it is pobably best to specify to avoid centroid bias in radius estimation, etc.
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

        #now I want a useful dictionary data structure containing the surface areas of the Voronoi regions
        dictionary_Voronoi_region_surface_areas_for_each_generator = {}
        for generator_index, Voronoi_polygon_sorted_vertex_array in dictionary_sorted_Voronoi_point_coordinates_for_each_generator.iteritems():
            current_Voronoi_polygon_surface_area_on_sphere = calculate_surface_area_of_a_spherical_Voronoi_polygon(Voronoi_polygon_sorted_vertex_array,self.estimated_sphere_radius)
            assert current_Voronoi_polygon_surface_area_on_sphere > 0, "Obtained a surface area of zero for a Voronoi region."
            dictionary_Voronoi_region_surface_areas_for_each_generator[generator_index] = current_Voronoi_polygon_surface_area_on_sphere

        return (generator_Voronoi_region_dictionary, dictionary_sorted_Voronoi_point_coordinates_for_each_generator,dictionary_Voronoi_region_surface_areas_for_each_generator)

        


