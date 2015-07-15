'''Author: Tyler Reddy

The purpose of this Python module is to provide utility code for handling spherical Voronoi Diagrams.'''

import scipy
try:
    if int(scipy.__version__.split('.')[1]) < 13:
        raise ImportError('Module requires version of scipy module >= 0.13.0')
except AttributeError: #handle this for sphinx build process on readthedocs because of module mocking
    pass 

import circumcircle
import scipy.spatial
import numpy
import numpy.linalg
import pandas
import math
import numpy.random

class IntersectionError(Exception):
    pass

def filter_tetrahedron_to_triangle(current_tetrahedron_coord_array):
    current_triangle_coord_array = [] #initialize as a list
    for row in current_tetrahedron_coord_array: #ugly to use for loop for this, but ok for now!
        if row[0] == 0 and row[1] == 0 and row[2] == 0: #filter out origin row
            continue
        else:
            current_triangle_coord_array.append(row)
    current_triangle_coord_array = numpy.array(current_triangle_coord_array)
    return current_triangle_coord_array

def test_polygon_for_self_intersection(array_ordered_Voronoi_polygon_vertices_2D):
    '''Test an allegedly properly-ordered numpy array of Voronoi region vertices in 2D for self-intersection of edges based on algorithm described at http://algs4.cs.princeton.edu/91primitives/'''
    total_vertices = array_ordered_Voronoi_polygon_vertices_2D.shape[0]
    total_edges = total_vertices

    def intersection_test(a,b,c,d):
        #code in r & s equations provided on above website, which operate on the 2D coordinates of the edge vertices for edges a - b and c - d
        #so: a, b, c, d are numpy arrays of vertex coordinates -- presumably with shape (2,)
        intersection = False
        denominator = (b[0] - a[0]) * (d[1] - c[1]) - (b[1] - a[1]) * (d[0] - c[0])
        r = ( (a[1] - c[1]) * (d[0] - c[0]) - (a[0] - c[0]) * (d[1] - c[1]) ) / denominator
        s = ( (a[1] - c[1]) * (b[0] - a[0]) - (a[0] - c[0]) * (b[1] - a[1]) ) / denominator
        if (r >= 0 and r <= 1) and (s >= 0 and s <= 1): #conditions for intersection
            intersection = True
        if intersection:
            raise IntersectionError("Voronoi polygon line intersection !")

    #go through and test all possible non-consecutive edge combinations for intersection
    list_vertex_indices_in_edges = [ [vertex_index, vertex_index + 1] for vertex_index in xrange(total_vertices)]
    #for the edge starting from the last point in the Voronoi polygon the index of the final point should be switched to the starting index -- to close the polygon
    filtered_list_vertex_indices_in_edges = []
    for list_vertex_indices_in_edge in list_vertex_indices_in_edges:
        if list_vertex_indices_in_edge[1] == total_vertices:
            filtered_list_vertex_indices_in_edge = [list_vertex_indices_in_edge[0],0]
        else:
            filtered_list_vertex_indices_in_edge = list_vertex_indices_in_edge
        filtered_list_vertex_indices_in_edges.append(filtered_list_vertex_indices_in_edge)

    for edge_index, list_vertex_indices_in_edge in enumerate(filtered_list_vertex_indices_in_edges):
        for edge_index_2, list_vertex_indices_in_edge_2 in enumerate(filtered_list_vertex_indices_in_edges):
            if (list_vertex_indices_in_edge[0] not in list_vertex_indices_in_edge_2) and (list_vertex_indices_in_edge[1] not in list_vertex_indices_in_edge_2): #non-consecutive edges
                a = array_ordered_Voronoi_polygon_vertices_2D[list_vertex_indices_in_edge[0]]
                b = array_ordered_Voronoi_polygon_vertices_2D[list_vertex_indices_in_edge[1]]
                c = array_ordered_Voronoi_polygon_vertices_2D[list_vertex_indices_in_edge_2[0]]
                d = array_ordered_Voronoi_polygon_vertices_2D[list_vertex_indices_in_edge_2[1]]
                intersection_test(a,b,c,d)


def calculate_Vincenty_distance_between_spherical_points(cartesian_array_1,cartesian_array_2,sphere_radius):
    '''Apparently, the special case of the Vincenty formula (http://en.wikipedia.org/wiki/Great-circle_distance) may be the most accurate method for calculating great-circle distances.'''
    spherical_array_1 = convert_cartesian_array_to_spherical_array(cartesian_array_1)
    spherical_array_2 = convert_cartesian_array_to_spherical_array(cartesian_array_2)
    lambda_1 = spherical_array_1[1]
    lambda_2 = spherical_array_2[1]
    phi_1 = spherical_array_1[2]
    phi_2 = spherical_array_2[2]
    delta_lambda = abs(lambda_2 - lambda_1)
    delta_phi = abs(phi_2 - phi_1)
    radian_angle = math.atan2( math.sqrt( (math.sin(phi_2)*math.sin(delta_lambda))**2 + (math.sin(phi_1)*math.cos(phi_2) - math.cos(phi_1)*math.sin(phi_2)*math.cos(delta_lambda)  )**2 ),  (math.cos(phi_1) * math.cos(phi_2) + math.sin(phi_1) * math.sin(phi_2) * math.cos(delta_lambda) ) )
    spherical_distance = sphere_radius * radian_angle
    return spherical_distance

def calculate_haversine_distance_between_spherical_points(cartesian_array_1,cartesian_array_2,sphere_radius):
    '''Calculate the haversine-based distance between two points on the surface of a sphere. Should be more accurate than the arc cosine strategy. See, for example: http://en.wikipedia.org/wiki/Haversine_formula'''
    spherical_array_1 = convert_cartesian_array_to_spherical_array(cartesian_array_1)
    spherical_array_2 = convert_cartesian_array_to_spherical_array(cartesian_array_2)
    lambda_1 = spherical_array_1[1]
    lambda_2 = spherical_array_2[1]
    phi_1 = spherical_array_1[2]
    phi_2 = spherical_array_2[2]
    #we rewrite the standard Haversine slightly as long/lat is not the same as spherical coordinates - phi differs by pi/4
    spherical_distance = 2.0 * sphere_radius * math.asin(math.sqrt( ((1 - math.cos(phi_2-phi_1))/2.) + math.sin(phi_1) * math.sin(phi_2) * ( (1 - math.cos(lambda_2-lambda_1))/2.)  ))
    return spherical_distance
    
def generate_random_array_spherical_generators(num_generators,sphere_radius,prng_object):
    '''Recoded using standard uniform selector over theta and acos phi, http://mathworld.wolfram.com/SpherePointPicking.html
    Same as in iPython notebook version'''
    u = prng_object.uniform(low=0,high=1,size=num_generators)
    v = prng_object.uniform(low=0,high=1,size=num_generators)
    theta_array = 2 * math.pi * u
    phi_array = numpy.arccos((2*v - 1.0))
    r_array = sphere_radius * numpy.ones((num_generators,))
    spherical_polar_data = numpy.column_stack((r_array,theta_array,phi_array))
    cartesian_random_points = convert_spherical_array_to_cartesian_array(spherical_polar_data)
    #filter out any duplicate generators:
    df_random_points = pandas.DataFrame(cartesian_random_points)
    df_random_points_no_duplicates = df_random_points.drop_duplicates()
    array_random_spherical_generators = df_random_points_no_duplicates.as_matrix()
    return array_random_spherical_generators

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
                    spherical_polar_coords_new_vertex[0] = sphere_radius #project back to surface of sphere
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
    '''Calculate the surface area of a polygon on the surface of a sphere. Based on equation provided here: http://mathworld.wolfram.com/LHuiliersTheorem.html
    Decompose into triangles, calculate excess for each'''
    #have to convert to unit sphere before applying the formula
    spherical_coordinates = convert_cartesian_array_to_spherical_array(array_ordered_Voronoi_polygon_vertices)
    spherical_coordinates[...,0] = 1.0
    array_ordered_Voronoi_polygon_vertices = convert_spherical_array_to_cartesian_array(spherical_coordinates)
    #handle nearly-degenerate vertices on the unit sphere by returning an area close to 0 -- may be better options, but this is my current solution to prevent crashes, etc.
    #seems to be relatively rare in my own work, but sufficiently common to cause crashes when iterating over large amounts of messy data
    if scipy.spatial.distance.pdist(array_ordered_Voronoi_polygon_vertices).min() < (10 ** -7):
        return 10 ** -8
    else:
        n = array_ordered_Voronoi_polygon_vertices.shape[0]
        #point we start from
        root_point = array_ordered_Voronoi_polygon_vertices[0]
        totalexcess = 0
        #loop from 1 to n-2, with point 2 to n-1 as other vertex of triangle
        # this could definitely be written more nicely
        b_point = array_ordered_Voronoi_polygon_vertices[1]
        root_b_dist = calculate_haversine_distance_between_spherical_points(root_point, b_point, 1.0)
        for i in 1 + numpy.arange(n - 2):
            a_point = b_point
            b_point = array_ordered_Voronoi_polygon_vertices[i+1]
            root_a_dist = root_b_dist
            root_b_dist = calculate_haversine_distance_between_spherical_points(root_point, b_point, 1.0)
            a_b_dist = calculate_haversine_distance_between_spherical_points(a_point, b_point, 1.0)
            s = (root_a_dist + root_b_dist + a_b_dist) / 2
            totalexcess += 4 * math.atan(math.sqrt( math.tan(0.5 * s) * math.tan(0.5 * (s-root_a_dist)) * math.tan(0.5 * (s-root_b_dist)) * math.tan(0.5 * (s-a_b_dist))))
        return totalexcess * (sphere_radius ** 2)

def calculate_and_sum_up_inner_sphere_surface_angles_Voronoi_polygon(array_ordered_Voronoi_polygon_vertices,sphere_radius):
    '''Takes an array of ordered Voronoi polygon vertices (for a single generator) and calculates the sum of the inner angles on the sphere surface. The resulting value is theta in the equation provided here: http://mathworld.wolfram.com/SphericalPolygon.html '''
    #if sphere_radius != 1.0:
        #try to deal with non-unit circles by temporarily normalizing the data to radius 1:
        #spherical_polar_polygon_vertices = convert_cartesian_array_to_spherical_array(array_ordered_Voronoi_polygon_vertices)
        #spherical_polar_polygon_vertices[...,0] = 1.0
        #array_ordered_Voronoi_polygon_vertices = convert_spherical_array_to_cartesian_array(spherical_polar_polygon_vertices)

    num_vertices_in_Voronoi_polygon = array_ordered_Voronoi_polygon_vertices.shape[0] #the number of rows == number of vertices in polygon

    #some debugging here -- I'm concerned that some sphere radii are demonstrating faulty projection of coordinates (some have r = 1, while others have r = sphere_radius -- see workflowy for more detailed notes)
    spherical_polar_polygon_vertices = convert_cartesian_array_to_spherical_array(array_ordered_Voronoi_polygon_vertices)
    min_vertex_radius = spherical_polar_polygon_vertices[...,0].min()
    #print 'before array projection check'
    assert sphere_radius - min_vertex_radius < 0.1, "The minimum projected Voronoi vertex r value should match the sphere_radius of {sphere_radius}, but got {r_min}.".format(sphere_radius=sphere_radius,r_min=min_vertex_radius)
    #print 'after array projection check'

    #two edges (great circle arcs actually) per vertex are needed to calculate tangent vectors / inner angle at that vertex
    current_vertex_index = 0
    list_Voronoi_poygon_angles_radians = []
    while current_vertex_index < num_vertices_in_Voronoi_polygon:
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
        #produce a,b,c for law of cosines using spherical distance (http://mathworld.wolfram.com/SphericalDistance.html)
        #old_a = math.acos(numpy.dot(current_vertex,next_vertex))
        #old_b = math.acos(numpy.dot(next_vertex,previous_vertex))
        #old_c = math.acos(numpy.dot(previous_vertex,current_vertex))
        #print 'law of cosines a,b,c:', old_a,old_b,old_c
        #a = calculate_haversine_distance_between_spherical_points(current_vertex,next_vertex,sphere_radius)
        #b = calculate_haversine_distance_between_spherical_points(next_vertex,previous_vertex,sphere_radius)
        #c = calculate_haversine_distance_between_spherical_points(previous_vertex,current_vertex,sphere_radius)
        a = calculate_Vincenty_distance_between_spherical_points(current_vertex,next_vertex,sphere_radius)
        b = calculate_Vincenty_distance_between_spherical_points(next_vertex,previous_vertex,sphere_radius)
        c = calculate_Vincenty_distance_between_spherical_points(previous_vertex,current_vertex,sphere_radius)
        #print 'law of haversines a,b,c:', a,b,c
        #print 'Vincenty edge lengths a,b,c:', a,b,c
        pre_acos_term = (math.cos(b) - math.cos(a)*math.cos(c)) / (math.sin(a)*math.sin(c))
        if abs(pre_acos_term) > 1.0:
            print 'angle calc vertex coords (giving acos violation):', [convert_cartesian_array_to_spherical_array(vertex) for vertex in [current_vertex,previous_vertex,next_vertex]]
            print 'Vincenty edge lengths (giving acos violation) a,b,c:', a,b,c
            print 'pre_acos_term:', pre_acos_term
            #break
        current_vertex_inner_angle_on_sphere_surface = math.acos(pre_acos_term)

        list_Voronoi_poygon_angles_radians.append(current_vertex_inner_angle_on_sphere_surface)

        current_vertex_index += 1

    if abs(pre_acos_term) > 1.0:
        theta = 0
    else:
        theta = numpy.sum(numpy.array(list_Voronoi_poygon_angles_radians))

    return theta 

def convert_cartesian_array_to_spherical_array(coord_array,angle_measure='radians'):
    '''Take shape (N,3) cartesian coord_array and return an array of the same shape in spherical polar form (r, theta, phi). Based on StackOverflow response: http://stackoverflow.com/a/4116899
    use radians for the angles by default, degrees if angle_measure == 'degrees' '''
    spherical_coord_array = numpy.zeros(coord_array.shape)
    xy = coord_array[...,0]**2 + coord_array[...,1]**2
    spherical_coord_array[...,0] = numpy.sqrt(xy + coord_array[...,2]**2)
    spherical_coord_array[...,1] = numpy.arctan2(coord_array[...,1], coord_array[...,0])
    spherical_coord_array[...,2] = numpy.arccos(coord_array[...,2] / spherical_coord_array[...,0])
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
    cartesian_coord_array[...,0] = spherical_coord_array[...,0] * numpy.cos(spherical_coord_array[...,1]) * numpy.sin(spherical_coord_array[...,2])
    cartesian_coord_array[...,1] = spherical_coord_array[...,0] * numpy.sin(spherical_coord_array[...,1]) * numpy.sin(spherical_coord_array[...,2])
    cartesian_coord_array[...,2] = spherical_coord_array[...,0] * numpy.cos(spherical_coord_array[...,2])
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
    #draft numpy vectorized workflow to avoid Python for loop
    facet_normals_array = numpy.cross(facet_coordinate_array_Delaunay_triangulation[...,1,...] - facet_coordinate_array_Delaunay_triangulation[...,0,...],facet_coordinate_array_Delaunay_triangulation[...,2,...] - facet_coordinate_array_Delaunay_triangulation[...,0,...])
    facet_normal_magnitudes = numpy.linalg.norm(facet_normals_array,axis=1)
    facet_normal_unit_vector_array = facet_normals_array / numpy.column_stack((facet_normal_magnitudes,facet_normal_magnitudes,facet_normal_magnitudes))
    #try to ensure that facet normal faces the correct direction (i.e., out of sphere)
    triangle_centroid_array = numpy.average(facet_coordinate_array_Delaunay_triangulation,axis=1)
    #normalize the triangle_centroid to unit sphere distance for the purposes of the following directionality check
    array_triangle_centroid_spherical_coords = convert_cartesian_array_to_spherical_array(triangle_centroid_array)
    array_triangle_centroid_spherical_coords[...,0] = 1.0
    triangle_centroid_array = convert_spherical_array_to_cartesian_array(array_triangle_centroid_spherical_coords)
    #the Euclidean distance between the triangle centroid and the facet normal should be smaller than the sphere centroid to facet normal distance, otherwise, need to invert the vector
    triangle_to_normal_distance_array = numpy.linalg.norm(triangle_centroid_array - facet_normal_unit_vector_array,axis=1)
    sphere_centroid_to_normal_distance_array = numpy.linalg.norm(sphere_centroid-facet_normal_unit_vector_array,axis=1)
    delta_value_array = sphere_centroid_to_normal_distance_array - triangle_to_normal_distance_array
    facet_normal_unit_vector_array[delta_value_array < -0.1] *= -1.0 #need to rotate the vector so that it faces out of the circle 
    facet_normal_unit_vector_array *= sphere_radius #adjust for radius of sphere
    array_Voronoi_vertices = facet_normal_unit_vector_array
    assert array_Voronoi_vertices.shape[1] == 3, "The array of Voronoi vertices on the sphere should have shape (N,3)."
    return array_Voronoi_vertices


class Voronoi_Sphere_Surface:
    '''Voronoi diagrams on the surface of a sphere.
    
    Parameters
    ----------
    points : *array, shape (npoints, 3)*
        Coordinates of points used to construct a Voronoi diagram on the surface of a sphere.
    sphere_radius : *float*
        Radius of the sphere (providing radius is more accurate than forcing an estimate). Default: None (force estimation).
    sphere_center_origin_offset_vector : *array, shape (3,)*
        A 1D numpy array that can be subtracted from the generators (original data points) to translate the center of the sphere back to the origin. Default: None assumes already centered at origin.
    
    Notes
    -----

    The spherical Voronoi diagram algorithm proceeds as follows. The Convex Hull of the input points (generators) is calculated, and is equivalent to their Delaunay triangulation on the surface of the sphere [Caroli]_. A 3D Delaunay tetrahedralization is obtained by including the origin of the coordinate system as the fourth vertex of each simplex of the Convex Hull. The circumcenters of all tetrahedra in the system are calculated and projected to the surface of the sphere, producing the Voronoi vertices. The Delaunay tetrahedralization neighbour information is then used to order the Voronoi region vertices around each generator. The latter approach is substantially less sensitive to floating point issues than angle-based methods of Voronoi region vertex sorting.

    The surface area of spherical polygons is calculated by decomposing them into triangles and using L'Huilier's Theorem to calculate the spherical excess of each triangle [Weisstein]_. The sum of the spherical excesses is multiplied by the square of the sphere radius to obtain the surface area of the spherical polygon. For nearly-degenerate spherical polygons an area of approximately 0 is returned by default, rather than attempting the unstable calculation. 

    Empirical assessment of spherical Voronoi algorithm performance suggests quadratic time complexity (loglinear is optimal, but algorithms are more challenging to implement). The reconstitution of the surface area of the sphere, measured as the sum of the surface areas of all Voronoi regions, is closest to 100 % for larger (>> 10) numbers of generators. 

    References
    ----------
    
    .. [Caroli] Caroli et al. Robust and Efficient Delaunay triangulations of points on or close to a sphere. Research Report RR-7004, 2009.
    .. [Weisstein] "L'Huilier's Theorem." From MathWorld--A Wolfram Web Resource. http://mathworld.wolfram.com/LHuiliersTheorem.html
    
    Examples
    --------

    Produce a Voronoi diagram for a pseudo-random set of points on the unit sphere:

    >>> import matplotlib
    >>> import matplotlib.pyplot as plt
    >>> import matplotlib.colors as colors
    >>> from mpl_toolkits.mplot3d import Axes3D
    >>> from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    >>> import numpy as np
    >>> import scipy as sp
    >>> import voronoi_utility
    >>> #pin down the pseudo random number generator (prng) object to avoid certain pathological generator sets
    >>> prng = np.random.RandomState(117) #otherwise, would need to filter the random data to ensure Voronoi diagram is possible
    >>> #produce 1000 random points on the unit sphere using the above seed
    >>> random_coordinate_array = voronoi_utility.generate_random_array_spherical_generators(1000,1.0,prng)
    >>> #produce the Voronoi diagram data
    >>> voronoi_instance = voronoi_utility.Voronoi_Sphere_Surface(random_coordinate_array,1.0)
    >>> dictionary_voronoi_polygon_vertices = voronoi_instance.voronoi_region_vertices_spherical_surface()
    >>> #plot the Voronoi diagram
    >>> fig = plt.figure()
    >>> fig.set_size_inches(2,2)
    >>> ax = fig.add_subplot(111, projection='3d')
    >>> for generator_index, voronoi_region in dictionary_voronoi_polygon_vertices.iteritems():
    ...    random_color = colors.rgb2hex(sp.rand(3))
    ...    #fill in the Voronoi region (polygon) that contains the generator:
    ...    polygon = Poly3DCollection([voronoi_region],alpha=1.0)
    ...    polygon.set_color(random_color)
    ...    ax.add_collection3d(polygon)
    >>> ax.set_xlim(-1,1);ax.set_ylim(-1,1);ax.set_zlim(-1,1);
    (-1, 1)
    (-1, 1)
    (-1, 1)
    >>> ax.set_xticks([-1,1]);ax.set_yticks([-1,1]);ax.set_zticks([-1,1]); #doctest: +ELLIPSIS
    [<matplotlib.axis.XTick object at 0x...>, <matplotlib.axis.XTick object at 0x...>]
    [<matplotlib.axis.XTick object at 0x...>, <matplotlib.axis.XTick object at 0x...>]
    [<matplotlib.axis.XTick object at 0x...>, <matplotlib.axis.XTick object at 0x...>]
    >>> plt.tick_params(axis='both', which='major', labelsize=6)

    .. image:: example_random_Voronoi_plot.png

    Now, calculate the surface areas of the Voronoi region polygons and verify that the reconstituted surface area is sensible:

    >>> import math
    >>> dictionary_voronoi_polygon_surface_areas = voronoi_instance.voronoi_region_surface_areas_spherical_surface()
    >>> theoretical_surface_area_unit_sphere = 4 * math.pi
    >>> reconstituted_surface_area_Voronoi_regions = sum(dictionary_voronoi_polygon_surface_areas.itervalues())
    >>> percent_area_recovery = round((reconstituted_surface_area_Voronoi_regions / theoretical_surface_area_unit_sphere) * 100., 5)
    >>> print percent_area_recovery
    99.91979

    For completeness, produce the Delaunay triangulation on the surface of the unit sphere for the same data set:

    >>> Delaunay_triangles = voronoi_instance.delaunay_triangulation_spherical_surface()
    >>> fig2 = plt.figure()
    >>> fig2.set_size_inches(2,2)
    >>> ax = fig2.add_subplot(111, projection='3d')
    >>> for triangle_coordinate_array in Delaunay_triangles:
    ...     m = ax.plot(triangle_coordinate_array[...,0],triangle_coordinate_array[...,1],triangle_coordinate_array[...,2],c='r',alpha=0.1)
    ...     connecting_array = np.delete(triangle_coordinate_array,1,0)
    ...     n = ax.plot(connecting_array[...,0],connecting_array[...,1],connecting_array[...,2],c='r',alpha=0.1)
    >>> o = ax.scatter(random_coordinate_array[...,0],random_coordinate_array[...,1],random_coordinate_array[...,2],c='k',lw=0,s=0.9)
    >>> ax.set_xlim(-1,1);ax.set_ylim(-1,1);ax.set_zlim(-1,1);
    (-1, 1)
    (-1, 1)
    (-1, 1)
    >>> ax.set_xticks([-1,1]);ax.set_yticks([-1,1]);ax.set_zticks([-1,1]); #doctest: +ELLIPSIS
    [<matplotlib.axis.XTick object at 0x...>, <matplotlib.axis.XTick object at 0x...>]
    [<matplotlib.axis.XTick object at 0x...>, <matplotlib.axis.XTick object at 0x...>]
    [<matplotlib.axis.XTick object at 0x...>, <matplotlib.axis.XTick object at 0x...>]
    >>> plt.tick_params(axis='both', which='major', labelsize=6)

    .. image:: example_Delaunay.png

    '''

    def __init__(self,points,sphere_radius=None,sphere_center_origin_offset_vector=None):
        if numpy.all(sphere_center_origin_offset_vector):
            self.original_point_array = points - sphere_center_origin_offset_vector #translate generator data such that sphere center is at origin
        else:
            self.original_point_array = points
        self.sphere_centroid = numpy.zeros((3,)) #already at origin, or has been moved to origin
        if not sphere_radius:
            self.estimated_sphere_radius = numpy.average(scipy.spatial.distance.cdist(self.original_point_array,self.sphere_centroid[numpy.newaxis,:]))
        else: 
            self.estimated_sphere_radius = sphere_radius #if the radius of the sphere is known, it is pobably best to specify to avoid centroid bias in radius estimation, etc.

    def delaunay_triangulation_spherical_surface(self):
        '''Delaunay tessellation of the points on the surface of the sphere. This is simply the 3D convex hull of the points. Returns a shape (N,3,3) array of points representing the vertices of the Delaunay triangulation on the sphere (i.e., N three-dimensional triangle vertex arrays).'''
        hull = scipy.spatial.ConvexHull(self.original_point_array)
        array_points_vertices_Delaunay_triangulation = produce_triangle_vertex_coordinate_array_Delaunay_sphere(hull)
        return array_points_vertices_Delaunay_triangulation

    def voronoi_region_vertices_spherical_surface(self):
        '''Returns a dictionary with the sorted (non-intersecting) polygon vertices for the Voronoi regions associated with each generator (original data point) index. A dictionary entry would be structured as follows: `{generator_index : array_polygon_vertices, ...}`.'''
        #use strategy for Voronoi region generation discussed at PyData London 2015 with Ross Hemsley and Nikolai Nowaczyk
        #step 2: perform 3D Delaunay triangulation on data set that includes the extra generator
        tri = scipy.spatial.ConvexHull(self.original_point_array) #using ConvexHull is much faster in scipy (vs. Delaunay), but here we only get the triangles on the sphere surface in the simplices object (no longer adding an extra point at the origin at this stage)
        #add the origin to each of the simplices to get the same tetrahedra we'd have gotten from Delaunay tetrahedralization
        simplex_coords = tri.points[tri.simplices] #triangles on surface surface
        simplex_coords = numpy.insert(simplex_coords, 3, numpy.zeros((1,3)), axis = 1)
        #step 3: produce circumspheres / circumcenters of tetrahedra from 3D Delaunay
        array_circumcenter_coords = circumcircle.calc_circumcenter_circumsphere_tetrahedron_vectorized(simplex_coords)
        #step 4: project tetrahedron circumcenters up to the surface of the sphere, to produce the Voronoi vertices
        array_vector_lengths = scipy.spatial.distance.cdist(array_circumcenter_coords, numpy.zeros((1,3)))
        array_Voronoi_vertices = (self.estimated_sphere_radius / numpy.abs(array_vector_lengths)) * array_circumcenter_coords
        #step 5: use the Delaunay tetrahedralization neighbour information to connect the Voronoi vertices around the generators, to produce the Voronoi regions
        dictionary_sorted_Voronoi_point_coordinates_for_each_generator = {}
        array_tetrahedra = simplex_coords
        generator_index = 0
        generator_index_array = numpy.arange(self.original_point_array.shape[0])
        filter_tuple = numpy.where((numpy.expand_dims(tri.simplices, -1) == generator_index_array).any(axis=1))
        df = pandas.DataFrame({'generator_indices' : filter_tuple[1]}, index = filter_tuple[0])
        gb = df.groupby('generator_indices')
        dictionary_generators_and_triangle_indices_containing_those_generators = gb.groups
        for generator in tri.points[:-1]:
            indices_of_triangles_surrounding_generator = dictionary_generators_and_triangle_indices_containing_those_generators[generator_index]
            #pick any one of the triangles surrounding the generator and pick a non-generator vertex
            first_tetrahedron_index = indices_of_triangles_surrounding_generator[0]
            first_tetrahedron = array_tetrahedra[first_tetrahedron_index]
            first_triangle = first_tetrahedron[:-1,...]
            #pick one of the two non-generator vertices in the first triangle
            indices_non_generator_vertices_first_triangle = numpy.unique(numpy.where(first_triangle != generator)[0])
            ordered_list_tetrahedron_indices_surrounding_current_generator = [first_tetrahedron_index] 
            #determine the appropriate ordering of Voronoi vertices to close the Voronoi region (polygon) by traversing the Delaunay neighbour data structure from scipy
            vertices_remaining = len(indices_of_triangles_surrounding_generator) - 1
            #choose the neighbour opposite the first non-generator vertex of the first triangle
            neighbour_tetrahedral_index = tri.neighbors[first_tetrahedron_index][indices_non_generator_vertices_first_triangle[0]]
            ordered_list_tetrahedron_indices_surrounding_current_generator.append(neighbour_tetrahedral_index)
            vertices_remaining -= 1
            
            #for all subsequent triangles it is the common non-generator vertex with the previous neighbour that should be used to propagate the connection chain to the following neighbour
            #the common vertex with the previous neighbour is the the vertex of the previous neighbour that was NOT used to locate the current neighbour
            #since there are only two candidate vertices on the previous neighbour and I've chosen to use the vertex with index 0, the remaining vertex on the previous neighbour is the non-generator vertex with index 1
            common_vertex_coordinate = first_triangle[indices_non_generator_vertices_first_triangle[1]]
            while vertices_remaining > 0:
                current_tetrahedron_index = ordered_list_tetrahedron_indices_surrounding_current_generator[-1]
                current_tetrahedron_coord_array = array_tetrahedra[current_tetrahedron_index]
                current_triangle_coord_array = current_tetrahedron_coord_array[:-1,...]
                indices_candidate_vertices_current_triangle_excluding_generator = numpy.unique(numpy.where(current_triangle_coord_array != generator)[0])
                array_candidate_vertices = current_triangle_coord_array[indices_candidate_vertices_current_triangle_excluding_generator]
                current_tetrahedron_index_for_neighbour_propagation = numpy.unique(numpy.where(current_tetrahedron_coord_array == common_vertex_coordinate)[0])
                next_tetrahedron_index_surrounding_generator = tri.neighbors[current_tetrahedron_index][current_tetrahedron_index_for_neighbour_propagation][0]
                common_vertex_coordinate = array_candidate_vertices[array_candidate_vertices != common_vertex_coordinate] #for the next iteration
                ordered_list_tetrahedron_indices_surrounding_current_generator.append(next_tetrahedron_index_surrounding_generator)
                vertices_remaining -= 1
            dictionary_sorted_Voronoi_point_coordinates_for_each_generator[generator_index] = array_Voronoi_vertices[ordered_list_tetrahedron_indices_surrounding_current_generator]
            generator_index += 1
        return dictionary_sorted_Voronoi_point_coordinates_for_each_generator

    def voronoi_region_surface_areas_spherical_surface(self):
        '''Returns a dictionary with the estimated surface areas of the Voronoi region polygons corresponding to each generator (original data point) index. An example dictionary entry: `{generator_index : surface_area, ...}`.'''
        dictionary_sorted_Voronoi_point_coordinates_for_each_generator = self.voronoi_region_vertices_spherical_surface()
        dictionary_Voronoi_region_surface_areas_for_each_generator = {}
        for generator_index, Voronoi_polygon_sorted_vertex_array in dictionary_sorted_Voronoi_point_coordinates_for_each_generator.iteritems():
            current_Voronoi_polygon_surface_area_on_sphere = calculate_surface_area_of_a_spherical_Voronoi_polygon(Voronoi_polygon_sorted_vertex_array,self.estimated_sphere_radius)
            assert current_Voronoi_polygon_surface_area_on_sphere > 0, "Obtained a surface area of zero for a Voronoi region."
            dictionary_Voronoi_region_surface_areas_for_each_generator[generator_index] = current_Voronoi_polygon_surface_area_on_sphere
        return dictionary_Voronoi_region_surface_areas_for_each_generator

        
if __name__ == "__main__":
    import doctest
    doctest.testmod()
