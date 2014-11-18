import unittest
import numpy
import numpy.linalg
import numpy.random
import numpy.testing
import scipy
import scipy.spatial
import math
import voronoi_utility
import networkx as nx
import random

class Test_delaunay_triangulation_on_sphere_surface(unittest.TestCase):

    def setUp(self):
        #simple sphere parameters for testing a small triangulation:
        self.simple_sphere_circumdiameter = 4.0
        self.u, self.v = numpy.mgrid[0.01:2*numpy.pi-0.01:20j, 0.01:numpy.pi-0.01:10j]
        self.big_u,self.big_v = numpy.mgrid[0:2*numpy.pi:2000j, 0:numpy.pi:1000j] #will have to adjust the boundaries here as well if the large data set is also to be used in testing

        def generate_sphere_coords(u,v):
            x = (self.simple_sphere_circumdiameter/2.0 * (numpy.cos(u)*numpy.sin(v))).ravel()
            y = (self.simple_sphere_circumdiameter/2.0 * (numpy.sin(u)*numpy.sin(v))).ravel()
            z = (self.simple_sphere_circumdiameter/2.0 * (numpy.cos(v))).ravel()
            simple_sphere_coordinate_array = numpy.zeros((x.size,3))
            simple_sphere_coordinate_array[...,0] = x
            simple_sphere_coordinate_array[...,1] = y
            simple_sphere_coordinate_array[...,2] = z
            return simple_sphere_coordinate_array

        #trying to work with small and large N points for triangulation to generate data for asymptotic limit test
        self.simple_sphere_coordinate_array = generate_sphere_coords(self.u,self.v)
        self.simple_sphere_coordinate_array_LARGE = generate_sphere_coords(self.big_u,self.big_v)

        self.num_triangulation_input_points = self.simple_sphere_coordinate_array.shape[0]
        self.num_triangulation_input_points_LARGE = self.simple_sphere_coordinate_array_LARGE.shape[0]

    def tearDown(self):
        del self.simple_sphere_circumdiameter
        del self.u
        del self.v
        del self.big_u
        del self.big_v
        del self.simple_sphere_coordinate_array
        del self.simple_sphere_coordinate_array_LARGE
        del self.num_triangulation_input_points
        del self.num_triangulation_input_points_LARGE

    @unittest.skip("Test not working yet.")
    def test_asymptotic_upper_bound_polytopes_Delaunay_sphere_surface(self):
        '''Any triangulation of n points in d dimensions has O(n ** ( d / 2)) simplices (triangles). This is based on the asymptotic version of the upper bound theorem for polytopes [Seidel1995]_. This doesn't appear to be an expression that may be used to determine the exact number of triangles in a Delaunay triangulation based solely on the number of input points and dimensionality, even though that would be rahter convenient for testing purposes. Rather, the O-notation implies a description of the growth behaviour of the number of simplices as the number of input points approaches infinity, by convention neglecting terms in the equation (for the total number of simplices) that contribute negligibly in the asymptotic limit. This property should hold for the Delaunay triangulation of the points on the surface of a sphere, since it is a property of any triangulation.

        References
        ----------

        .. [Seidel1995] R. Seidel (1995) Computational Geometry 5: 115-116. 

        '''
        dimensions = 3.0
        increase_in_number_data_set_points = self.num_triangulation_input_points_LARGE - self.num_triangulation_input_points
        expected_approximate_increase_in_num_triangulation_facets = increase_in_number_data_set_points ** (dimensions / 2.0)
        print 'expected_approximate_increase_in_num_triangulation_facets:', expected_approximate_increase_in_num_triangulation_facets

        voronoi_instance_small = voronoi_utility.Voronoi_Sphere_Surface(self.simple_sphere_coordinate_array)
        voronoi_instance_large = voronoi_utility.Voronoi_Sphere_Surface(self.simple_sphere_coordinate_array_LARGE)
        actual_Delaunay_point_array_small = voronoi_instance_small.Delaunay_triangulation_spherical_surface()
        actual_Delaunay_point_array_large = voronoi_instance_large.Delaunay_triangulation_spherical_surface()
        actual_increase_in_num_triangulation_facets = actual_Delaunay_point_array_large.shape[0] - actual_Delaunay_point_array_small.shape[0]
        print 'actual_increase_in_num_triangulation_facets:', actual_increase_in_num_triangulation_facets

        self.assertEqual(actual_increase_in_num_triangulation_facets,expected_approximate_increase_in_num_triangulation_facets)



    def test_geometric_spanner_condition_Delaunay_triangulation_sphere_surface(self):
        '''The geometric spanner condition (http://en.wikipedia.org/wiki/Delaunay_triangulation#Properties) indicates that the length of the shortest edge-traveled path between two nodes in a Delaunay triangulation is no longer than 2.42 times the straight-line Euclidean distance between them.'''
        #create a networkx graph object of the Delaunay triangulation vertices & edges

        
        voronoi_instance_small = voronoi_utility.Voronoi_Sphere_Surface(self.simple_sphere_coordinate_array)
        Delaunay_point_array_small = voronoi_instance_small.Delaunay_triangulation_spherical_surface() #should be shape (N,3,3) for N triangles and their vertices in 3D space

        node_dictionary = {}
        node_counter = 1
        list_nodes_identified_debug = []

        #assign an integer node (vertex) number to each unique coordinate on the test sphere:
        for node_coordinate in self.simple_sphere_coordinate_array:
            node_dictionary[node_counter] = node_coordinate
            node_counter += 1

        #print 'self.simple_sphere_coordinate_array:', self.simple_sphere_coordinate_array
        #print 'self.simple_sphere_coordinate_array.shape:', self.simple_sphere_coordinate_array.shape
        #print 'node_dictionary.values():', node_dictionary.values() #there seem to be multiple duplicates / rounding variations for the polar point at [0. 0. 2.]

        def identify_node_based_on_coordinate(coordinate_array,node_dictionary):
            '''Return the node number based on the coordinates in the original test sphere data set.'''
            nodenum = 0
            num_positives_debug = 0
            for node_number, node_coordinates in node_dictionary.iteritems():
                if numpy.allclose(node_coordinates,coordinate_array,atol=1e-18):
                    nodenum = node_number
                    num_positives_debug += 1
                    #if num_positives_debug > 1:
                        #print 'duplicate offender:', node_coordinates, coordinate_array
                    #else:
                        #print 'original match:', node_coordinates, coordinate_array
            assert num_positives_debug == 1, "Only a single node should correspond to the input coordinates."
            return nodenum

        def produce_networkx_edges_from_triangle_data(triangle_array_data,node_dictionary):
            '''Input should be shape (3,3) array of coordinate data for a Delaunay triangle.'''
            list_networkx_edge_tuples = []
            #each triangle will, of course, have 3 edges
            triangle_array_row_indices_for_edge_vertices = [[0,1],[1,2],[2,0]]
            for triangle_row_indices_of_edge in triangle_array_row_indices_for_edge_vertices:
                first_triangle_row_index, second_triangle_row_index = triangle_row_indices_of_edge
                first_vertex_coord = triangle_array_data[first_triangle_row_index]
                second_vertex_coord = triangle_array_data[second_triangle_row_index]
                graph_node_number_first_vertex_current_edge = identify_node_based_on_coordinate(first_vertex_coord,node_dictionary)
                graph_node_number_second_vertex_current_edge = identify_node_based_on_coordinate(second_vertex_coord,node_dictionary)
                list_nodes_identified_debug.extend([graph_node_number_first_vertex_current_edge,graph_node_number_second_vertex_current_edge]) #missing nodes with debug list growth here, but no missing nodes if I grow the debug list from within the identify_node_based_on_coordinate() function itself; why????
                #the edge weight for networkx should be the Euclidean straight-line distance between the vertices
                weight = scipy.spatial.distance.euclidean(first_vertex_coord,second_vertex_coord)
                networkx_edge_tuple = (graph_node_number_first_vertex_current_edge,graph_node_number_second_vertex_current_edge,weight)
                list_networkx_edge_tuples.append(networkx_edge_tuple)
            return list_networkx_edge_tuples

        #build networkx graph object from Delaunay triangles (simplices)
        G = nx.Graph()
        triangle_counter = 0
        for triangle_coord_array in Delaunay_point_array_small: 
            current_list_networkx_edge_tuples = produce_networkx_edges_from_triangle_data(triangle_coord_array,node_dictionary)
            #print 'current_list_networkx_edge_tuples:', current_list_networkx_edge_tuples
            G.add_weighted_edges_from(current_list_networkx_edge_tuples) #duplicates will simply be updated
            triangle_counter += 1
            #print 'Triangle:', triangle_counter, 'total edges:', G.size(), 'total nodes:', G.order()

        #print 'size:', G.size()
        #print 'list of edges:', G.edges()
        #print 'num nodes:', len(G.nodes())
        #print 'dict size:', len(node_dictionary)
        #print 'dict keys:', node_dictionary.keys()
        #print 'nodes:', G.nodes()
        #print 'edges:', G.edges()

        #print 'ordered set nodes identified:', set(sorted(list_nodes_identified_debug))
        self.assertEqual(len(G),self.num_triangulation_input_points) #obviously, the number of nodes in the graph should match the number of points on the sphere

        #perform the geometric spanner test for a random subset of nodes:
        num_tests = 0
        while num_tests < 10: #do 10 random tests 
            first_node_number = random.randrange(1,len(G),1)
            second_node_number = random.randrange(1,len(G),1)
            minimum_distance_between_nodes_following_Delaunay_edges = nx.dijkstra_path_length(G,first_node_number,second_node_number)
            #compare with straight line Euclidean distance:
            first_node_coordinate = node_dictionary[first_node_number]
            second_node_coordinate = node_dictionary[second_node_number]
            Euclidean_distance = scipy.spatial.distance.euclidean(first_node_coordinate,second_node_coordinate)
            self.assertLess(minimum_distance_between_nodes_following_Delaunay_edges/Euclidean_distance,2.42) #the geometric spanner condition
            num_tests += 1
            

class Test_derivative_great_circle_arc(unittest.TestCase):

    def setUp(self):
        self.unit_circle_great_arc_array = numpy.array([[-1,0,0],[0,1,0]]) #should have second-point derivative near zero
        self.unit_circle_radius = 1.0
        self.unit_circle_great_arc_array_deriv_slope_one = numpy.array([[-1,0,0],[-math.sqrt(0.5),math.sqrt(0.5),0]]) #should have second-point derivative of exactly 1.0 (-x/y = 1.0)

    def tearDown(self):
        del self.unit_circle_great_arc_array
        del self.unit_circle_radius

    def test_3D_unit_circle_great_arc_derivative_zero(self):
        '''Test 3D unit sphere great arc derivative at the equator (basically derivative of 3D unit circle) at a point where the 2D slope should be zero.''' 
        derivative_vector_3D = voronoi_utility.calculate_derivative_great_circle_arc_specified_point(self.unit_circle_great_arc_array,self.unit_circle_radius)
        #because Z is 0 for the input great circle (it is planar even if it is in 3D space) we can basically say that the slope (with respect to y) is zero (the derivative vector should be approx a flat line)
        derivative_estimate = derivative_vector_3D[1]/derivative_vector_3D[0] # dy/dx slope should be close to zero, and vectors are origin-based (sphere should have center at origin)
        numpy.testing.assert_approx_equal(derivative_estimate,0.0,significant=5,err_msg="Derivative should be close to 0.")

    def test_3D_unit_circle_great_arc_derivative_NON_zero(self):
        '''Test 3D unit sphere great arc derivative at the equator (basically derivative of 3D unit circle) at a point where the 2D slope should be ONE.
        Inspired by http://betterexplained.com/articles/calculus-building-intuition-for-the-derivative/
        We know that the derivative of a unit circle at point (x,y) is -x/y for all points where y is not = 0. I think this should be fine for 3D as well as long as I stick with an equatorial (Z=0) input data set.''' 
        derivative_vector_3D = voronoi_utility.calculate_derivative_great_circle_arc_specified_point(self.unit_circle_great_arc_array_deriv_slope_one,self.unit_circle_radius)
        derivative_estimate = derivative_vector_3D[1] / derivative_vector_3D[0] # dy/dx slope should be close to ONE
        numpy.testing.assert_approx_equal(derivative_estimate,1.0,significant=5,err_msg="Derivative should be close to 1.")

class Test_spherical_polygon_angle_summation(unittest.TestCase):

    def setUp(self):
        self.spherical_triangle_coordinate_array = numpy.array([[0,0,1],[0,1,0],[1,0,0]]) #3 points on a unit sphere
        self.spherical_polygon_4_vertices_coord_array = numpy.array([[0,0,1],[0,1,0],[1,0,0],[0,-1,0]]) #4 points on a unit sphere

    def tearDown(self):
        del self.spherical_triangle_coordinate_array

    def test_spherical_triangle_sum_inner_angles(self):
        '''Test my spherical polygon inner angle summation code on the simple case of a spherical triangle where we know that the sum of the inner angles must be between pi and 3 * pi radians (http://mathworld.wolfram.com/SphericalTriangle.html).'''
        theta = voronoi_utility.calculate_and_sum_up_inner_sphere_surface_angles_Voronoi_polygon(self.spherical_triangle_coordinate_array,1.0)
        self.assertLess(theta,3. * math.pi,msg='theta must be less than 3 * pi radians for a spherical triangle but got theta = {theta}'.format(theta=theta))
        self.assertGreater(theta,math.pi,msg='theta must be greater than pi radians for a spherical triangle but got theta = {theta}'.format(theta=theta))

    def test_spherical_polygon_4_vertices_sum_inner_angles(self):
        '''Test my spherical polygon inner angle summation code on a slightly more complex case of a spherical polygon with n = 4 vertices. The sum of the inner angles should exceed (n - 2) * pi according to http://mathworld.wolfram.com/SphericalPolygon.html.'''
        theta = voronoi_utility.calculate_and_sum_up_inner_sphere_surface_angles_Voronoi_polygon(self.spherical_polygon_4_vertices_coord_array,1.0)
        minimum_allowed_angle = 2 * math.pi
        self.assertGreater(theta,minimum_allowed_angle,msg='theta must be greater than 2 * pi for spherical polygon with 4 vertices but got theta = {theta}'.format(theta=theta))



class Test_voronoi_surface_area_calculations(unittest.TestCase):

    def setUp(self):
        #generate a random distribution of points on the unit sphere (http://mathworld.wolfram.com/SpherePointPicking.html)
        #go for 1000 random points
        self.u = numpy.random.random((1000,)) #200 points on interval [0,1); ideally want (0,1), but perhaps close enough?
        self.v = numpy.random.random((1000,))
        self.theta_array = 2 * math.pi * self.u
        self.phi_array = numpy.arccos((2*self.v - 1.0))
        self.r_array = numpy.ones((1000,))
        self.spherical_polar_coord_array = numpy.column_stack((self.r_array,self.theta_array,self.phi_array))
        #convert to Cartesian coordinates
        self.cartesian_coord_array = voronoi_utility.convert_spherical_array_to_cartesian_array(self.spherical_polar_coord_array)

        self.spherical_triangle_coordinate_array = numpy.array([[0,0,1],[0,1,0],[1,0,0]]) #3 points on a unit sphere
        self.spherical_polygon_4_vertices_coord_array = numpy.array([[0,0,1],[0,1,0],[1,0,0],[0,-1,0]]) #4 points on a unit sphere

    def tearDown(self):
        del self.cartesian_coord_array
        del self.spherical_polar_coord_array
        del self.spherical_triangle_coordinate_array


    def test_spherical_voronoi_regular_surface_area_reconstitution(self):
        '''Surface area reconstitution for Voronoi diagram generated from regularly spaced points on unit sphere.'''
        unit_sphere_surface_area = 4 * math.pi
        circumdiameter = 2.0
        u, v = numpy.mgrid[0.01:2*numpy.pi-0.01:20j, 0.01:numpy.pi-0.01:10j]
        x=circumdiameter/2.0 * (numpy.cos(u)*numpy.sin(v))
        y=circumdiameter/2.0 * (numpy.sin(u)*numpy.sin(v))
        z=circumdiameter/2.0 * (numpy.cos(v))
        input_sphere_coordinate_array = numpy.zeros((200,3))
        input_sphere_coordinate_array[...,0] = x.ravel()
        input_sphere_coordinate_array[...,1] = y.ravel()
        input_sphere_coordinate_array[...,2] = z.ravel()
        voronoi_instance = voronoi_utility.Voronoi_Sphere_Surface(input_sphere_coordinate_array)
        dictionary_sorted_Voronoi_point_coordinates_for_each_generator = voronoi_instance.Voronoi_polygons_spherical_surface()[1]
        sum_Voronoi_polygon_surface_areas = 0
        for generator_index, Voronoi_polygon_sorted_vertex_array in dictionary_sorted_Voronoi_point_coordinates_for_each_generator.iteritems():
            for vector in Voronoi_polygon_sorted_vertex_array:
                norm = numpy.linalg.norm(vector)
                self.assertGreater(norm,0.99,'Vector to polygon vertex should have length near 1.0 but got length = {norm}'.format(norm=norm))
            print 'generator_index:', generator_index
            print 'Voronoi_polygon_sorted_vertex_array:', Voronoi_polygon_sorted_vertex_array
            current_Voronoi_polygon_surface_area_on_sphere = voronoi_utility.calculate_surface_area_of_a_spherical_Voronoi_polygon(Voronoi_polygon_sorted_vertex_array,1.0)
            sum_Voronoi_polygon_surface_areas += current_Voronoi_polygon_surface_area_on_sphere
        numpy.testing.assert_almost_equal(sum_Voronoi_polygon_surface_areas, unit_sphere_surface_area,decimal=7,err_msg='Reconstituted surface area of Voronoi polygons on unit sphere should match theoretical surface area of sphere.')

    def test_spherical_voronoi_surface_area_reconstitution(self):
        '''Given a pseudo-random set of points on the unit sphere, the sum of the surface areas of the Voronoi polygons should be equal to the surface area of the sphere itself.'''
        unit_sphere_surface_area = 4 * math.pi
        random_dist_voronoi_instance = voronoi_utility.Voronoi_Sphere_Surface(self.cartesian_coord_array)
        dictionary_sorted_Voronoi_point_coordinates_for_each_generator = random_dist_voronoi_instance.Voronoi_polygons_spherical_surface()[1]
        sum_Voronoi_polygon_surface_areas = 0
        for generator_index, Voronoi_polygon_sorted_vertex_array in dictionary_sorted_Voronoi_point_coordinates_for_each_generator.iteritems():
            for vector in Voronoi_polygon_sorted_vertex_array:
                norm = numpy.linalg.norm(vector)
                self.assertGreater(norm,0.99,'Vector to polygon vertex should have length near 1.0 but got length = {norm}'.format(norm=norm))
            print 'generator_index:', generator_index
            print 'Voronoi_polygon_sorted_vertex_array:', Voronoi_polygon_sorted_vertex_array
            current_Voronoi_polygon_surface_area_on_sphere = voronoi_utility.calculate_surface_area_of_a_spherical_Voronoi_polygon(Voronoi_polygon_sorted_vertex_array,1.0)
            sum_Voronoi_polygon_surface_areas += current_Voronoi_polygon_surface_area_on_sphere
        numpy.testing.assert_almost_equal(sum_Voronoi_polygon_surface_areas, unit_sphere_surface_area,decimal=7,err_msg='Reconstituted surface area of Voronoi polygons on unit sphere should match theoretical surface area of sphere.')
            
    def test_spherical_triangle_surface_area_calculation(self):
        '''Test spherical polygon surface area calculation on the relatively simple case of a spherical triangle on the surface of a unit sphere.'''
        #the surface area of a spherical triangle is a special case of a spherical polygon (http://mathworld.wolfram.com/SphericalTriangle.html)
        sum_spherical_triangle_inner_angles = voronoi_utility.calculate_and_sum_up_inner_sphere_surface_angles_Voronoi_polygon(self.spherical_triangle_coordinate_array,1.0)
        spherical_excess = sum_spherical_triangle_inner_angles - math.pi #because the radius of the sphere is 1 the spherical excess is also the surface area
        self.assertGreater(spherical_excess,0.0)
        test_surface_area = voronoi_utility.calculate_surface_area_of_a_spherical_Voronoi_polygon(self.spherical_triangle_coordinate_array,1.0)
        self.assertEqual(test_surface_area,spherical_excess,msg='Surface area of a spherical triangle (a special case of spherical polygon) not calculated correctly.')

    def test_spherical_polygon_4_vertices_surface_area_calculation(self):
        '''Test spherical polygon surface area calculation on the more complex case of a spherical polygon with 4 vertices on a unit sphere.'''
        sum_spherical_polygon_inner_angles = voronoi_utility.calculate_and_sum_up_inner_sphere_surface_angles_Voronoi_polygon(self.spherical_polygon_4_vertices_coord_array,1.0)
        subtraction_value = 2 * math.pi # (n-2) * pi
        target_area = sum_spherical_polygon_inner_angles - subtraction_value
        self.assertGreater(sum_spherical_polygon_inner_angles,subtraction_value,'The polygon with 4 vertices has a negative surface area.')
        measured_surface_area = voronoi_utility.calculate_surface_area_of_a_spherical_Voronoi_polygon(self.spherical_polygon_4_vertices_coord_array, 1.0)
        self.assertEqual(measured_surface_area,target_area,msg='Surface area of a 4-vertex spherical polygon is not calculated correctly.')
        
    def test_spherical_polygon_4_vertices_nearly_colinear_surface_area_calculation(self):
        '''Test spherical polygon surface area calculation for a polygon with 4 vertices AND an internal angle that is very close to 180 degrees. Trying to stress test / probe possible issues with arc cosine accuracy, etc.'''
        regular_spherical_triangle_coords = self.spherical_triangle_coordinate_array #([[0,0,1],[0,1,0],[1,0,0]]) #3 points on a unit sphere
        #I want to generate a fourth point that is ALMOST on the same great circle arc as two other vertices, because this is a nasty test case
        linear_midpoint_last_two_vertices = (regular_spherical_triangle_coords[1] + regular_spherical_triangle_coords[2]) / 2.0
        linear_midpoint_spherical_polar_coords = voronoi_utility.convert_cartesian_array_to_spherical_array(linear_midpoint_last_two_vertices)
        spherical_midpoint_spherical_polar_coords = numpy.zeros((1,3))
        spherical_midpoint_spherical_polar_coords[0,0] = 1.0
        spherical_midpoint_spherical_polar_coords[0,1] = linear_midpoint_spherical_polar_coords[1] + 0.000001 #slightly off the arc
        spherical_midpoint_spherical_polar_coords[0,2] = linear_midpoint_spherical_polar_coords[2] + 0.000001 #slightly off the arc
        near_midpoint_cartesian = voronoi_utility.convert_spherical_array_to_cartesian_array(spherical_midpoint_spherical_polar_coords)
        polygon_coords = numpy.zeros((4,3))
        polygon_coords[0] = regular_spherical_triangle_coords[0]
        polygon_coords[1] = regular_spherical_triangle_coords[1]
        polygon_coords[2] = near_midpoint_cartesian[0,...]
        polygon_coords[3] = regular_spherical_triangle_coords[2]
        measured_surface_area = voronoi_utility.calculate_surface_area_of_a_spherical_Voronoi_polygon(polygon_coords,1.0) #the function itself will pass an exception if there is a negative surface area
        self.assertGreater(measured_surface_area,0.0)


    def test_spherical_polygon_4_vertices_exactly_colinear_surface_area_calculation(self):
        '''Test spherical polygon surface area calculation for a polygon with 4 vertices--3 of which are on the same great circle arc--this should surely cause a problem!! (Apparently not.)'''
        regular_spherical_triangle_coords = self.spherical_triangle_coordinate_array #([[0,0,1],[0,1,0],[1,0,0]]) #3 points on a unit sphere
        linear_midpoint_last_two_vertices = (regular_spherical_triangle_coords[1] + regular_spherical_triangle_coords[2]) / 2.0
        linear_midpoint_spherical_polar_coords = voronoi_utility.convert_cartesian_array_to_spherical_array(linear_midpoint_last_two_vertices)
        spherical_midpoint_spherical_polar_coords = numpy.zeros((1,3))
        spherical_midpoint_spherical_polar_coords[0,0] = 1.0
        spherical_midpoint_spherical_polar_coords[0,1] = linear_midpoint_spherical_polar_coords[1] 
        spherical_midpoint_spherical_polar_coords[0,2] = linear_midpoint_spherical_polar_coords[2]
        near_midpoint_cartesian = voronoi_utility.convert_spherical_array_to_cartesian_array(spherical_midpoint_spherical_polar_coords)
        polygon_coords = numpy.zeros((4,3))
        polygon_coords[0] = regular_spherical_triangle_coords[0]
        polygon_coords[1] = regular_spherical_triangle_coords[1]
        polygon_coords[2] = near_midpoint_cartesian[0,...]
        polygon_coords[3] = regular_spherical_triangle_coords[2]
        measured_surface_area = voronoi_utility.calculate_surface_area_of_a_spherical_Voronoi_polygon(polygon_coords,1.0) #the function itself will pass an exception if there is a negative surface area
        self.assertGreater(measured_surface_area,0.0)

    def test_problematic_spherical_polygon_surface_area(self):
        '''Test the surface area of a spherical polygon that I know has previously caused issues with negative surface area as part of a Voronoi diagram. Now using a simplified version of that polygon with 1 less vertex--but still get a negative result.'''
        problematic_polygon_array = numpy.array([[-0.12278101, 0.38828208, 0.90397089],
         [-0.18533492 ,0.28384049, 0.9317119 ],
         [ 0.07210294 ,0.29806975, 0.94284522],
         [ 0.1316095  ,0.32464041, 0.92751769]])
        measured_surface_area = voronoi_utility.calculate_surface_area_of_a_spherical_Voronoi_polygon(problematic_polygon_array,1.0)
        self.assertGreater(measured_surface_area,0.0)





            
        

        
        
        
        




