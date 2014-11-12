import unittest
import numpy
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
        self.unit_circle_great_arc_array_deriv_slope_one = numpy.array([[-1,0,0],[-0.5,0.5,0]]) #should have second-point derivative of exactly 1.0 (-x/y = 1.0)

    def tearDown(self):
        del self.unit_circle_great_arc_array
        del self.unit_circle_radius

    def test_3D_unit_circle_great_arc_derivative_zero(self):
        '''Test 3D unit sphere great arc derivative at the equator (basically derivative of 3D unit circle) at a point where the 2D slope should be zero.''' 
        derivative_vector_3D = voronoi_utility.calculate_derivative_great_circle_arc_specified_point(self.unit_circle_great_arc_array,self.unit_circle_radius)
        #because Z is 0 for the input great circle (it is planar even if it is in 3D space) we can basically say that the slope (with respect to y) is zero (the derivative vector should be approx a flat line)
        difference_array = numpy.diff(derivative_vector_3D,axis=0) #should be [dx,dy,dz]
        derivative_estimate = -difference_array[0,1] / difference_array[0,0] # dy/dx slope should be close to zero (negative because want to subtract final point from first)
        numpy.testing.assert_approx_equal(derivative_estimate,0.0,significant=5,err_msg="Derivative should be close to 0.")

    def test_3D_unit_circle_great_arc_derivative_NON_zero(self):
        '''Test 3D unit sphere great arc derivative at the equator (basically derivative of 3D unit circle) at a point where the 2D slope should be ONE.
        Inspired by http://betterexplained.com/articles/calculus-building-intuition-for-the-derivative/
        We know that the derivative of a unit circle at point (x,y) is -x/y for all points where y is not = 0. I think this should be fine for 3D as well as long as I stick with an equatorial (Z=0) input data set.''' 
        derivative_vector_3D = voronoi_utility.calculate_derivative_great_circle_arc_specified_point(self.unit_circle_great_arc_array_deriv_slope_one,self.unit_circle_radius)
        difference_array = numpy.diff(derivative_vector_3D,axis=0)
        derivative_estimate = -difference_array[0,1] / difference_array[0,0] # dy/dx slope should be close to ONE
        numpy.testing.assert_approx_equal(derivative_estimate,1.0,significant=5,err_msg="Derivative should be close to 1.")


        
        
        




