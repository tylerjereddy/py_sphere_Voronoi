import unittest
import numpy
import numpy.testing
import scipy
import scipy.spatial
import math
import voronoi_utility
import networkx as nx

class Test_delaunay_triangulation_on_sphere_surface(unittest.TestCase):

    def setUp(self):
        #simple sphere parameters for testing a small triangulation:
        self.simple_sphere_circumdiameter = 4.0
        self.u, self.v = numpy.mgrid[0:2*numpy.pi:20j, 0:numpy.pi:10j]
        self.big_u,self.big_v = numpy.mgrid[0:2*numpy.pi:2000j, 0:numpy.pi:1000j]

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

        #assign an integer node (vertex) number to each unique coordinate on the test sphere:
        for node_coordinate in self.simple_sphere_coordinate_array:
            node_dictionary[node_counter] = node_coordinate
            node_counter += 1

        def identify_node_based_on_coordinate(coordinate_array,node_dictionary):
            '''Return the node number based on the coordinates in the original test sphere data set.'''
            nodenum = 0
            for node_number, node_coordinates in node_dictionary.iteritems():
                if numpy.allclose(node_coordinates,coordinate_array):
                    nodenum = node_number
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
            print 'Triangle:', triangle_counter, 'total edges:', G.size(), 'total nodes:', G.order()

        #print 'size:', G.size()
        #print 'list of edges:', G.edges()
        #print 'num nodes:', len(G.nodes())
        #print 'dict size:', len(node_dictionary)
        #print 'dict keys:', node_dictionary.keys()
        #print 'nodes:', G.nodes()
        #print 'edges:', G.edges()

        self.assertEqual(len(G),self.num_triangulation_input_points) #obviously, the number of nodes in the graph should match the number of points on the sphere




