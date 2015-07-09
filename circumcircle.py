# coding: utf-8

import numpy, math

def calc_circumcircle(array_triangle_vertices):
    '''The input array of triangle vertices should have shape (N_triangles, 3, 2) and this function should use the vertex information to calculate and return numpy arrays containing the circumcenters and circumradii for the circumcircles of each triangle.'''
    def sum_squares_vertex(coordinates):
        square_coordinates = [coordinate**2 for coordinate in coordinates]
        sum_squares = sum(square_coordinates)
        return sum_squares
    def calc_sum_squares_each_vertex(vertex_1_coords,vertex_2_coords,vertex_3_coords):
        vertex_1_sum_squares = sum_squares_vertex(vertex_1_coords)
        vertex_2_sum_squares = sum_squares_vertex(vertex_2_coords)
        vertex_3_sum_squares = sum_squares_vertex(vertex_3_coords)
        return (vertex_1_sum_squares, vertex_2_sum_squares, vertex_3_sum_squares)
    list_circumcenters = [] #an ordered list of circumcenter coords
    list_circumradii = [] #a correspondingly ordered list of circumradii
    for triangle in array_triangle_vertices:
        vertex_1_sum_squares, vertex_2_sum_squares, vertex_3_sum_squares = calc_sum_squares_each_vertex(triangle[0,...],triangle[1,...],triangle[2,...])
        x_determinant = numpy.array([[vertex_1_sum_squares, triangle[0,1], 1],\
                                     [vertex_2_sum_squares, triangle[1,1], 1],\
                                     [vertex_3_sum_squares, triangle[2,1], 1]])
        y_determinant = numpy.array([[vertex_1_sum_squares, triangle[0,0], 1],\
                                     [vertex_2_sum_squares, triangle[1,0], 1],\
                                     [vertex_3_sum_squares, triangle[2,0], 1]])
        a_determinant = numpy.array([[triangle[0,0], triangle[0,1], 1],\
                                     [triangle[1,0], triangle[1,1], 1],\
                                     [triangle[2,0], triangle[2,1], 1]])
        c_determinant = numpy.array([[vertex_1_sum_squares,triangle[0,0],triangle[0,1]],\
                                     [vertex_2_sum_squares,triangle[1,0],triangle[1,1]],\
                                     [vertex_3_sum_squares,triangle[2,0],triangle[2,1]]])
        denominator = 2.0 * numpy.linalg.det(a_determinant)
        circumcenter_x_coordinate = numpy.linalg.det(x_determinant) / denominator
        circumcenter_y_coordinate = -1.0 * (numpy.linalg.det(y_determinant)/ denominator)
        #adjuting for the negative coefficient of c:
        circumradius = math.sqrt(numpy.linalg.det(x_determinant)**2 + numpy.linalg.det(y_determinant)**2 + (4 * numpy.linalg.det(a_determinant) * numpy.linalg.det(c_determinant))) / (2 * abs(numpy.linalg.det(a_determinant)))
        #append the values of interest to their respective lists:
        list_circumcenters.append([circumcenter_x_coordinate,circumcenter_y_coordinate])
        list_circumradii.append([circumradius])
    #return a tuple of numpy arrays:
    return (numpy.array(list_circumcenters),numpy.array(list_circumradii))
    
    #call the function to obtain the numpy arrays of circumcenters and circumradii:
    #array_circumcenters, array_circumradii = calc_circumcircle(triangle_vertex_coords)
    #array_circumcenters has shape (N_circles, 2)
    #array_circumradii has shape (N_circles, 1)

    #now plot the circumcircles on top of the Delaunay triangulation
    #start by plotting the circumcenters in red:
#    from matplotlib.collections import PatchCollection
#    patches = []
#    ax1.scatter(array_circumcenters[...,0],array_circumcenters[...,1],c='r',marker='o',alpha = 0.4, edgecolors = 'none')
#    #now plot the circles:
#    for circumcenter_coordinates, circumradius in zip(array_circumcenters,array_circumradii):
#        patches.append(matplotlib.patches.Circle((circumcenter_coordinates[0],circumcenter_coordinates[1]),circumradius[0],color='r',fill=False, alpha=0.4))
#    p = PatchCollection(patches, alpha=0.4,match_original = True)
#    ax1.add_collection(p)
#    fig4.savefig('voronoi_stage_7.png',dpi=300)

def calc_circumcenter_3D(triangle_coord_array):
    '''Return circumcenter of triangle in 3D space according to http://gamedev.stackexchange.com/a/60631'''
    a = triangle_coord_array[0, ...]
    b = triangle_coord_array[1, ...]
    c = triangle_coord_array[2, ...]
    cross_ba_ca = numpy.cross((b-a),(c-a))
    circumcenter = a + ( (numpy.abs(c-a) ** 2) * numpy.cross(cross_ba_ca,(b-a)) + (numpy.abs(b-a) ** 2) * numpy.cross((c-a),cross_ba_ca) ) / (2 * numpy.abs(cross_ba_ca) ** 2)
    return circumcenter

def calc_circumcenter_circumsphere_tetrahedron_2(tetrahedron_coord_array):
    '''An alternative implementation based on http://mathworld.wolfram.com/Circumsphere.html because of issues with the initial implementation from the Berkeley page.'''
    determinant_array_first_column = numpy.reshape(numpy.array([numpy.square(vertex_array).sum() for vertex_array in tetrahedron_coord_array]), (4,1))
    determinant_array_final_column = numpy.ones((4,1))
    D_x = numpy.linalg.det(numpy.hstack((determinant_array_first_column, tetrahedron_coord_array[...,1:], determinant_array_final_column)))
    middle_column_array_D_y = numpy.hstack((numpy.reshape(tetrahedron_coord_array[...,0], (4,1)), numpy.reshape(tetrahedron_coord_array[...,2], (4,1))))
    D_y = - numpy.linalg.det(numpy.hstack((determinant_array_first_column, middle_column_array_D_y, determinant_array_final_column)))
    D_z = numpy.linalg.det(numpy.hstack((determinant_array_first_column, tetrahedron_coord_array[...,:-1], determinant_array_final_column)))
    a = numpy.linalg.det(numpy.hstack((tetrahedron_coord_array,determinant_array_final_column)))
    denominator = 2. * a
    x_0 = D_x / denominator
    y_0 = D_y / denominator
    z_0 = D_z / denominator
    circumcenter = numpy.array([x_0, y_0, z_0])
    return circumcenter

def calc_circumcenter_circumsphere_tetrahedron_vectorized(tetrahedron_coord_array):
    '''An alternative implementation based on http://mathworld.wolfram.com/Circumsphere.html because of issues with the initial implementation from the Berkeley page.
    Vectorized version for use with multiple tetrahedra in tetrahedron_coord_array -- the latter should have shape (N, 4, 3).'''
    num_tetrahedra = tetrahedron_coord_array.shape[0]
    #reshape the tetrahedron_coord_array to place all tetrahedra consecutively without nesting
    tetrahedron_coord_array = numpy.reshape(tetrahedron_coord_array, (tetrahedron_coord_array.shape[0] * tetrahedron_coord_array.shape[1], tetrahedron_coord_array.shape[2]))
    array_stacked_a_matrices = numpy.hstack((tetrahedron_coord_array, numpy.ones((num_tetrahedra * 4, 1))))
    first_column_array_determinant_arrays = tetrahedron_coord_array[...,0] ** 2 + tetrahedron_coord_array[...,1] ** 2 + tetrahedron_coord_array[...,2] ** 2
    first_column_array_determinant_arrays = first_column_array_determinant_arrays[:,numpy.newaxis]
    final_column_array_determinant_arrays = numpy.ones((first_column_array_determinant_arrays.shape[0],1))
    array_D_x_contents_before_determinant_calculation = numpy.hstack((first_column_array_determinant_arrays, tetrahedron_coord_array[...,1:],final_column_array_determinant_arrays))
    array_middle_column_arrays_D_y = numpy.hstack((numpy.reshape(tetrahedron_coord_array[...,0], (tetrahedron_coord_array.shape[0],1)), numpy.reshape(tetrahedron_coord_array[...,2], (tetrahedron_coord_array.shape[0],1))))
    array_D_y_contents_before_determinant_calculation = numpy.hstack((first_column_array_determinant_arrays, array_middle_column_arrays_D_y, final_column_array_determinant_arrays))
    array_D_z_contents_before_determinant_calculation = numpy.hstack((first_column_array_determinant_arrays, tetrahedron_coord_array[...,:-1],final_column_array_determinant_arrays))
    #split the arrays back to stacks of matrices
    array_D_x_contents_before_determinant_calculation = numpy.array(numpy.split(array_D_x_contents_before_determinant_calculation, num_tetrahedra))
    array_D_y_contents_before_determinant_calculation = numpy.array(numpy.split(array_D_y_contents_before_determinant_calculation, num_tetrahedra))
    array_D_z_contents_before_determinant_calculation = numpy.array(numpy.split(array_D_z_contents_before_determinant_calculation, num_tetrahedra))
    array_a_contents_before_determinant_calculation = numpy.array(numpy.split(array_stacked_a_matrices, num_tetrahedra))
    #compute the determinants for the stacks of matrices assembled above
    array_Dx_values = numpy.linalg.det(array_D_x_contents_before_determinant_calculation)
    array_Dy_values = - numpy.linalg.det(array_D_y_contents_before_determinant_calculation)
    array_Dz_values = numpy.linalg.det(array_D_z_contents_before_determinant_calculation)
    array_a_values = numpy.linalg.det(array_a_contents_before_determinant_calculation)
    array_denominator_values = 2. * array_a_values
    array_x0_values = array_Dx_values / array_denominator_values
    array_y0_values = array_Dy_values / array_denominator_values
    array_z0_values = array_Dz_values / array_denominator_values
    circumcenter_array = numpy.hstack((array_x0_values[:,numpy.newaxis], array_y0_values[:,numpy.newaxis], array_z0_values[:,numpy.newaxis]))
    return circumcenter_array
