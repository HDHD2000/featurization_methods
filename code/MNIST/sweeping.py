import numpy as np
import gudhi as gd 


def sweep_right_to_left_filtration(data):
    num_data_points = data.shape[0]
    num_pixels = data.shape[1]
    num_x_pixels = np.sqrt(num_pixels)
    filt_vals = np.zeros((num_pixels,))
    filt_func_vals_data = np.zeros((num_data_points, num_pixels))
    for j in range(num_pixels):
        filt_vals[j] = int(j/num_x_pixels + 1)
    for i in range(num_data_points):
        filt_func_vals_data[i] = filt_vals
    return filt_func_vals_data   
    
def sweep_left_to_right_filtration(data):
    num_data_points = data.shape[0]
    num_pixels = data.shape[1]
    num_x_pixels = np.sqrt(num_pixels)
    filt_vals = np.zeros((num_pixels,))
    filt_func_vals_data = np.zeros((num_data_points, num_pixels))
    for j in range(num_pixels):
        filt_vals[j] = num_x_pixels - int(j/num_x_pixels + 1)
    for i in range(num_data_points):
        filt_func_vals_data[i] = filt_vals
    return filt_func_vals_data

def sweep_up_down_filtration(data):
    num_data_points = data.shape[0]
    num_pixels = data.shape[1]
    num_x_pixels = np.sqrt(num_pixels)
    filt_vals = np.zeros((num_pixels,))
    filt_func_vals_data = np.zeros((num_data_points, num_pixels))
    for j in range(num_pixels):
        filt_vals[j] = j % num_x_pixels
    for i in range(num_data_points):
        filt_func_vals_data[i] = filt_vals
    return filt_func_vals_data

def sweep_down_up_filtration(data):
    num_data_points = data.shape[0]
    num_pixels = data.shape[1]
    num_x_pixels = np.sqrt(num_pixels)
    filt_vals = np.zeros((num_pixels,))
    filt_func_vals_data = np.zeros((num_data_points, num_pixels))
    for j in range(num_pixels):
        filt_vals[j] = (num_x_pixels - j - 1) % num_x_pixels
    for i in range(num_data_points):
        filt_func_vals_data[i] = filt_vals
    return filt_func_vals_data

def build_point_cloud(data_point, threshold_grsc_perc):  
    num_x_pixels = np.sqrt(data_point.shape[0]).astype(int)
    num_y_pixels = num_x_pixels     
    image = data_point.reshape((num_x_pixels, num_y_pixels))    
    binary_image = image >= threshold_grsc_perc * np.max(image)          
    num_black_pixels = np.sum(binary_image) 
    point_cloud = np.zeros((num_black_pixels, 2))
    point = 0        
    for i in range(num_x_pixels):
        for j in range(num_y_pixels):
            if binary_image[i, j] > 0:
                point_cloud[point, 0] = j
                point_cloud[point, 1] = num_y_pixels - i
                point = point + 1        
    return point_cloud

##------------------------------------------------------##

def pers_intervals_across_homdims(filt_func_vals_data, data = [], threshold_grsc_perc = 0.5):
    pers_intervals_homdim0_data = []
    pers_intervals_homdim1_data = []   
    
    for i, filt_func_vals_data_point in enumerate(filt_func_vals_data):
        num_x_pixels = np.sqrt(filt_func_vals_data.shape[1]).astype(int) 
        num_y_pixels = num_x_pixels
        simplicial_complex = gd.CubicalComplex(dimensions = [num_x_pixels, num_y_pixels], 
                                                   top_dimensional_cells = filt_func_vals_data_point) 
        simplex_tree = simplicial_complex
        
        homdims_pers_intervals = simplex_tree.persistence()          
        pers_intervals_homdim0 = simplex_tree.persistence_intervals_in_dimension(0)
        pers_intervals_homdim1 = simplex_tree.persistence_intervals_in_dimension(1)
        
        if(len(pers_intervals_homdim0) == 0):
                pers_intervals_homdim0 = np.asarray([[0, 0]])
        if(len(pers_intervals_homdim1) == 0):
                pers_intervals_homdim1 = np.asarray([[0, 0]])    
        
        pers_intervals_homdim0_data.append(pers_intervals_homdim0)
        pers_intervals_homdim1_data.append(pers_intervals_homdim1)

    return pers_intervals_homdim0_data, pers_intervals_homdim1_data