import sys
import os
import cv2
import numpy as np
import time
from math import *
from numpy import trapz

#from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
##from PyQt5.QtCore import Qt, QTimer

#from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib import cm
from scipy.interpolate import interp1d
from matplotlib.colors import LinearSegmentedColormap

#from pypylon import pylon
# from tifffile import imsave 

import threading
import imageio

from matplotlib.widgets import PolygonSelector, Slider, Button 

### Functions in use ###

def load_calibration_file(calibration_file_path): 
    """
    Load calibration data from a text file.
    
    Parameters
    ----------
    calibration_file_path : str
        The file path to the calibration file.
        
    Returns
    -------
    dict
        A dictionary containing the calibration data extracted from the file.
        Keys are parameter names, and values are their corresponding values.
        
    Note
    ----
    The calibration file should be formatted with each line containing a key-value pair
    separated by the '=' character. Numeric values are converted to float, and other values
    are kept as strings. Lines without the '=' character are skipped.
    """
    calibration_data = {}
    with open(calibration_file_path, 'r') as f:
        for line in f:
            # Skip lines that do not contain '=' character
            if '=' not in line:
                continue
            
            # Find the position of the first occurrence of '='
            separator_pos = line.find('=')
            
            # Extract key and value parts
            key = line[:separator_pos].strip()
            value = line[separator_pos+1:].strip().strip(";")  # Remove trailing semicolon
            
            # Convert numeric values to appropriate types
            if key.startswith("Average") or key.startswith("Width") or key.startswith("Length"):
                calibration_data[key] = float(value)
            
            elif key.startswith("Corner") or key.startswith("Average") or key.startswith("The distance") or key.startswith("Energy partition"):
                calibration_data[key] = float(value)
            
            elif key.startswith("Calibration Image") or key.startswith("Distance to Energy Calibration File") or key.startswith("Tracking Distance File"):
                calibration_data[key] = value.strip("'")
            
            else:
                calibration_data[key] = value

    return calibration_data

def proj_trans_interactive(img):

    
    """
    Opens an interactive plot for determining the projective transform matrix.

    This function displays an image and allows the user to interactively select 
    a region of interest by creating a 4-sided polygon. The order of points should 
    be Top Left (TL), Top Right (TR), Bottom Right (BR), Bottom Left (BL). The 
    selected coordinates are then rounded and returned.

    Parameters
    ----------
    img : numpy.ndarray
        Input image for perspective transformation.

    Returns
    -------
    numpy.ndarray
        Array of rounded screen coordinates (TL, TR, BR, BL).
    """

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.imshow(img, origin='lower', cmap= 'prism')
    # Adding colorbar to the ax object
    cb = plt.colorbar(ax.imshow(img, origin='lower', cmap= 'flag'), ax=ax)
    cb.set_label('Intensity')
    ax.set_title('Select 4-sided polygon defining screen corners')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True)

    # Create a list to store the coordinates
    screen_coords = []

    def on_polygon_end(verts):
        nonlocal screen_coords
        screen_coords = verts

    # Create a polygon selector for the figure to allow the user to interactively select the region of interest
    p_selector = PolygonSelector(ax, on_polygon_end)

    print("\n\nClick on the figure to create a 4-sided polygon defining the corners of the screen.")
    print("Please make sure that the order of points is Top Left (TL), Top Right (TR), Bottom Right (BR), Bottom Left (BL)")
    print("Press the 'esc' key to start a new polygon.")
    print("Try holding the 'shift' key to move all of the vertices.")
    print("Close the window when done.")

    plt.show()  # Allow the user to interact with the plot

    # Wait until the user closes the window
    while plt.fignum_exists(fig.number):
        plt.pause(1)

    return np.round(screen_coords, 3)

def calc_perspective_transform_matrix(src_points,dst_points):
    """
    Create the perspective transform matirx required to unwarp an image
    
    Given a set of source points on the raw image and a set of destination 
    points where the source points should be mapped to in real space, one can
    use opencv to calculate the corresponding perspective transform matrix

    Parameters
    ----------
    src_points : numpy array of tuples of floats
        The source points are listed in a numpy array where each point consists
        of a tuple of two floats corresponding to the x and y corrdinates of a point
        These are the coordinates of the points in the unwarps image
    dst_points : numpy array of tuples of floats
        The corresponding coordinates where the source points should be mapped to.
        Same format as src_points.

    Returns
    -------
     projective_matrix : matrix of floats
        Corresponding to the relevant projective transform matrix.
    """
    projective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    return projective_matrix

def unwarp_perspective(img_in, projective_matrix, newShape):
    """
    Applies perspective transformation to unwarp an image.

    parameters
    ----------
    img_in : numpy.ndarray
        The input image to be unwrapped.
        
    projective_matrix : numpy.ndarray
        The transformation matrix obtained from calibration.
    
    newShape : tuple 
        The desired shape of the output image after unwrapping in the format (rows, cols).

    Returns
    -------
    img_output : numpy.ndarray
        The unwrapped image after applying the perspective transformation.
        
    Notes
    -----
        - The projective_matrix is obtained from calibration and defines the perspective transformation to be applied.
        - The newShape parameter determines the desired shape of the output image after unwrapping.
        - The function uses OpenCV's 'warpPerspective' method to perform the perspective transformation.
        - The resulting unwrapped image is cropped to match the specified newShape.

    """
    rows,cols = img_in.shape
    rowsNew,colsNew = newShape
    
    img_output = cv2.warpPerspective(img_in, projective_matrix, (cols,rows))
    img_output = img_output[:rowsNew,:colsNew]

    return (img_output)

def input_for_proj_mat(s_TL, s_TR, s_BR, s_BL):
    """
    Calculate source and destination points for perspective transformation matrix.

    Parameters
    ----------
    s_TL : list
        Coordinates of the top-left point in the source image [x, y].
    s_TR : list
        Coordinates of the top-right point in the source image [x, y].
    s_BR : list
        Coordinates of the bottom-right point in the source image [x, y].
    s_BL : list
        Coordinates of the bottom-left point in the source image [x, y].

    Returns
    -------
    source_pts, destin_pts : tuples
        A tuple containing two NumPy arrays representing the source and destination points.
    """
    # Calculating the distance between the points making up the parallel lines 

    wid_1 = np.sqrt (((s_TL[0] - s_BL[0]) ** 2) + ((s_TL[1] - s_BL[1]) ** 2))
    wid_2 = np.sqrt (((s_TR[0] - s_BR[0]) ** 2) + ((s_TR[1] - s_BR[1]) ** 2))
    maxWid = max(int(wid_1), int(wid_2))

    len_1 = np.sqrt (((s_TL[0] - s_TR[0]) ** 2) + ((s_TL[1] - s_TR[1]) ** 2))
    len_2 = np.sqrt (((s_BR[0] - s_BL[0]) ** 2) + ((s_BR[1] - s_BL[1]) ** 2))
    maxLen = max(int(len_1), int(len_2))

    source_pts = np.float32([s_TL, s_TR, s_BR, s_BL])                        
    destin_pts = np.float32([[0, 0], 
                             [maxLen - 1,0],
                             [maxLen - 1, maxWid - 1],
                             [0, maxWid - 1]
                            ])
    
    return (source_pts, destin_pts, maxWid, maxLen)

def create_energy_image(x_axis, y_axis, intensity_data, cmap='viridis', vmin=None, vmax=None):
    """
    Create an image using imshow given x_axis, y_axis, and intensity_data.

    Parameters
    ----------
        x_axis (array): 1D array representing the X-axis values.
        y_axis (array): 1D array representing the Y-axis values.
        intensity_data (array): 2D array representing the intensity values.
        cmap (str): Colormap (default is 'viridis').
        vmin (float): Minimum value for color normalization (default is None).
        vmax (float): Maximum value for color normalization (default is None).

    Returns 
    -------
        None (displays the plot)
    
    Example-usage
    -------------
        create_energy_image(x_axis, y_axis, spectrum_img_corr)
    """
    plt.imshow(intensity_data, extent=(x_axis[0], x_axis[-1], y_axis[0], y_axis[-1]),
               aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    plt.xlabel('Energy (MeV)')
    plt.ylabel('Vertical Position (pixels)')
    plt.title('Post Perspective Transformation Image - Energy Coordinates')
    plt.colorbar()
    
def cuts_and_BG(post_PT_image, screen_energy_axis, x_axis, y_axis, pixel2mm, average_distance_covered, fine_cut = False, bg_sub = True,
                energy_partition = 5.0, selection_area_percentage = 0.68):
    """
    Cuts energy spectra in slices and performs background subtraction.

    This function divides the energy axis into sections based on the provided energy_partition. 
    For each energy section, it identifies the average intensity peak and its surrounding area.
    Background values are then subtracted from the corresponding columns in the post_PT_image. 
    Negative values and values outside the identified area are set to zero.

    Parameters
    ----------
    post_PT_image : numpy.ndarray
        The post-perspective-transform image.

    screen_energy_axis : numpy.ndarray
        Energy values corresponding to the screen coordinates.

    x_axis : numpy.ndarray
        Energy values.

    y_axis : numpy.ndarray
        Vertical position values.

    pixel2mm : float
        Conversion factor from pixels to millimeters.

    average_distance_covered : float
        (Temporary) Average distance parameter used for divergence calculations   
    
    fine_cut : bool, optional
        If True, performs a finer cut (symmetric around max intensity). Defaults to False.
        
    bg_sub : bool, optional
        If True, performs background subtraction. Defaults to True.

    energy_partition : float, optional
        Energy partition value for dividing the energy axis into sections. Defaults to 5.0 (MeV).

    selection_area_percentage : float, optional
        Percentage of the area to select around the intensity peak. Defaults to 0.68 (1 sigma).

    Returns
    -------
    post_bg_sub_image : numpy.ndarray
        Post-background-subtraction image.

    divergence_vals : list
        Estimated divergence values for each energy section.

    energy_section_centres : list
        Energy section centres corresponding to divergence values.

    Note
    ----
    The input post_PT_image is modified in-place.
    """
    # Working on a copy of the image received
    PT_image = post_PT_image
    
    # Normalising the colormap to the energy values on the screen
    norm_cmap = Normalize(vmin=min(screen_energy_axis), vmax=max(screen_energy_axis))
    cmap = cm.get_cmap('viridis')
    
    # Create a ScalarMappable to map energy values to colors
    sm = ScalarMappable(cmap=cmap, norm=norm_cmap)
    
    # Need to set an array for the ScalarMappable
    sm.set_array([])
    
    # Calculate the number of sections
    num_sections = int(np.ceil((max(screen_energy_axis) - min(screen_energy_axis)) / energy_partition))
    
    # Divide the energy axis into sections
    energy_sections = np.linspace(min(screen_energy_axis), max(screen_energy_axis), num_sections + 1)
    
    # Create a variable to store the values of the pixels that represent the most bright sections on the image (energy cuts)
    intensity_distribution_centers_y = []
    
    # Variable to store the estimated energy-section-wise divergence
    divergence_vals, energy_section_centres = [], []
    
    # Iterate through energy sections in the PT_image
    for i in range(1, len(energy_sections)):

        # Select the bounds of the current energy section of the loop
        lower_bound = energy_sections[i-1]
        upper_bound = energy_sections[i]
        energy_section_centres.append(np.round(0.5*(lower_bound+upper_bound), 2))
        
        # Find indices within the current energy section
        section_indices = np.where((x_axis >= lower_bound) & (x_axis <= upper_bound))[0]
        
        # Sum along the y-axis for the current energy section
        intensity_distribution_section = np.sum(PT_image[:, section_indices], axis=1)

        # Normalizing the intensity distribution per section w.r.t. the number of pixel columns in that image (energy) section
        norm_int_dist_section = intensity_distribution_section/len(section_indices)
        
        # Find the index of the maximum value in intensity_distribution_section
        max_index = np.argmax(norm_int_dist_section)
        
        # Get the corresponding value from the y_axis array
        max_y_value = y_axis[max_index]
        intensity_distribution_centers_y.append(max_y_value)
    
        # Finding the points in y_axis that correspond to selection_area_percentage endpoints (symmetrically) w.r.t. max_index
        # Calculate the cumulative sum of the normalized intensity distribution
        cumulative_sum = np.cumsum(norm_int_dist_section)
        
        # Normalize the cumulative sum to get values between 0 and 1
        normalized_cumulative_sum = cumulative_sum / cumulative_sum[-1]
        
        # Find the indices where the normalized cumulative sum crosses the selection_area_percentage
        lower_index = np.argmax(normalized_cumulative_sum >= (1 - selection_area_percentage) / 2)
        upper_index = np.argmax(normalized_cumulative_sum >= 1 - (1 - selection_area_percentage) / 2)
        
        if (fine_cut == True):
            # Take the index which is closer to max_index and use that for cutoffs
            roi_index_y = np.min([np.abs(max_index-lower_index), np.abs(max_index-upper_index)])

            # Update the lower and upper indices
            lower_index = max_index - roi_index_y
            upper_index = max_index + roi_index_y

            # Indexing errors can occur for finer cuts so we set them to end values after a check 
            if lower_index < 0:
                lower_index = 0

            if upper_index >= (len(norm_int_dist_section)):
                upper_index = len(norm_int_dist_section) - 1
        
        # Get the corresponding values from the y_axis array
        lower_y_value = y_axis[lower_index]
        upper_y_value = y_axis[upper_index]
        
        # Use the y_values to estimate the spread of the beam in the non-dispersive axis
        non_dispersive_spread_mm = (upper_y_value - lower_y_value)*pixel2mm

        # Estimating the divergence 
        divergence_vals.append(non_dispersive_spread_mm/average_distance_covered)
        
        # Calculating the background from the respective energy sections
        bg_val = np.min([norm_int_dist_section[lower_index], norm_int_dist_section[upper_index]])
        
        # Check if the user wants to subtract background
        if (bg_sub == True):
            # Subtracting background value from each energy section (columns) and converting all negative values to zero
            PT_image[:, section_indices] -= bg_val
            
            # Suggestion for later: Since the slices of energy that you look at are often showing artefacts of folding
            #                       but are just results of cumulative summing up, maybe an edge to edge linear fit 
            #                       could dictate how the bg_val is subtracted

            # Convert negative values to zero
            PT_image[PT_image < 0] = 0

            # Convert values outside cutoffs to zero
            PT_image[:lower_index, section_indices] = 0
            PT_image[upper_index:, section_indices] = 0
        
    post_bg_sub_image = PT_image    
    return(post_bg_sub_image, divergence_vals, energy_section_centres)

def process_image(img, maxWid, maxLen, projmat, pixel2mm, 
                  screen_to_axis_distance, dE_ds_interpolator, 
                  average_distance_covered, energy_interpolator,
                  energy_partition, ROI_selection_percent = 0.68, fine_cut_flag = False, 
                  bg_flag = True, PT_flag = True, norm_flag = False, plot_flag = True, cam_flag = 'H'):
    """
    Process the input image to analyze energy spectrum.

    Parameters
    ----------
    - img : numpy.ndarray
        Input image.
    
    - maxWid : int
        Maximum width for perspective transformation.
    
    - maxLen : int
        Maximum length for perspective transformation.
    
    - projmat : numpy.ndarray
        Perspective transformation matrix.
    
    - pixel2mm : float
        Conversion factor from pixels to millimeters.
    
    - screen_to_axis_distance : float
        Distance from screen to beam axis in millimeters.

    - dE_ds_interpolator : scipy.interpolate._interpolate.interp1d
        An interpolation function for the dE_ds from distance conversion.

    - average_distance_covered : float
        Average distance covered by all the trajectories simulated in the particle tracking from the point of creation to the screen
    
    - energy_interpolator : scipy.interpolate._interpolate.interp1d
        An interpolation function for the energy distance conversion.
    
    - energy_partition : float
        The size of the energy range sections on the image for processing intensity distributions along the dispersive axis.
    
    - ROI_selection_percent : float, optional
        Percentage of the area to select around the intensity peak for region of interest (ROI) selection.
        Defaults to 0.68 (68%).
    
    - fine_cut_flag : bool, optional
        If True, performs finer (symmetric around peak and smaller) cuts for background subtraction. Default is False.

    - bg_flag : bool, optional
        If True, performs background subtraction during energy section-wise processing. Default is True.

    - PT_flag : bool, optional
        If True, performs perspective transformation on the input image. Default is True.

    - norm_flag : bool, optional
        If True, normalizes the intensity distribution in energy coordinates and saves the distribution in a text file.
        Default is False.

    - plot_flag : bool, optional
        If True, shows all the plots generated in the processsing. Default is True.  

    - cam_flag : string
        If H (Default), performs the processsing according to the HE calibration values. 
        If L, performs the processsing according to the LE calibration values. 

    Returns
    -------
    - None

    This function performs the following steps:
    1. Unwarps the input image using a perspective transformation and then adds the screen axis coordinates to the image.
    2. Computes the intensity distribution for screen coordinates.
    3. Converts screen coordinates to energy coordinates using the interpolation function from the calibration data.
    4. Uses the cuts_and_BG function to locate the ROI in the image and performs background subtraction (energy-section wise).
    5. The function also returns values of divergence calculated in those energy-sections.
    6. Divides each pixel in a column of the background-subtracted image with the respective dE_ds values.
    7. Plots the processed image and intensity distribution in energy coordinates, along with the estimated divergence variation.

    Note: Modify the function to suit specific requirements and input data.

    """
    # Reading the images one by one
    trial_image = img
    
    # Ensure the image data type is float32 for the openCV to work
    trial_image = trial_image.astype(np.float32)
    
    # Unwarping the image according to the projective transformation 
    if (PT_flag == True):
        post_PT_image = unwarp_perspective(trial_image, projmat, (maxWid, maxLen))
        
    else:
        post_PT_image = trial_image

    # Storing the new shape
    new_shape = post_PT_image.shape
    electron_spectrum_image_size = [new_shape[0] * pixel2mm, new_shape[1] * pixel2mm]
    
    # Creating an axis that shall contain the distances of the points of the screen from the beam axis in mm
    screen_distance_axis_mm = np.linspace(screen_to_axis_distance, (electron_spectrum_image_size[1] + screen_to_axis_distance), new_shape[1])
    # print(max(screen_distance_axis_mm), min(screen_distance_axis_mm))

    # Compute the intensity distribution for screen coordinates
    intensity_distribution_screen_mm = np.sum(post_PT_image, axis=0)
    
    # Create a 1D array for Y-axis (vertical position)
    y_axis = np.arange(post_PT_image.shape[0])
    
    # Convert mm coordinates to energy coordinates using the interpolation function
    screen_energy_axis = energy_interpolator(screen_distance_axis_mm)

    if plot_flag:    
        fig, ax = plt.subplots(3, 1, figsize = (12, 8))
        # Plot the image
        image_obj = ax[0].imshow(post_PT_image, 
                                extent=(screen_distance_axis_mm[0], screen_distance_axis_mm[-1], 
                                        0, post_PT_image.shape[0]*pixel2mm),
                                        aspect='auto', cmap='viridis', 
                                        vmin=0)#, vmax=16384)
        ax[0].set_xlabel('Distance (mm)')
        ax[0].set_ylabel('Vertical Position (mm)')
        ax[0].set_title('Post Perspective Transformation Image - Screen Coordinates')
        # Create colorbar for the image
        cb1 = fig.colorbar(image_obj, ax=ax[0], orientation='vertical', fraction=0.05, pad=0.02)
        cb1.set_label('Intensity')

    # Creating a new variable for the non_analysed_spectra
    non_analysed_spectra = post_PT_image
    
    # Create a 1D array for X-axis (energy)
    x_axis = screen_energy_axis
    
    # Create a meshgrid for X and Y
    X, Y = np.meshgrid(x_axis, y_axis)
    
    # Finding the vertical cuts in the image and performing the background subtraction     
    # Select a percentage of this area to select the edge points = 1 sigma; !!! Approx and needs to be finalized/changed !!!
    post_bg_sub_image, divergence_val_array, energy_section_array = cuts_and_BG(non_analysed_spectra, screen_energy_axis, 
                                                                                x_axis = x_axis, y_axis = y_axis, 
                                                                                pixel2mm = pixel2mm, 
                                                                                average_distance_covered = average_distance_covered, 
                                                                                fine_cut = fine_cut_flag,
                                                                                bg_sub = bg_flag, 
                                                                                energy_partition = energy_partition, 
                                                                                selection_area_percentage = ROI_selection_percent)

    # Compute the intensity distribution for mm coordinate BG removal image
    intensity_distribution_screen_nbg = np.sum(post_bg_sub_image, axis = 0)
    
    # Interpolate ds_dE values for each screen_distance_axis_mm value according to the particle tracking data
    dE_ds_axis = dE_ds_interpolator(screen_distance_axis_mm) *-1 # !!! Multiplying by -1 because the values were negative !!!
        
    # Divide each pixel in a column of post_bg_sub_image with the respective dE_ds values to conserve charge on screen
    spectrum_img_corr = post_bg_sub_image/dE_ds_axis
    
    # Compute the intensity distribution for energy coordinates
    intensity_distribution_energy = np.sum(spectrum_img_corr, axis=0)

    # Area under the curve in a given energy range
    area_roi = np.round(trapz(intensity_distribution_energy[(x_axis>=70) & (x_axis<=150)]), 3)
    print("Area under the curve in the specified range: {:.2e}".format(area_roi))

    if plot_flag:
        # Estimation of divergence variation needs to be plotted on the axis with the intensity variation    
        image_obj = ax[1].pcolormesh(X, Y, spectrum_img_corr, shading='auto', cmap='cool', 
                                    vmin=0, vmax = 16384)
        ax[1].set_xlabel('Energy (MeV)')
        ax[1].set_ylabel('Vertical Position (pixels)')
        ax[1].set_title('Energy Spectrum Measurements')
        # Create colorbar for the image
        cb1 = fig.colorbar(image_obj, ax=ax[1], orientation='vertical', fraction=0.04, pad=0.02)
        cb1.set_label('Intensity')
    
    # Adding a normailzation to the intensity distribution if wanted. Will also save the distribution in a
    if (norm_flag == True):
        
        # Dividing with the maximum intensity on the deconvoluted spectrum
        intensity_distribution_energy /= np.max(intensity_distribution_energy)        
       
        # !!! I suppressed this so as to work faster with normalized plots. Uncomment if you want to save the data in a txt file !!!
        # Ask the user for the filename
        # file_name = input("Enter the name of the txt file (including extension): ")

        # # Stack x_axis and intensity_distribution_energy horizontally
        # data_to_save = np.column_stack((x_axis, intensity_distribution_energy))

        # # Save the values to the user-specified txt file
        # np.savetxt(file_name, data_to_save, delimiter='\t')
    
    if plot_flag:
        # Plotting the intensity distribution on the lanex screen w.r.t. energy coordinates
        ax[2].plot(x_axis, intensity_distribution_energy)
        ax[2].set_ylim(0, 50000) # Setting the upper limit of the y axis to 1.5e5
        ax[2].set_xlabel('Energy (MeV)')
        ax[2].set_ylabel('Intensity')
        ax[2].set_title("Area under the curve in the specified range: {:.2e}".format(area_roi))
        ax[2].grid(True)
        # Create a twin Axes sharing the xaxis
        ax2 = ax[2].twinx() 
        # Plot the divergence values on the right axis
        ax2.plot(energy_section_array, divergence_val_array, label='Divergence', color='red', alpha = 0.3)
        ax2.set_ylabel('Divergence (rad)')  # Set y-axis label for the right axis
        ax2.legend()
        
        plt.axvline(x=70, color= 'r', lw = 2)
        plt.axvline(x=150, color= 'r', lw = 2)
        # Adjust layout to prevent overlapping
        fig.tight_layout()

        # Showing the plots
        plt.show()

        # Display the plot for 10 seconds
        plt.pause(4.5)

        # Close the plot
        plt.close()

    return intensity_distribution_energy, x_axis, intensity_distribution_screen_mm, screen_distance_axis_mm
    
def txt_to_img(txt_file_path, flip_horizontal=False, flip_vertical=False, skip_lines=0, show_image=True, add_colorbar=False, save_as_tiff=False):
    """
    Read image data from a text (.txt) file, visualize it using matplotlib, and optionally save it as a TIFF file.

    Parameters
    ----------
    txt_file_path : str
        The path to the text file containing the image data.

    flip_horizontal : bool, optional
        If True, flip the image horizontally. Defaults to False.

    flip_vertical : bool, optional
        If True, flip the image vertically. Defaults to False.
    
    skip_lines : int, optional
        The number of lines to skip at the beginning of the text file. Defaults to 0.
         
    show_image : bool, optional
        If True, display the plotted image. Defaults to True.
        
    add_colorbar : bool, optional
        If True and show_image is True, add a colorbar to the plotted image. Defaults to False.

    save_as_tiff : bool, optional
        If True, save the image data as a TIFF file. Defaults to False.

    Returns
    -------
    image_data : numpy.ndarray or None
        The image data read from the TXT file as a NumPy array. Returns None if an error occurs during the process.

    Raises
    ------
    Exception
        Raises an exception if there is an error during file reading or data conversion.

    Notes
    -----
    - The text file should contain numerical data separated by spaces or tabs.
    - Floating-point numbers are supported for pixel values.
    """
    try:
        # Read the image data from the txt file
        with open(txt_file_path, 'r') as file:
            lines = file.readlines()
            if skip_lines > 0:
                lines = lines[skip_lines:]

        # Convert the text data to a NumPy array (handling floating-point numbers)
        image_data = np.array([[int(float(pixel)) for pixel in line.split()] for line in lines])

        # Flip the image horizontally if specified
        if flip_horizontal:
            image_data = np.fliplr(image_data)
        
        # Flip the image vertically if specified
        if flip_vertical:
            image_data = np.flipud(image_data)

        # Plot the image using matplotlib
        if show_image:
            plt.imshow(image_data, cmap='viridis')
            
            # Calculate the mix and min pixel values that are present in the image
            max_pixel = np.max(image_data)
            min_pixel = np.min(image_data)

            plt.title(f'Image from {os.path.basename(txt_file_path)}\nMax pixel: {max_pixel}, Min pixel: {min_pixel}')


            font = {'family': 'serif',
                    'color':  'white',
                    'weight': 'normal',
                    'size': 8,
                    }
            
            if add_colorbar:
                plt.colorbar()

            #plt.text(len(image_data)*0.475, 20, 'peak count {np.max(image_data)}', fontdict=font)
            plt.show()

        # Save the image data as a TIFF file if specified
        if save_as_tiff:
            tiff_file_path = os.path.splitext(txt_file_path)[0] + '.tiff'
            imageio.imwrite(tiff_file_path, image_data, format='TIFF')
            print(f"Saved TIFF file: {tiff_file_path}")

        # Return the image data
        return image_data

    except Exception as e:
        print(f"Error: {e}")
        return None

### Secondary tools ###

# Define color values and anchors
D = np.array([[1, 1, 1],       # White 
              [0, 1, 1],       # Cyan
              [0, 0, 1],       # Blue
              [0, 1, 0],       # Green
              [1, 1, 0],       # Yellow
              [1.0, 0.647, 0.0], # Orange
              [1, 0, 0],       # Red
              ])
                  
F = np.array([0, 0.5, 0.6, 0.7, 0.8, 0.9, 1])  # Anchor points

# Create the colormap
cmap_CV_50 = LinearSegmentedColormap.from_list('my_cmap', list(zip(F, D)), N=256)

# Example usage in a plot
# x = np.linspace(0, 1, 256)
# y = np.sin(4 * np.pi * x)
# plt.figure(1)
# plt.plot(x, y, color='blue')
# plt.scatter(x, y, c=x, cmap=cmap_CV_50)
# plt.colorbar(label='cmap_CV_50')
# plt.show()

# Define color values and anchors
D = np.array([[1, 1, 1],       # White 
              [1, 0, 0],       # Red
              [1.0, 0.647, 0.0], # Orange
              [1, 1, 0],       # Yellow
              [0, 1, 0],       # Green
              [0, 1, 1],       # Cyan
              [0, 0, 1],       # Blue
              [0.294, 0.0, 0.51], # Indigo
                 
              ])
                  
F = np.array([0, 0.6, 0.8, 0.84, 0.88, 0.92, 0.96, 1])  # Anchor points

# Create the colormap
cmap_CV_ibgyoR = LinearSegmentedColormap.from_list('my_cmap', list(zip(F, D)), N=256)

# # Example usage in a plot
# x = np.linspace(0, 1, 256)
# y = np.sin(4 * np.pi * x)
# plt.figure(2)
# plt.plot(x, y, color='blue')
# plt.scatter(x, y, c=x, cmap=cmap_CV_ibgyoR)
# plt.colorbar(label='Custom Colormap')
# plt.show()

