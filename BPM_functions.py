# Beam Position Monitor (BPM) functions

# importing the necessary libraries
import sys
import cv2
import numpy as np
from math import *
from numpy import trapz
# from scipy.integrate import simps
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from tifffile import imsave
import threading
import imageio
import easygui
import os
from datetime import datetime
from skimage import measure
from skimage.measure import label, regionprops, regionprops_table
from skimage.draw import polygon2mask

from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt, QTimer

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
from matplotlib.widgets import PolygonSelector, Slider, Button 
import matplotlib.image as mpimg
# %matplotlib qt

# Functions for BPM analysis

def create_BPM_calibration(screen_width_mm, screen_length_mm):
    """
    Performs beam profile monitor (BPM) calibration by selecting a calibration image, 
    interactively defining screen corners, applying a projective transformation, 
    and calculating the pixel-to-mm conversion factor. The calibration data, 
    including image path, screen corners, and conversion factors, is saved to 
    a timestamped text file.

    Parameters
    ----------
    screen_width_mm : float
        Width of the screen in millimeters.
    screen_length_mm : float
        Length of the screen in millimeters.

    Returns
    -------
    calib_full_path : str
        Full path to the saved calibration data file.
    """  
    # Prompt the user to select the calibration image file
    calib_image_path = easygui.fileopenbox(msg="Select Calibration Image", filetypes=["*.tiff", "*.tif"])
    if not calib_image_path:
        print("No calibration image file selected. Exiting.")
    
    # Load the selected image
    calib_image_pre = imageio.imread(calib_image_path)
    calib_image = calib_image_pre.astype(np.float32) # Ensure the image data type is float32 for the openCV to work

    # Display the image using Matplotlib
    fig, ax = plt.subplots(2,1,tight_layout=True)
    ax[0].imshow(calib_image, origin='lower', cmap = cmap_CV_ibgyoR)
    # Interactive plot to select the corners of the screen on the image
    corners = proj_trans_interactive(calib_image)
    # Saving coordinates in seperate variables
    TL_x, TL_y = corners[0][0], corners[0][1]
    TR_x, TR_y = corners[1][0], corners[1][1]
    BR_x, BR_y = corners[2][0], corners[2][1]
    BL_x, BL_y = corners[3][0], corners[3][1] 

    square_flag = input("Is the screen a square? (y/n): ") # Ask the user to select if the screen is a square or not
    if square_flag == 'y':
        square_flag = True
    else:
        square_flag = False
        
    # Calculate the points for the projective transformation of each image assuming the screen is a square
    source_pts, destin_pts, maxWid, maxLen, v2h_conversion = input_for_proj_mat(corners[0], corners[1], corners[2], corners[3], 
                                                                                square_flag, screen_width_mm, screen_length_mm)

    # Calculate the projective transform matrix
    projmat = calc_perspective_transform_matrix(source_pts, destin_pts) 
    # Displaying the result of the projective transformation
    post_PT_calib_image = unwarp_perspective(calib_image, projmat, (maxWid, maxLen))
    ax[1].imshow(post_PT_calib_image, origin = 'lower', cmap = cmap_CV_ibgyoR)

    # Calculate the pixel to mm conversion factor save upto 3 decimal places
    px_to_mm = np.round(np.mean([screen_length_mm/maxLen, screen_width_mm/maxWid]), 3)

    # Print the directory of the calibration image
    print("Calibration image path: ", calib_image_path)

    # Metadeta for the calibration data
    current_time = datetime.now()
    timestamp = current_time.strftime("%Y%m%d_%H%M%S")
    current_path = calib_image_path # Get the path for the image used for calibration
    path_parts = current_path.split(os.sep)[7:] # Splitting the path and get everything but the first few folders of the path
    sub_path = "_".join(path_parts [:-1]) # Creating a single string with the sub-path and not the file name
    
    # Generate filename with timestamp and include the sub-path
    filename = f"BM_Data_{sub_path}_calibration_data_{timestamp}.txt"   
    base_directory = "C:\\Users\\chait\\Desktop\\LOA\\April_2024_run_related\\BM_Data\\Analysis_calib\\"
    calib_full_path = os.path.join(base_directory, filename)
    # Ensure the directory exists
    os.makedirs(os.path.dirname(calib_full_path), exist_ok=True)

    # Save the calibration data to a text file
    with open(calib_full_path, "w") as f:
        f.write(f"Calibration Image Path: {calib_image_path};\n\n")
        f.write(f"Corner_TL_x={TL_x};\n\n")
        f.write(f"Corner_TL_y={TL_y};\n\n")
        f.write(f"Corner_TR_x={TR_x};\n\n")
        f.write(f"Corner_TR_y={TR_y};\n\n")
        f.write(f"Corner_BR_x={BR_x};\n\n")
        f.write(f"Corner_BR_y={BR_y};\n\n")
        f.write(f"Corner_BL_x={BL_x};\n\n")
        f.write(f"Corner_BL_y={BL_y};\n\n")
        f.write(f"Average pixel to mm conversion ratio={px_to_mm};\n\n")
        f.write(f"Square_flag={square_flag};\n\n")
        if square_flag == False:
            f.write(f"Vertical to horizontal ratio={v2h_conversion};\n\n")
            
        f.write(f"Filename = {filename};\n\n")
        f.write(f"Sub-path = {sub_path};\n\n")
        f.write(f"calib_full_path = {calib_full_path};\n\n")

    # Display the path of the saved calibration data
    print("Calibration data saved successfully to: ", calib_full_path)
    return calib_full_path

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
    ax.imshow(img, cmap = 'prism')

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

def input_for_proj_mat(s_TL, s_TR, s_BR, s_BL, square_flag, screen_width_mm = 50, screen_length_mm = 50):
    """
    Calculate source and destination points for perspective transformation matrix.

    Parameters
    ----------
    s_TL, s_TR, s_BR, s_BL : lists
        Coordinates of the top-left, top-right, bottom-right, bottom-left points in the source image [x, y].
    square_flag : bool
        If True, enforces a square transformation by making the width and height equal.
    screen_width_mm : float
        The width of the screen selected in mm. Default is 50 mm.
    screen_length_mm : float
        The length of the screen selected in mm. Default is 50 mm.

    Returns
    -------
    source_pts, destin_pts, maxWid, maxLen, v2h_wanted : tuples
        A tuple containing two NumPy arrays representing the source and destination points.
        Additionally, the maximum width (maxWid) and maximum length (maxLen) of the perspective transformation.
        Also, the ratio of vertical to horizontal (v2h_wanted) if the screen is not a square.
    """
    # Calculating the distance between the points on a single side of the polygon 
    wid_1 = np.sqrt (((s_TL[0] - s_BL[0]) ** 2) + ((s_TL[1] - s_BL[1]) ** 2))
    wid_2 = np.sqrt (((s_TR[0] - s_BR[0]) ** 2) + ((s_TR[1] - s_BR[1]) ** 2))
    len_1 = np.sqrt (((s_TL[0] - s_TR[0]) ** 2) + ((s_TL[1] - s_TR[1]) ** 2))
    len_2 = np.sqrt (((s_BR[0] - s_BL[0]) ** 2) + ((s_BR[1] - s_BL[1]) ** 2))

    maxWid = max(int(wid_1), int(wid_2)) # Select the maximum width of the image
    maxLen = max(int(len_1), int(len_2)) # Select the maximum length of the image

    # fixing the v2h ratio
    v2h_wanted = 0 # default value 

    if (square_flag == True):
        maxLen = maxWid
        maxWid = maxLen
        print(maxLen, maxWid)
        v2h_wanted = 1
    else:
        v2h_wanted = np.round((screen_width_mm / screen_length_mm), 3)
        maxWid = int(maxLen*v2h_wanted)
        print("Length: ", maxLen, ", Width: ", maxWid, ", ratio wanted: ", v2h_wanted)
        
    source_pts = np.float32([s_TL, s_TR, s_BR, s_BL])                        
    destin_pts = np.float32([[0, 0], 
                             [maxLen - 1,0],
                             [maxLen - 1, maxWid - 1],
                             [0, maxWid - 1]
                            ])
    
    return (source_pts, destin_pts, maxWid, maxLen, v2h_wanted)

def calculate_contour_centroid(contour):
    """
    Calculate the geometric centroid by taking the mean of the rows and columns of the contour points

    Parameters
    ----------
    contour (np.ndarray): A numpy array of contour points, where each point is represented by its (row, column) coordinates.

    Returns
    -------
    tuple: A tuple (centroid_row, centroid_col) representing the centroid's row and column.
    """
    # Calculate the mean of the rows (y-coordinates) of the contour points
    centroid_row = np.mean(contour[:, 0])
    # Calculate the mean of the columns (x-coordinates) of the contour points
    centroid_col = np.mean(contour[:, 1])
    
    # Return the calculated centroid coordinates as a tuple
    return centroid_row, centroid_col

def image_to_BP(image_array, threshold_cutoff, projmat, maxWid, maxLen):
    """
    Acquires the beam profile from a provided image array, applies a projective 
    transformation, finds and returns the largest contour data.

    Parameters
    ----------
    image_array : numpy.ndarray
        The input image array.
    threshold_cutoff : float
        The threshold value (0 to 1) used to binarize the image.
    projmat : numpy.ndarray 
        The projective transformation matrix obtained from calibration.
    maxWid : int
        The maximum width of the perspective-transformed image.
    maxLen : int    
        The maximum length of the perspective-transformed image.

    Returns
    -------
    post_PT_beam_profile_img : numpy.ndarray
        The perspective-transformed image after applying the projective transformation.

    largest_contour : numpy.ndarray
        The largest contour found in the transformed image.
    """
    # Assume image_array is already a numpy array, no need to read from file
    beam_profile_img = image_array
    beam_profile_img = beam_profile_img.astype(np.float32)  # Ensure the image data type is float32 for the openCV to work

    # Apply the projective transformation   
    post_PT_beam_profile_img = unwarp_perspective(beam_profile_img, projmat, (maxWid, maxLen))

    # Apply threshold
    max_intensity = np.max(post_PT_beam_profile_img)
    threshold_value = threshold_cutoff * max_intensity
    threshold_value = max(threshold_value, 10)  # Ensure minimum threshold

    # Threshold the image to create a binary mask
    spot_mask = post_PT_beam_profile_img > threshold_value

    # Find contours in the binary mask
    contours = measure.find_contours(spot_mask, 0.5)
    largest_contour = max(contours, key=len).astype(np.int32)

    return post_PT_beam_profile_img, largest_contour

def update_moving_average(spot_storing_array, image, largest_contour, image_index):
    """
    Updates the moving average of the intensity values based on the contour of the current image.

    Parameters
    ----------
    spot_storing_array : numpy.ndarray
        A 3D array storing the pixel intensity and position values for beam spot in the largest contour.
    image : numpy.ndarray
        The current image for which the weighted sum is calculated.
    largest_contour : numpy.ndarray
        The largest contour found in the current image.
    image_index : int
        The index of the current image in the 3D array.
    
    Returns
    -------
    weighted_sum_positional : numpy.ndarray
        The updated 3D array with the intensity values weighted based on the contour of the current image.
    """
    # Update the weighted sum based on the contour of the current image
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if cv2.pointPolygonTest(largest_contour, (i, j), False) >= 0:
                spot_storing_array[image_index, i, j] = image[i, j]
    return spot_storing_array

def calculate_spot_size_variations(spot_shapes, average_weighted_spot, threshold_cutoff):
    """
    Calculate the spot size variations based on the average weighted spot.

    Parameters
    ----------
    spot_shapes : list
        A list of contours representing the spot shapes in the images.
    average_weighted_spot : numpy.ndarray
        The average weighted spot image.

    Returns
    -------
    numpy.ndarray
        An array containing the spot size variations for each spot shape
        as a percentage relative to the average weighted spot.
    """

    # Calculate spot size variations based on average weighted spot
    area_weighted_sum = np.sum(average_weighted_spot > (threshold_cutoff * np.max(average_weighted_spot)))
    spot_size_variations = []

    for contour in spot_shapes:
        mask = polygon2mask(average_weighted_spot.shape, contour)
        area_difference = np.sum(mask) - area_weighted_sum
        spot_size_variations.append(area_difference * 100 / area_weighted_sum)

    return np.array(spot_size_variations)

def plot_live(ax, canvas, spot_size_variations):
    """
    Plot the spot size variations on the provided matplotlib axis and update the canvas.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis object where the plot will be drawn.
    canvas : matplotlib.backends.backend_qt.FigureCanvasQTAgg
        The canvas where the plot is displayed (PyQt6).
    spot_size_variations : list or numpy.ndarray
        The data for spot size variations that will be plotted.
    """
    ax.clear()  # Clear previous plot
    ax.plot(spot_size_variations)  # Plot new data
    ax.set_title('Spot Size Variations')
    ax.set_xlabel('Shot Number')
    ax.set_ylabel('Percentage Spot Size Variation (%)')
    ax.grid(True)  # Enable grid
    canvas.draw()  # Update the canvas with the new plot

def plot_results(average_weighted_spot, spot_size_variations, maxWid, maxLen, save_dir):
    """
    Plot the average weighted spot and spot size variations.

    Parameters
    ----------
    average_weighted_spot : numpy.ndarray
        The average weighted spot image.
    spot_size_variations : numpy.ndarray    
        An array containing the spot size variations for each spot shape as a percentage relative to the average weighted spot.
    maxWid : int
        The maximum width of the perspective-transformed image.
    maxLen : int
        The maximum length of the perspective-transformed image.
    save_dir : str
        The directory path where the plots will be saved.
    """
    # Plot average weighted spot
    plt.figure(figsize=(10, 10))
    plt.imshow(average_weighted_spot, cmap='inferno', origin='lower', extent=[-maxLen/2, maxLen/2, -maxWid/2, maxWid/2])
    plt.colorbar(label='Intensity')
    plt.title('Averaged Beam Profile')
    plt.xlabel('Width (Pixels)')
    plt.ylabel('Length (Pixels)')
    plt.grid(True)
    save_path = os.path.join(save_dir, 'Averaged_Spot.svg')
    plt.savefig(save_path, format='svg', dpi=1200)
    plt.show()

    # Plot spot size variations
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(spot_size_variations)
    ax.set_title('Spot Size Variations')
    ax.set_xlabel('Shot Number')
    ax.set_ylabel('Percentage Spot Size Variation (%)')
    ax.grid(True)
    ax.axhline(np.mean(spot_size_variations), color='r', linestyle='dashed', linewidth=2, label='Mean')
    ax.legend()
    save_path = os.path.join(save_dir, 'Spot_Size_Variations.svg')
    plt.savefig(save_path, format='svg', dpi=1200)
    plt.show()




###### Secondary utilities ######

# Define color values and anchors
D1 = np.array([[1, 1, 1],       # White 
              [0, 1, 1],       # Cyan
              [0, 0, 1],       # Blue
              [0, 1, 0],       # Green
              [1, 1, 0],       # Yellow
              [1.0, 0.647, 0.0], # Orange
              [1, 0, 0],       # Red
              ])

F1 = np.array([0, 0.5, 0.6, 0.7, 0.8, 0.9, 1])  # Anchor points

# Create the colormap
cmap_CV_50 = LinearSegmentedColormap.from_list('my_cmap', list(zip(F1, D1)), N=256)

# Define color values and anchors
D2 = np.array([[0, 0, 0],       # Black 
              [1, 0, 0],       # Red
              [1.0, 0.647, 0.0], # Orange
              [1, 1, 0],       # Yellow
              [0, 1, 0],       # Green
              [0, 1, 1],       # Cyan
              [0, 0, 1],       # Blue
              [0.294, 0.0, 0.51], # Indigo
                 
              ])
                  
F2 = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.75, 1])  # Anchor points

# Create the colormap
cmap_CV_ibgyoR = LinearSegmentedColormap.from_list('my_cmap', list(zip(F2, D2)), N=256)

# Example usage in a plot
# x = np.linspace(0, 1, 256)
# y = np.sin(4 * np.pi * x)
# plt.plot(x, y, color='blue')
# plt.scatter(x, y, c=x, cmap=cmap_CV_ibgyoR)
# plt.colorbar(label='Custom Colormap')
# plt.show()