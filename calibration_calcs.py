# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import imageio
# from scipy.interpolate import interp1d
# import easygui
# import sys

# from Analysis_code_functions import proj_trans_interactive, input_for_proj_mat, calc_perspective_transform_matrix, unwarp_perspective

# # Input files from the particle tracking

# # Prompt the user to select the distance on screen to energy calibration file
# distance_energy_calibration_file = easygui.fileopenbox(msg="Select Distance on Screen to Energy Calibration File", filetypes=["*.txt"])

# if not distance_energy_calibration_file:
#     print("No distance on screen to energy calibration file selected. Exiting.")
#     sys.exit() 

# # Prompt the user to select the tracking distance file
# tracking_distance_file = easygui.fileopenbox(msg="Select Tracking Distance File", filetypes=["*.txt"])

# if not tracking_distance_file:
#     print("No tracking distance file selected. Exiting.")
#     sys.exit() 

# # Read the data from the ds_dE file
# calib_data = np.loadtxt(distance_energy_calibration_file)

# # Extract the first, second, and third columns
# energy = calib_data[:, 0]
# distance = -calib_data[:, 2]
# ds_dE = -calib_data[:, 1]

# # Calculate the derivative of energy with respect to distance
# dE_ds = np.gradient(energy, distance)

# # Create the interpolator for the charge conservation calculations
# dE_ds_interpolator = interp1d(distance, (dE_ds), kind = 'slinear') # !!! Try to make it smoother !!!

# # Read the data from the text file
# d_source = np.loadtxt(tracking_distance_file)

# # Estimating the average distance covered by electrons !!! Needs to be improved/finalized !!!
# average_distance_covered= np.mean(d_source)

# # Plot energy against distance and ds_dE
# plt.figure()
# plt.plot(ds_dE, energy, '.', color='green', label='ds_dE')
# plt.plot(distance, ds_dE, '.', color='green', label='ds_dE')
# plt.plot(distance, energy, '.', color='b', label='data')

# # Plot the calculated derivative on top
# plt.plot(distance, -1*dE_ds, '.', color='red', label='dE/ds calc')
# plt.plot(-1*dE_ds, energy, '.', color='pink', label='dE/ds calc')

# plt.xlabel('Distance on the Lanex Plane (mm)')
# plt.ylabel('Energy (MeV)')
# plt.title('Energy vs. Distance on the Lanex Plane with dE/dx')
# plt.grid(True)
# plt.legend()

# plt.show()

# # Prompt the user to select the calibration image file
# calib_image_path = easygui.fileopenbox(msg="Select Calibration Image", filetypes=["*.tiff", "*.tif"])

# if not calib_image_path:
#     print("No calibration image file selected. Exiting.")
  
# # Load the selected image
# calib_image = imageio.imread(calib_image_path)

# # Ensure the image data type is float32 for the openCV to work
# calib_image = calib_image.astype(np.float32)

# # Interactive plot to select the corners of the screen on the image
# corners = proj_trans_interactive(calib_image)

# # # In case the corner points are already known one can use the corners directly and not use the interactive selection
# # corners = [[   7.634,  533.971],
# #            [1389.681,  533.935],
# #            [1389.681,  334.201],
# #            [   6.906,  350.962]]

# # Calculate the points for the projective transformation of each image
# source_pts, destin_pts, maxWid, maxLen = input_for_proj_mat(corners[0], corners[1], corners[2], corners[3])

# # Calculate the projective transform matrix
# projmat = calc_perspective_transform_matrix(source_pts, destin_pts)

# # Displaying the result of the projective transformation
# post_PT_calib_image = unwarp_perspective(calib_image, projmat, (maxWid, maxLen))

# post_PT_calib_image = np.flipud(post_PT_calib_image)

# # Ask the user for the length of the lanex that is visible on the camera
# lanex_length_mm = float(input("Enter the length of the lanex that is visible in the camera (in mm): "))

# # ### Experiment specific parameter selection: ###
# # lanex_length_mm = 103 # for low energy
# # lanex_length_mm = 230 # for high energy

# # Conversion ratio !!! Still doesn't account for radial distortion !!!
# pixel2mm = np.round(lanex_length_mm/maxLen, 3) # We multiply each pixel value by this to convert it to mm

# # Ask the user for the distance between the lanex and the laser axis along the length of the screen
# screen_to_axis_distance_mm = float(input("The distance between the high-energy tip of the lanex and the laser axis along the length of the screen (in mm): "))

# # ### Experiment specific parameter selection: ###
# # screen_to_axis_distance_mm = 50.2 # Approx for high energy
# # screen_to_axis_distance_mm = 247.2 # Approx for low energy

# # Create an interpolation function for the energy distance conversion
# energy_interpolator = interp1d(distance, energy, kind='linear', fill_value='extrapolate')

# # Choose the number of energy values for each energy cut
# energy_partition = float(input("Enter the energy partition to be used in the deconvolution calculations: "))

# print("Calibration data saved successfully.")

import numpy as np
import imageio
import easygui
from Analysis_code_functions import proj_trans_interactive, input_for_proj_mat, calc_perspective_transform_matrix, unwarp_perspective
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def save_calibration_data():
    
    ##############################################
    ### Particle tracking related calibrations ###
    ##############################################
    
    # Prompt the user to select the distance on screen to energy calibration file
    distance_energy_calibration_file = easygui.fileopenbox(msg="Select Distance on Screen to Energy Calibration File", filetypes=["*.txt"])

    if not distance_energy_calibration_file:
        print("No distance on screen to energy calibration file selected. Exiting.")
        return
    
    # Prompt the user to select the tracking distance file
    tracking_distance_file = easygui.fileopenbox(msg="Select Tracking Distance File", filetypes=["*.txt"])

    if not tracking_distance_file:
        print("No tracking distance file selected. Exiting.")
        return
    
    # Read the data from the ds_dE file
    data = np.loadtxt(distance_energy_calibration_file)

    # Extract the first, second, and third columns
    energy = data[:, 0]
    distance = -data[:, 2]
    ds_dE = -data[:, 1]
    
    # Create an interpolation function for the energy distance conversion
    energy_interpolator = interp1d(distance, energy, kind='linear', fill_value='extrapolate')
    
    # Calculate the derivative of energy with respect to distance
    dE_ds = np.gradient(energy, distance)

    # Create the interpolator for the charge conservation calculations
    dE_ds_interpolator = interp1d(distance, (dE_ds), kind = 'slinear') # !!! Try to make it smoother !!!

    # Read the data from the d_source file
    d_source = np.loadtxt(tracking_distance_file)

    # Estimating the average distance covered by electrons !!! Needs to be improved/finalized !!!
    average_distance_covered= np.mean(d_source)

    # Choose the number of energy values for each energy cut
    energy_partition = 1.5  # MeV

    
    # Plot energy against distance and ds_dE
    plt.figure()
    plt.plot(ds_dE, energy, '.', color='green', label='ds_dE')
    plt.plot(distance, ds_dE, '.', color='green', label='ds_dE')
    plt.plot(distance, energy, '.', color='b', label='data')

    # Plot the calculated derivative on top
    plt.plot(distance, -1*dE_ds, '.', color='red', label='dE/ds calc')
    plt.plot(-1*dE_ds, energy, '.', color='pink', label='dE/ds calc')

    plt.xlabel('Distance on the Lanex Plane (mm)')
    plt.ylabel('Energy (MeV)')
    plt.title('Energy vs. Distance on the Lanex Plane with dE/dx')
    plt.grid(True)
    plt.legend()

    plt.show()
    
    ###############################################
    ### Image transformation based calibrations ###
    ###############################################
    
    # Prompt the user to select the calibration image file
    calib_image_path = easygui.fileopenbox(msg="Select Calibration Image", filetypes=["*.tiff", "*.tif"])

    if not calib_image_path:
        print("No calibration image file selected. Exiting.")
        return

    # Load the selected image
    calib_image = imageio.imread(calib_image_path)

    # Ensure the image data type is float32 for the openCV to work
    calib_image = calib_image.astype(np.float32)
    # plt.figure(1)
    # plt.imshow(calib_image, origin ='upper')
    # plt.show()
    # plt.pause(10)
    # plt.close()
    
    # Interactive plot to select the corners of the screen on the image
    corners = proj_trans_interactive(calib_image)

    # Saving coordinates in seperate variables
    TL_x, TL_y = corners[0][0], corners[0][1]
    TR_x, TR_y = corners[1][0], corners[1][1]
    BR_x, BR_y = corners[2][0], corners[2][1]
    BL_x, BL_y = corners[3][0], corners[3][1] 

    # Calculate the points for the projective transformation of each image
    source_pts, destin_pts, maxWid, maxLen = input_for_proj_mat(corners[0], corners[1], corners[2], corners[3])

    # Calculate the projective transform matrix
    projmat = calc_perspective_transform_matrix(source_pts, destin_pts)

    # Displaying the result of the projective transformation
    post_PT_calib_image = unwarp_perspective(calib_image, projmat, (maxWid, maxLen))
    
    # Flipping because we are simple creatures and can understand the image better?
    post_PT_calib_image = np.flipud(post_PT_calib_image)
    
    # Ask the user for the length of the lanex that is visible on the camera
    lanex_length_mm = float(input("Enter the length of the lanex that is visible in the camera (in mm): "))

    ### Experiment specific parameter selection: ###
    # lanex_length_mm = 103 # for low energy
    # lanex_length_mm = 230 # for high energy

    # Conversion ratio !!! Still doesn't account for radial distortion !!!
    pixel2mm = np.round(lanex_length_mm/maxLen, 3) # We multiply each pixel value by this to convert it to mm

    # Ask the user for the distance between the lanex and the laser axis along the length of the screen
    screen_to_axis_distance_mm = float(input("The distance between the high-energy tip of the lanex and the laser axis along the length of the screen (in mm): "))

    ### Experiment specific parameter selection: ###
    # screen_to_axis_distance_mm = 50.2 # Approx for high energy
    # screen_to_axis_distance_mm = 247.2 # Approx for low energy

    # Get the current date and time
    current_time = datetime.now()
    timestamp = current_time.strftime("%Y%m%d_%H%M%S")  # Format: YearMonthDay_HourMinuteSecond

    # Generate filename with timestamp
    filename = f"calibration_data_{timestamp}.txt"

    # Save calibration data to the file with timestamp in the filename
    with open(filename, 'w') as f:
        f.write(f"Calibration Image='{calib_image_path}';\n\n")
        f.write(f"Distance to Energy Calibration File='{distance_energy_calibration_file}';\n\n")
        f.write(f"Tracking Distance File='{tracking_distance_file}';\n\n")
        f.write(f"Average Distance Covered={average_distance_covered};\n\n")
        f.write(f"Corner_TL_x={TL_x};\n\n")
        f.write(f"Corner_TL_y={TL_y};\n\n")
        f.write(f"Corner_TR_x={TR_x};\n\n")
        f.write(f"Corner_TR_y={TR_y};\n\n")
        f.write(f"Corner_BR_x={BR_x};\n\n")
        f.write(f"Corner_BR_y={BR_y};\n\n")
        f.write(f"Corner_BL_x={BL_x};\n\n")
        f.write(f"Corner_BL_y={BL_y};\n\n")
        f.write(f"Average pixel to mm conversion ratio={pixel2mm};\n\n")
        f.write(f"The distance of the Lanex to the laser beam axis along the screen axis (mm)={screen_to_axis_distance_mm};\n\n")
        f.write(f"Energy partition used for background and divergence calculations (MeV)={energy_partition};\n\n")

    print("Calibration data saved successfully.")

    return maxWid, maxLen, projmat, pixel2mm, screen_to_axis_distance_mm, dE_ds_interpolator, average_distance_covered, energy_interpolator, energy_partition 

    

if __name__ == "__main__":
    # Save calibration data when the file is run directly
    maxWid, maxLen, projmat, pixel2mm, screen_to_axis_distance_mm, dE_ds_interpolator, average_distance_covered, energy_interpolator, energy_partition  = save_calibration_data()

