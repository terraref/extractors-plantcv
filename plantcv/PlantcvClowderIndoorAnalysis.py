# PlantCV analysis code package for TERRA-REF indoor system analysis

from __future__ import print_function
import plantcv as pcv
import numpy as np
import cv2


# Create a list of fields and a trait dictionary
def get_traits_table():
    # Compiled traits table
    fields = ('entity', 'cultivar', 'treatment', 'local_datetime', 'sv_area', 'tv_area', 'hull_area',
              'solidity', 'height', 'perimeter', 'access_level', 'species', 'site',
              'citation_author', 'citation_year', 'citation_title', 'method')
    traits = {'plant_barcode': '',
              'genotype': '',
              'treatment': '',
              'imagedate': '',
              'sv_area': [],
              'tv_area': '',
              'hull_area': [],
              'solidity': [],
              'height': [],
              'perimeter': [],
              'access_level': '2',
              'species': 'Sorghum bicolor',
              'site': 'Danforth Plant Science Center Bellweather Phenotyping Facility',
              'citation_author': '"Fahlgren, Noah"',
              'citation_year': '2016',
              'citation_title': 'Unpublished Data from Sorghum Test Runs',
              'method': 'PlantCV'}

    return fields, traits


# Compose the summary traits
def generate_traits_list(traits):
    trait_list = [traits['plant_barcode'],
                  traits['genotype'],
                  '"' + traits['treatment'] + '"',
                  traits['imagedate'],
                  average_trait(traits['sv_area']),
                  traits['tv_area'],
                  average_trait(traits['hull_area']),
                  average_trait(traits['solidity']),
                  average_trait(traits['height']),
                  average_trait(traits['perimeter']),
                  traits['access_level'],
                  traits['species'],
                  traits['site'],
                  traits['citation_author'],
                  traits['citation_year'],
                  traits['citation_title'],
                  traits['method']
                  ]

    return trait_list


# Compose the output CSV file, will be imported into BETYdb
def generate_average_csv(fname, fields, trait_list):
    """ Generate CSV called fname with fields and trait_list """
    csv = open(fname, 'w')
    csv.write(','.join(map(str, fields)) + '\n')
    csv.write(','.join(map(str, trait_list)) + '\n')
    csv.close()

    return fname


# Convert color data from string to float and wrap in a list
def serialize_color_data(input_list):
    newlist = [float(x) for x in input_list]

    return newlist


# PlantCV code for processing side-view images
def process_sv_images_core(vis_id, vis_img, nir_id, nir_rgb, nir_cv2, traits, experiment, vis_out, nir_out):
    if experiment == "Pilot_060214":
        vn_traits = process_sv_images_pilot(vis_id, vis_img, nir_id, nir_rgb, nir_cv2, traits, vis_out, nir_out)
    elif experiment == "TM015_F_051616":
        vn_traits = process_sv_images_lt1a(vis_id, vis_img, nir_id, nir_rgb, nir_cv2, traits, vis_out, nir_out)
    elif experiment == "TM016_F_052716":
        vn_traits = process_sv_images_lt1b(vis_id, vis_img, nir_id, nir_rgb, nir_cv2, traits, vis_out, nir_out)
    else:
        raise ValueError("Experiment {0} is not a valid experiment_id.".format(experiment))
    return vn_traits


# PlantCV code for processing top-view images
def process_tv_images_core(vis_id, vis_img, nir_id, nir_rgb, nir_cv2, brass_mask, traits, experiment, vis_out, nir_out):
    if experiment == "Pilot_060214":
        vn_traits = process_tv_images_pilot(vis_id, vis_img, nir_id, nir_rgb, nir_cv2, brass_mask, traits, vis_out,
                                            nir_out)
    elif experiment == "TM015_F_051616" or experiment == "TM016_F_052716":
        vn_traits = process_tv_images_lt1(vis_id, vis_img, nir_id, nir_rgb, nir_cv2, traits, vis_out, nir_out)
    else:
        raise ValueError("Experiment {0} is not a valid experiment_id.".format(experiment))
    return vn_traits


# Calculate the average value of a trait list
def average_trait(input_list):
    total = sum(input_list)
    if len(input_list) > 0:
        average = total / len(input_list)
    else:
        average = -1

    return average


# Helper function to fit VIS images onto NIR images
def crop_sides_equally(mask, nir, device, debug):
    device += 1
    # NumPy refers to y first then x
    mask_shape = np.shape(mask)  # type: tuple
    nir_shape = np.shape(nir)  # type: tuple
    final_y = mask_shape[0]
    final_x = mask_shape[1]
    difference_x = final_x - nir_shape[1]
    difference_y = final_y - nir_shape[0]
    if difference_x % 2 == 0:
        x1 = difference_x / 2
        x2 = difference_x / 2
    else:
        x1 = difference_x / 2
        x2 = (difference_x / 2) + 1

    if difference_y % 2 == 0:
        y1 = difference_y / 2
        y2 = difference_y / 2
    else:
        y1 = difference_y / 2
        y2 = (difference_y / 2) + 1
    crop_img = mask[y1:final_y - y2, x1:final_x - x2]

    if debug == "print":
        pcv.print_image(crop_img, str(device) + "_crop_sides_equally.png")
    elif debug == "plot":
        pcv.plot_image(crop_img, cmap="gray")

    return device, crop_img


# Helper function to fit VIS images onto NIR images
def conv_ratio(y=606.0, x=508.0, conv_x=1.125, conv_y=1.125, rat=1):
    prop2 = (x / 2056)
    prop1 = (y / 2454)
    prop2 = prop2 * (conv_y * rat)
    prop1 = prop1 * (conv_x * rat)
    return prop2, prop1


# Remove contours completely contained within a region of interest
def remove_countors_roi(mask, contours, hierarchy, roi, device, debug=None):
    clean_mask = np.copy(mask)
    # Loop over all contours in the image
    for n, contour in enumerate(contours):
        # This is the number of vertices in the contour
        contour_points = len(contour) - 1
        # Generate a list of x, y coordinates
        stack = np.vstack(contour)
        tests = []
        # Loop over the vertices for the contour and do a point polygon test
        for i in range(0, contour_points):
            # The point polygon test returns
            # 1 if the contour vertex is inside the ROI contour
            # 0 if the contour vertex is on the ROI contour
            # -1 if the contour vertex is outside the ROI contour
            pptest = cv2.pointPolygonTest(contour=roi[0], pt=(stack[i][0], stack[i][1]), measureDist=False)
            # Append the test result to the list of point polygon tests for this contour
            tests.append(pptest)
        # If all of the point polygon tests have a value of 1, then all the contour vertices are in the ROI
        if all(t == 1 for t in tests):
            # Fill in the contour as black
            cv2.drawContours(image=clean_mask, contours=contours, contourIdx=n, color=0, thickness=-1, lineType=8,
                             hierarchy=hierarchy)
    if debug == "print":
        pcv.print_image(filename=str(device) + "_remove_contours.png", img=clean_mask)
    elif debug == "plot":
        pcv.plot_image(clean_mask, cmap='gray')

    return device, clean_mask


# PlantCV code for processing side-view images from the experiment Pilot_060214
def process_sv_images_pilot(vis_id, vis_img, nir_id, nir_rgb, nir_cv2, traits, vis_out, nir_out):
    # Pipeline step
    device = 0
    debug = None

    # Convert RGB to HSV and extract the Saturation channel
    device, s = pcv.rgb2gray_hsv(vis_img, 's', device, debug)

    # Threshold the Saturation image
    device, s_thresh = pcv.binary_threshold(s, 36, 255, 'light', device, debug)

    # Median Filter
    device, s_mblur = pcv.median_blur(s_thresh, 5, device, debug)
    device, s_cnt = pcv.median_blur(s_thresh, 5, device, debug)

    # Fill small objects
    # device, s_fill = pcv.fill(s_mblur, s_cnt, 0, device, debug)

    # Convert RGB to LAB and extract the Blue channel
    device, b = pcv.rgb2gray_lab(vis_img, 'b', device, debug)

    # Threshold the blue image
    device, b_thresh = pcv.binary_threshold(b, 137, 255, 'light', device, debug)
    device, b_cnt = pcv.binary_threshold(b, 137, 255, 'light', device, debug)

    # Fill small objects
    # device, b_fill = pcv.fill(b_thresh, b_cnt, 10, device, debug)

    # Join the thresholded saturation and blue-yellow images
    device, bs = pcv.logical_and(s_mblur, b_cnt, device, debug)

    # Apply Mask (for vis images, mask_color=white)
    device, masked = pcv.apply_mask(vis_img, bs, 'white', device, debug)

    # Convert RGB to LAB and extract the Green-Magenta and Blue-Yellow channels
    device, masked_a = pcv.rgb2gray_lab(masked, 'a', device, debug)
    device, masked_b = pcv.rgb2gray_lab(masked, 'b', device, debug)

    # Threshold the green-magenta and blue images
    device, maskeda_thresh = pcv.binary_threshold(masked_a, 127, 255, 'dark', device, debug)
    device, maskedb_thresh = pcv.binary_threshold(masked_b, 128, 255, 'light', device, debug)

    # Join the thresholded saturation and blue-yellow images (OR)
    device, ab = pcv.logical_or(maskeda_thresh, maskedb_thresh, device, debug)
    device, ab_cnt = pcv.logical_or(maskeda_thresh, maskedb_thresh, device, debug)

    # Fill small noise
    device, ab_fill1 = pcv.fill(ab, ab_cnt, 200, device, debug)

    # Dilate to join small objects with larger ones
    device, ab_cnt1 = pcv.dilate(ab_fill1, 3, 2, device, debug)
    device, ab_cnt2 = pcv.dilate(ab_fill1, 3, 2, device, debug)

    # Fill dilated image mask
    device, ab_cnt3 = pcv.fill(ab_cnt2, ab_cnt1, 150, device, debug)
    device, masked2 = pcv.apply_mask(masked, ab_cnt3, 'white', device, debug)

    # Convert RGB to LAB and extract the Green-Magenta and Blue-Yellow channels
    device, masked2_a = pcv.rgb2gray_lab(masked2, 'a', device, debug)
    device, masked2_b = pcv.rgb2gray_lab(masked2, 'b', device, debug)

    # Threshold the green-magenta and blue images
    device, masked2a_thresh = pcv.binary_threshold(masked2_a, 127, 255, 'dark', device, debug)
    device, masked2b_thresh = pcv.binary_threshold(masked2_b, 128, 255, 'light', device, debug)

    device, masked2a_thresh_blur = pcv.median_blur(masked2a_thresh, 5, device, debug)
    device, masked2b_thresh_blur = pcv.median_blur(masked2b_thresh, 13, device, debug)

    device, ab_fill = pcv.logical_or(masked2a_thresh_blur, masked2b_thresh_blur, device, debug)

    # Identify objects
    device, id_objects, obj_hierarchy = pcv.find_objects(masked2, ab_fill, device, debug)

    # Define ROI
    device, roi1, roi_hierarchy = pcv.define_roi(masked2, 'rectangle', device, None, 'default', debug, True, 700,
                                                 0, -600, -300)

    # Decide which objects to keep
    device, roi_objects, hierarchy3, kept_mask, obj_area = pcv.roi_objects(vis_img, 'partial', roi1, roi_hierarchy,
                                                                           id_objects, obj_hierarchy, device,
                                                                           debug)

    # Object combine kept objects
    device, obj, mask = pcv.object_composition(vis_img, roi_objects, hierarchy3, device, debug)

    # ############# VIS Analysis ################
    # Find shape properties, output shape image (optional)
    device, shape_header, shape_data, shape_img = pcv.analyze_object(vis_img, vis_id, obj, mask, device, debug, vis_out)

    # Shape properties relative to user boundary line (optional)
    device, boundary_header, boundary_data, boundary_img1 = pcv.analyze_bound(vis_img, vis_id, obj, mask, 384, device,
                                                                              debug, vis_out)

    # Determine color properties: Histograms, Color Slices and
    # Pseudocolored Images, output color analyzed images (optional)
    device, color_header, color_data, color_img = pcv.analyze_color(vis_img, vis_id, mask, 256, device, debug,
                                                                    None, 'v', 'img', 300, vis_out)

    # Output shape and color data
    vis_traits = {}
    for i in range(1, len(shape_header)):
        vis_traits[shape_header[i]] = shape_data[i]
    for i in range(1, len(boundary_header)):
        vis_traits[boundary_header[i]] = boundary_data[i]
    for i in range(2, len(color_header)):
        vis_traits[color_header[i]] = serialize_color_data(color_data[i])

    # ############################ Use VIS image mask for NIR image#########################
    # Flip mask
    device, f_mask = pcv.flip(mask, "vertical", device, debug)

    # Reize mask
    device, nmask = pcv.resize(f_mask, 0.1154905775, 0.1154905775, device, debug)

    # position, and crop mask
    device, newmask = pcv.crop_position_mask(nir_rgb, nmask, device, 30, 4, "top", "right", debug)

    # Identify objects
    device, nir_objects, nir_hierarchy = pcv.find_objects(nir_rgb, newmask, device, debug)

    # Object combine kept objects
    device, nir_combined, nir_combinedmask = pcv.object_composition(nir_rgb, nir_objects, nir_hierarchy, device, debug)

    # ###################################### Analysis #############################################
    device, nhist_header, nhist_data, nir_imgs = pcv.analyze_NIR_intensity(nir_cv2, nir_rgb, nir_combinedmask, 256,
                                                                           device, False, debug, nir_out)
    device, nshape_header, nshape_data, nir_shape = pcv.analyze_object(nir_cv2, nir_id, nir_combined, nir_combinedmask,
                                                                       device, debug, nir_out)

    nir_traits = {}
    for i in range(1, len(nshape_header)):
        nir_traits[nshape_header[i]] = nshape_data[i]
    for i in range(2, len(nhist_header)):
        nir_traits[nhist_header[i]] = serialize_color_data(nhist_data[i])

    # Add data to traits table
    traits['sv_area'].append(vis_traits['area'])
    traits['hull_area'].append(vis_traits['hull-area'])
    traits['solidity'].append(vis_traits['solidity'])
    traits['height'].append(vis_traits['height_above_bound'])
    traits['perimeter'].append(vis_traits['perimeter'])

    output_images = []
    for row in shape_img + boundary_img1 + color_img + nir_imgs:
        output_images.append(row[2])
    return [vis_traits, nir_traits, output_images]


# PlantCV code for processing top-view images from the experiment Pilot_060214
def process_tv_images_pilot(vis_id, vis_img, nir_id, nir_rgb, nir_cv2, brass_mask, traits, vis_out, nir_out):
    device = 0
    debug = None

    # Convert RGB to HSV and extract the Saturation channel
    device, s = pcv.rgb2gray_hsv(vis_img, 's', device, debug)

    # Threshold the Saturation image
    device, s_thresh = pcv.binary_threshold(s, 75, 255, 'light', device, debug)

    # Median Filter
    device, s_mblur = pcv.median_blur(s_thresh, 5, device, debug)
    device, s_cnt = pcv.median_blur(s_thresh, 5, device, debug)

    # Fill small objects
    device, s_fill = pcv.fill(s_mblur, s_cnt, 150, device, debug)

    # Convert RGB to LAB and extract the Blue channel
    device, b = pcv.rgb2gray_lab(vis_img, 'b', device, debug)

    # Threshold the blue image
    device, b_thresh = pcv.binary_threshold(b, 138, 255, 'light', device, debug)
    device, b_cnt = pcv.binary_threshold(b, 138, 255, 'light', device, debug)

    # Fill small objects
    device, b_fill = pcv.fill(b_thresh, b_cnt, 100, device, debug)

    # Join the thresholded saturation and blue-yellow images
    device, bs = pcv.logical_and(s_fill, b_fill, device, debug)

    # Apply Mask (for vis images, mask_color=white)
    device, masked = pcv.apply_mask(vis_img, bs, 'white', device, debug)

    # Mask pesky brass piece
    device, brass_mask1 = pcv.rgb2gray_hsv(brass_mask, 'v', device, debug)
    device, brass_thresh = pcv.binary_threshold(brass_mask1, 0, 255, 'light', device, debug)
    device, brass_inv = pcv.invert(brass_thresh, device, debug)
    device, brass_masked = pcv.apply_mask(masked, brass_inv, 'white', device, debug)

    # Further mask soil and car
    device, masked_a = pcv.rgb2gray_lab(brass_masked, 'a', device, debug)
    device, soil_car1 = pcv.binary_threshold(masked_a, 128, 255, 'dark', device, debug)
    device, soil_car2 = pcv.binary_threshold(masked_a, 128, 255, 'light', device, debug)
    device, soil_car = pcv.logical_or(soil_car1, soil_car2, device, debug)
    device, soil_masked = pcv.apply_mask(brass_masked, soil_car, 'white', device, debug)

    # Convert RGB to LAB and extract the Green-Magenta and Blue-Yellow channels
    device, soil_a = pcv.rgb2gray_lab(soil_masked, 'a', device, debug)
    device, soil_b = pcv.rgb2gray_lab(soil_masked, 'b', device, debug)

    # Threshold the green-magenta and blue images
    device, soila_thresh = pcv.binary_threshold(soil_a, 124, 255, 'dark', device, debug)
    device, soilb_thresh = pcv.binary_threshold(soil_b, 148, 255, 'light', device, debug)

    # Join the thresholded saturation and blue-yellow images (OR)
    device, soil_ab = pcv.logical_or(soila_thresh, soilb_thresh, device, debug)
    device, soil_ab_cnt = pcv.logical_or(soila_thresh, soilb_thresh, device, debug)

    # Fill small objects
    device, soil_cnt = pcv.fill(soil_ab, soil_ab_cnt, 300, device, debug)

    # Apply mask (for vis images, mask_color=white)
    device, masked2 = pcv.apply_mask(soil_masked, soil_cnt, 'white', device, debug)

    # Identify objects
    device, id_objects, obj_hierarchy = pcv.find_objects(masked2, soil_cnt, device, debug)

    # Define ROI
    device, roi1, roi_hierarchy = pcv.define_roi(vis_img, 'rectangle', device, None, 'default', debug, True, 600, 450,
                                                 -600,
                                                 -350)

    # Decide which objects to keep
    device, roi_objects, hierarchy3, kept_mask, obj_area = pcv.roi_objects(vis_img, 'partial', roi1, roi_hierarchy,
                                                                           id_objects, obj_hierarchy, device, debug)

    # Object combine kept objects
    device, obj, mask = pcv.object_composition(vis_img, roi_objects, hierarchy3, device, debug)

    # Find shape properties, output shape image (optional)
    device, shape_header, shape_data, shape_img = pcv.analyze_object(vis_img, vis_id, obj, mask, device, debug, vis_out)

    # Determine color properties
    device, color_header, color_data, color_img = pcv.analyze_color(vis_img, vis_id, mask, 256, device, debug, None,
                                                                    'v', 'img', 300, vis_out)

    # Output shape and color data
    vis_traits = {}
    for i in range(1, len(shape_header)):
        vis_traits[shape_header[i]] = shape_data[i]
    for i in range(2, len(color_header)):
        vis_traits[color_header[i]] = serialize_color_data(color_data[i])

    # ############################ Use VIS image mask for NIR image #########################

    # Flip mask
    device, f_mask = pcv.flip(mask, "horizontal", device, debug)

    # Reize mask
    device, nmask = pcv.resize(f_mask, 0.116148, 0.116148, device, debug)

    # position, and crop mask
    device, newmask = pcv.crop_position_mask(nir_cv2, nmask, device, 15, 5, "top", "right", debug)

    # Identify objects
    device, nir_objects, nir_hierarchy = pcv.find_objects(nir_rgb, newmask, device, debug)

    # Object combine kept objects
    device, nir_combined, nir_combinedmask = pcv.object_composition(nir_rgb, nir_objects, nir_hierarchy, device, debug)

    # ###################################### Analysis #############################################
    device, nhist_header, nhist_data, nir_imgs = pcv.analyze_NIR_intensity(nir_cv2, nir_rgb, nir_combinedmask, 256,
                                                                           device, False, debug, nir_out)
    device, nshape_header, nshape_data, nir_shape = pcv.analyze_object(nir_cv2, nir_id, nir_combined, nir_combinedmask,
                                                                       device, debug, nir_out)

    nir_traits = {}
    for i in range(1, len(nshape_header)):
        nir_traits[nshape_header[i]] = nshape_data[i]
    for i in range(2, len(nhist_header)):
        nir_traits[nhist_header[i]] = serialize_color_data(nhist_data[i])

    # Add data to traits table
    traits['tv_area'] = vis_traits['area']

    output_images = []
    for row in shape_img + color_img + nir_imgs:
        output_images.append(row[2])
    return [vis_traits, nir_traits, output_images]


# PlantCV code for processing side-view images from the experiment TM015_F_051616
def process_sv_images_lt1a(vis_id, vis_img, nir_id, nir_rgb, nir_cv2, traits, vis_out, nir_out):
    # Pipeline step
    device = 0
    debug = None

    # Convert RGB to LAB and extract the Blue-Yellow channel
    device, blue_channel = pcv.rgb2gray_lab(img=vis_img, channel="b", device=device, debug=debug)

    # Threshold the blue image using the triangle autothreshold method
    device, blue_tri = pcv.triangle_auto_threshold(device=device, img=blue_channel, maxvalue=255, object_type="light",
                                                   xstep=1, debug=debug)

    # Extract core plant region from the image to preserve delicate plant features during filtering
    device += 1
    plant_region = blue_tri[0:1750, 600:2080]
    if debug is not None:
        pcv.print_image(filename=str(device) + "_extract_plant_region.png", img=plant_region)

    # Use a Gaussian blur to disrupt the strong edge features in the cabinet
    device, blur_gaussian = pcv.gaussian_blur(device=device, img=blue_tri, ksize=(3, 3), sigmax=0, sigmay=None,
                                              debug=debug)

    # Threshold the blurred image to remove features that were blurred
    device, blur_thresholded = pcv.binary_threshold(img=blur_gaussian, threshold=250, maxValue=255, object_type="light",
                                                    device=device, debug=debug)

    # Add the plant region back in to the filtered image
    device += 1
    blur_thresholded[0:1750, 600:2080] = plant_region
    if debug is not None:
        pcv.print_image(filename=str(device) + "_replace_plant_region.png", img=blur_thresholded)

    # Fill small noise
    device, blue_fill_50 = pcv.fill(img=np.copy(blur_thresholded), mask=np.copy(blur_thresholded), size=50,
                                    device=device, debug=debug)

    # Identify objects
    device, contours, contour_hierarchy = pcv.find_objects(img=vis_img, mask=blue_fill_50, device=device, debug=debug)

    # Define ROI
    device, roi, roi_hierarchy = pcv.define_roi(img=vis_img, shape="rectangle", device=device, roi=None,
                                                roi_input="default", debug=debug, adjust=True, x_adj=565, y_adj=0,
                                                w_adj=-490, h_adj=-250)

    # Decide which objects to keep
    device, roi_contours, roi_contour_hierarchy, _, _ = pcv.roi_objects(img=vis_img, roi_type="partial",
                                                                        roi_contour=roi, roi_hierarchy=roi_hierarchy,
                                                                        object_contour=contours,
                                                                        obj_hierarchy=contour_hierarchy, device=device,
                                                                        debug=debug)

    # If there are no contours left we cannot measure anything
    if len(roi_contours) > 0:
        # Object combine kept objects
        device, plant_contour, plant_mask = pcv.object_composition(img=vis_img, contours=roi_contours,
                                                                   hierarchy=roi_contour_hierarchy, device=device,
                                                                   debug=debug)

        # Find shape properties, output shape image (optional)
        device, shape_header, shape_data, shape_img = pcv.analyze_object(img=vis_img, imgname=vis_id, obj=plant_contour,
                                                                         mask=plant_mask, device=device,
                                                                         debug=debug, filename=vis_out)

        # Shape properties relative to user boundary line (optional)
        device, boundary_header, boundary_data, boundary_img = pcv.analyze_bound(img=vis_img, imgname=vis_id,
                                                                                 obj=plant_contour, mask=plant_mask,
                                                                                 line_position=380, device=device,
                                                                                 debug=debug, filename=vis_out)

        # Determine color properties: Histograms, Color Slices and Pseudocolored Images,
        # output color analyzed images (optional)
        device, color_header, color_data, color_img = pcv.analyze_color(img=vis_img, imgname=vis_id, mask=plant_mask,
                                                                        bins=256, device=device, debug=debug,
                                                                        hist_plot_type=None, pseudo_channel="v",
                                                                        pseudo_bkg="img", resolution=300,
                                                                        filename=vis_out)

        # Output shape and color data
        vis_traits = {}
        for i in range(1, len(shape_header)):
            vis_traits[shape_header[i]] = shape_data[i]
        for i in range(1, len(boundary_header)):
            vis_traits[boundary_header[i]] = boundary_data[i]
        for i in range(2, len(color_header)):
            vis_traits[color_header[i]] = serialize_color_data(color_data[i])

        # Make mask glovelike in proportions via dilation
        device, d_mask = pcv.dilate(plant_mask, kernel=1, i=0, device=device, debug=debug)

        # Resize mask
        prop2, prop1 = conv_ratio()
        device, nmask = pcv.resize(img=d_mask, resize_x=prop1, resize_y=prop2, device=device, debug=debug)

        # Convert the resized mask to a binary mask
        device, bmask = pcv.binary_threshold(img=nmask, threshold=0, maxValue=255, object_type="light", device=device,
                                             debug=debug)

        device, crop_img = crop_sides_equally(mask=bmask, nir=nir_cv2, device=device, debug=debug)

        # position, and crop mask
        device, newmask = pcv.crop_position_mask(img=nir_cv2, mask=crop_img, device=device, x=34, y=9, v_pos="top",
                                                 h_pos="right", debug=debug)

        # Identify objects
        device, nir_objects, nir_hierarchy = pcv.find_objects(img=nir_rgb, mask=newmask, device=device,
                                                              debug=debug)

        # Object combine kept objects
        device, nir_combined, nir_combinedmask = pcv.object_composition(img=nir_rgb, contours=nir_objects,
                                                                        hierarchy=nir_hierarchy, device=device,
                                                                        debug=debug)

        # Analyze NIR signal data
        device, nhist_header, nhist_data, nir_imgs = pcv.analyze_NIR_intensity(img=nir_cv2, rgbimg=nir_rgb,
                                                                               mask=nir_combinedmask, bins=256,
                                                                               device=device, histplot=False,
                                                                               debug=debug, filename=nir_out)

        # Analyze the shape of the plant contour from the NIR image
        device, nshape_header, nshape_data, nir_shape = pcv.analyze_object(img=nir_cv2, imgname=nir_id,
                                                                           obj=nir_combined, mask=nir_combinedmask,
                                                                           device=device, debug=debug,
                                                                           filename=nir_out)

        nir_traits = {}
        for i in range(1, len(nshape_header)):
            nir_traits[nshape_header[i]] = nshape_data[i]
        for i in range(2, len(nhist_header)):
            nir_traits[nhist_header[i]] = serialize_color_data(nhist_data[i])

        # Add data to traits table
        traits['sv_area'].append(vis_traits['area'])
        traits['hull_area'].append(vis_traits['hull-area'])
        traits['solidity'].append(vis_traits['solidity'])
        traits['height'].append(vis_traits['height_above_bound'])
        traits['perimeter'].append(vis_traits['perimeter'])

        output_images = []
        for row in shape_img + [boundary_img] + color_img + nir_imgs:
            output_images.append(row[2])
        return [vis_traits, nir_traits, output_images]
    else:
        return []


# PlantCV code for processing top-view images from the experiment TM015_F_051616 or TM016_F_052716
def process_tv_images_lt1(vis_id, vis_img, nir_id, nir_rgb, nir_cv2, traits, vis_out, nir_out):
    device = 0
    debug = None

    # Convert RGB to LAB and extract the Green-Magenta channel
    device, green_channel = pcv.rgb2gray_lab(img=vis_img, channel="a", device=device, debug=debug)

    # Invert the Green-Magenta image because the plant is dark green
    device, green_inv = pcv.invert(img=green_channel, device=device, debug=debug)

    # Threshold the inverted Green-Magenta image to mostly isolate green pixels
    device, green_thresh = pcv.binary_threshold(img=green_inv, threshold=134, maxValue=255, object_type="light",
                                                device=device, debug=debug)

    # Extract core plant region from the image to preserve delicate plant features during filtering
    device += 1
    plant_region = green_thresh[100:2000, 250:2250]
    if debug is not None:
        pcv.print_image(filename=str(device) + "_extract_plant_region.png", img=plant_region)

    # Use a Gaussian blur to disrupt the strong edge features in the cabinet
    device, blur_gaussian = pcv.gaussian_blur(device=device, img=green_thresh, ksize=(7, 7), sigmax=0, sigmay=None,
                                              debug=debug)

    # Threshold the blurred image to remove features that were blurred
    device, blur_thresholded = pcv.binary_threshold(img=blur_gaussian, threshold=250, maxValue=255, object_type="light",
                                                    device=device, debug=debug)

    # Add the plant region back in to the filtered image
    device += 1
    blur_thresholded[100:2000, 250:2250] = plant_region
    if debug is not None:
        pcv.print_image(filename=str(device) + "_replace_plant_region.png", img=blur_thresholded)

    # Use a median blur to breakup the horizontal and vertical lines caused by shadows from the track edges
    device, med_blur = pcv.median_blur(img=blur_thresholded, ksize=5, device=device, debug=debug)

    # Fill in small contours
    device, green_fill_50 = pcv.fill(img=np.copy(med_blur), mask=np.copy(med_blur), size=50,
                                     device=device, debug=debug)

    # Define an ROI for the brass stopper
    device, stopper_roi, stopper_hierarchy = pcv.define_roi(img=vis_img, shape="rectangle", device=device, roi=None,
                                                            roi_input="default", debug=debug, adjust=True,
                                                            x_adj=1420, y_adj=890, w_adj=-920, h_adj=-1040)

    # Identify all remaining contours in the binary image
    device, contours, hierarchy = pcv.find_objects(img=vis_img, mask=np.copy(green_fill_50), device=device,
                                                   debug=debug)

    # Remove contours completely contained within the stopper region of interest
    device, remove_stopper_mask = remove_countors_roi(mask=green_fill_50, contours=contours, hierarchy=hierarchy,
                                                      roi=stopper_roi, device=device, debug=debug)

    # Define an ROI for a screw hole
    device, screw_roi, screw_hierarchy = pcv.define_roi(img=vis_img, shape="rectangle", device=device, roi=None,
                                                        roi_input="default", debug=debug, adjust=True, x_adj=1870,
                                                        y_adj=1010, w_adj=-485, h_adj=-960)

    # Remove contours completely contained within the screw region of interest
    device, remove_screw_mask = remove_countors_roi(mask=remove_stopper_mask, contours=contours, hierarchy=hierarchy,
                                                    roi=screw_roi, device=device, debug=debug)

    # Identify objects
    device, contours, contour_hierarchy = pcv.find_objects(img=vis_img, mask=remove_screw_mask, device=device,
                                                           debug=debug)

    # Define ROI
    device, roi, roi_hierarchy = pcv.define_roi(img=vis_img, shape="rectangle", device=device, roi=None,
                                                roi_input="default", debug=debug, adjust=True, x_adj=565,
                                                y_adj=200, w_adj=-490, h_adj=-250)

    # Decide which objects to keep
    device, roi_contours, roi_contour_hierarchy, _, _ = pcv.roi_objects(img=vis_img, roi_type="partial",
                                                                        roi_contour=roi, roi_hierarchy=roi_hierarchy,
                                                                        object_contour=contours,
                                                                        obj_hierarchy=contour_hierarchy, device=device,
                                                                        debug=debug)

    # If there are no contours left we cannot measure anything
    if len(roi_contours) > 0:
        # Object combine kept objects
        device, plant_contour, plant_mask = pcv.object_composition(img=vis_img, contours=roi_contours,
                                                                   hierarchy=roi_contour_hierarchy, device=device,
                                                                   debug=debug)

        # Find shape properties, output shape image (optional)
        device, shape_header, shape_data, shape_img = pcv.analyze_object(img=vis_img, imgname=vis_id, obj=plant_contour,
                                                                         mask=plant_mask, device=device,
                                                                         debug=debug, filename=vis_out)

        # Determine color properties: Histograms, Color Slices and Pseudocolored Images,
        # output color analyzed images (optional)
        device, color_header, color_data, color_img = pcv.analyze_color(img=vis_img, imgname=vis_id, mask=plant_mask,
                                                                        bins=256, device=device, debug=debug,
                                                                        hist_plot_type=None, pseudo_channel="v",
                                                                        pseudo_bkg="img", resolution=300,
                                                                        filename=vis_out)
        # Output shape and color data
        vis_traits = {}
        for i in range(1, len(shape_header)):
            vis_traits[shape_header[i]] = shape_data[i]
        for i in range(2, len(color_header)):
            vis_traits[color_header[i]] = serialize_color_data(color_data[i])

        # Make mask glovelike in proportions via dilation
        device, d_mask = pcv.dilate(plant_mask, kernel=1, i=0, device=device, debug=debug)

        # Resize mask
        prop2, prop1 = conv_ratio()
        device, nmask = pcv.resize(img=d_mask, resize_x=prop1, resize_y=prop2, device=device, debug=debug)

        # Convert the resized mask to a binary mask
        device, bmask = pcv.binary_threshold(img=nmask, threshold=0, maxValue=255, object_type="light", device=device,
                                             debug=debug)

        device, crop_img = crop_sides_equally(mask=bmask, nir=nir_cv2, device=device, debug=debug)

        # position, and crop mask
        device, newmask = pcv.crop_position_mask(img=nir_cv2, mask=crop_img, device=device, x=0, y=1, v_pos="bottom",
                                                 h_pos="right", debug=debug)

        # Identify objects
        device, nir_objects, nir_hierarchy = pcv.find_objects(img=nir_rgb, mask=newmask, device=device,
                                                              debug=debug)

        # Object combine kept objects
        device, nir_combined, nir_combinedmask = pcv.object_composition(img=nir_rgb, contours=nir_objects,
                                                                        hierarchy=nir_hierarchy, device=device,
                                                                        debug=debug)

        # Analyze NIR signal data
        device, nhist_header, nhist_data, nir_imgs = pcv.analyze_NIR_intensity(img=nir_cv2, rgbimg=nir_rgb,
                                                                               mask=nir_combinedmask, bins=256,
                                                                               device=device, histplot=False,
                                                                               debug=debug, filename=nir_out)

        # Analyze the shape of the plant contour from the NIR image
        device, nshape_header, nshape_data, nir_shape = pcv.analyze_object(img=nir_cv2, imgname=nir_id,
                                                                           obj=nir_combined, mask=nir_combinedmask,
                                                                           device=device, debug=debug,
                                                                           filename=nir_out)
        nir_traits = {}
        for i in range(1, len(nshape_header)):
            nir_traits[nshape_header[i]] = nshape_data[i]
        for i in range(2, len(nhist_header)):
            nir_traits[nhist_header[i]] = serialize_color_data(nhist_data[i])

        # Add data to traits table
        traits['tv_area'] = vis_traits['area']
        output_images = []
        for row in shape_img + color_img + nir_imgs:
            output_images.append(row[2])
        return [vis_traits, nir_traits, output_images]
    else:
        return []


# PlantCV code for processing side-view images from the experiment TM016_F_052716
def process_sv_images_lt1b(vis_id, vis_img, nir_id, nir_rgb, nir_cv2, traits, vis_out, nir_out):
    # Pipeline step
    device = 0
    debug = None

    # Convert RGB to LAB and extract the Blue-Yellow channel
    device, blue_channel = pcv.rgb2gray_lab(img=vis_img, channel="b", device=device, debug=debug)

    # Threshold the blue image using the triangle autothreshold method
    device, blue_tri = pcv.triangle_auto_threshold(device=device, img=blue_channel, maxvalue=255, object_type="light",
                                                   xstep=1, debug=debug)

    # Extract core plant region from the image to preserve delicate plant features during filtering
    device += 1
    plant_region = blue_tri[0:1750, 600:2080]
    if debug is not None:
        pcv.print_image(filename=str(device) + "_extract_plant_region.png", img=plant_region)

    # Use a Gaussian blur to disrupt the strong edge features in the cabinet
    device, blur_gaussian = pcv.gaussian_blur(device=device, img=blue_tri, ksize=(3, 3), sigmax=0, sigmay=None,
                                              debug=debug)

    # Threshold the blurred image to remove features that were blurred
    device, blur_thresholded = pcv.binary_threshold(img=blur_gaussian, threshold=250, maxValue=255, object_type="light",
                                                    device=device, debug=debug)

    # Add the plant region back in to the filtered image
    device += 1
    blur_thresholded[0:1750, 600:2080] = plant_region
    if debug is not None:
        pcv.print_image(filename=str(device) + "_replace_plant_region.png", img=blur_thresholded)

    # Fill small noise
    device, blue_fill_50 = pcv.fill(img=np.copy(blur_thresholded), mask=np.copy(blur_thresholded), size=50,
                                    device=device, debug=debug)

    # Identify objects
    device, contours, contour_hierarchy = pcv.find_objects(img=vis_img, mask=blue_fill_50, device=device, debug=debug)

    # Define ROI
    device, roi, roi_hierarchy = pcv.define_roi(img=vis_img, shape="rectangle", device=device, roi=None,
                                                roi_input="default", debug=debug, adjust=True, x_adj=565, y_adj=0,
                                                w_adj=-490, h_adj=-250)

    # Decide which objects to keep
    device, roi_contours, roi_contour_hierarchy, _, _ = pcv.roi_objects(img=vis_img, roi_type="partial",
                                                                        roi_contour=roi, roi_hierarchy=roi_hierarchy,
                                                                        object_contour=contours,
                                                                        obj_hierarchy=contour_hierarchy, device=device,
                                                                        debug=debug)

    # If there are no contours left we cannot measure anything
    if len(roi_contours) > 0:
        # Object combine kept objects
        device, plant_contour, plant_mask = pcv.object_composition(img=vis_img, contours=roi_contours,
                                                                   hierarchy=roi_contour_hierarchy, device=device,
                                                                   debug=debug)

        # Find shape properties, output shape image (optional)
        device, shape_header, shape_data, shape_img = pcv.analyze_object(img=vis_img, imgname=vis_id, obj=plant_contour,
                                                                         mask=plant_mask, device=device,
                                                                         debug=debug, filename=vis_out)

        # Shape properties relative to user boundary line (optional)
        device, boundary_header, boundary_data, boundary_img = pcv.analyze_bound(img=vis_img, imgname=vis_id,
                                                                                 obj=plant_contour, mask=plant_mask,
                                                                                 line_position=440, device=device,
                                                                                 debug=debug, filename=vis_out)

        # Determine color properties: Histograms, Color Slices and Pseudocolored Images,
        # output color analyzed images (optional)
        device, color_header, color_data, color_img = pcv.analyze_color(img=vis_img, imgname=vis_id, mask=plant_mask,
                                                                        bins=256, device=device, debug=debug,
                                                                        hist_plot_type=None, pseudo_channel="v",
                                                                        pseudo_bkg="img", resolution=300,
                                                                        filename=vis_out)

        # Output shape and color data
        vis_traits = {}
        for i in range(1, len(shape_header)):
            vis_traits[shape_header[i]] = shape_data[i]
        for i in range(1, len(boundary_header)):
            vis_traits[boundary_header[i]] = boundary_data[i]
        for i in range(2, len(color_header)):
            vis_traits[color_header[i]] = serialize_color_data(color_data[i])

        # Make mask glovelike in proportions via dilation
        device, d_mask = pcv.dilate(plant_mask, kernel=1, i=0, device=device, debug=debug)

        # Resize mask
        prop2, prop1 = conv_ratio()
        device, nmask = pcv.resize(img=d_mask, resize_x=prop1, resize_y=prop2, device=device, debug=debug)

        # Convert the resized mask to a binary mask
        device, bmask = pcv.binary_threshold(img=nmask, threshold=0, maxValue=255, object_type="light", device=device,
                                             debug=debug)

        device, crop_img = crop_sides_equally(mask=bmask, nir=nir_cv2, device=device, debug=debug)

        # position, and crop mask
        device, newmask = pcv.crop_position_mask(img=nir_cv2, mask=crop_img, device=device, x=34, y=9, v_pos="top",
                                                 h_pos="right", debug=debug)

        # Identify objects
        device, nir_objects, nir_hierarchy = pcv.find_objects(img=nir_rgb, mask=newmask, device=device,
                                                              debug=debug)

        # Object combine kept objects
        device, nir_combined, nir_combinedmask = pcv.object_composition(img=nir_rgb, contours=nir_objects,
                                                                        hierarchy=nir_hierarchy, device=device,
                                                                        debug=debug)

        # Analyze NIR signal data
        device, nhist_header, nhist_data, nir_imgs = pcv.analyze_NIR_intensity(img=nir_cv2, rgbimg=nir_rgb,
                                                                               mask=nir_combinedmask, bins=256,
                                                                               device=device, histplot=False,
                                                                               debug=debug, filename=nir_out)

        # Analyze the shape of the plant contour from the NIR image
        device, nshape_header, nshape_data, nir_shape = pcv.analyze_object(img=nir_cv2, imgname=nir_id,
                                                                           obj=nir_combined, mask=nir_combinedmask,
                                                                           device=device, debug=debug,
                                                                           filename=nir_out)

        nir_traits = {}
        for i in range(1, len(nshape_header)):
            nir_traits[nshape_header[i]] = nshape_data[i]
        for i in range(2, len(nhist_header)):
            nir_traits[nhist_header[i]] = serialize_color_data(nhist_data[i])

        # Add data to traits table
        traits['sv_area'].append(vis_traits['area'])
        traits['hull_area'].append(vis_traits['hull-area'])
        traits['solidity'].append(vis_traits['solidity'])
        traits['height'].append(vis_traits['height_above_bound'])
        traits['perimeter'].append(vis_traits['perimeter'])

        output_images = []
        for row in shape_img + [boundary_img] + color_img + nir_imgs:
            output_images.append(row[2])
        return [vis_traits, nir_traits, output_images]
    else:
        return []
