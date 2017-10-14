# PlantCV analysis code package for TERRA-REF indoor system analysis

from __future__ import print_function
import sys
import plantcv as pcv


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
def process_sv_images_core(vis_id, vis_img, nir_id, nir_rgb, nir_cv2, traits, debug=None):
    # Pipeline step
    device = 0

    # Convert RGB to HSV and extract the Saturation channel
    device, s = pcv.rgb2gray_hsv(vis_img, 's', device, debug)

    # Threshold the Saturation image
    device, s_thresh = pcv.binary_threshold(s, 36, 255, 'light', device, debug)

    # Median Filter
    device, s_mblur = pcv.median_blur(s_thresh, 5, device, debug)
    device, s_cnt = pcv.median_blur(s_thresh, 5, device, debug)

    # Fill small objects
    # device, s_fill = pcv.fill(s_mblur, s_cnt, 0, device, args.debug)

    # Convert RGB to LAB and extract the Blue channel
    device, b = pcv.rgb2gray_lab(vis_img, 'b', device, debug)

    # Threshold the blue image
    device, b_thresh = pcv.binary_threshold(b, 137, 255, 'light', device, debug)
    device, b_cnt = pcv.binary_threshold(b, 137, 255, 'light', device, debug)

    # Fill small objects
    # device, b_fill = pcv.fill(b_thresh, b_cnt, 10, device, args.debug)

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
    device, shape_header, shape_data, shape_img = pcv.analyze_object(vis_img, vis_id, obj, mask, device, debug)

    # Shape properties relative to user boundary line (optional)
    device, boundary_header, boundary_data, boundary_img1 = pcv.analyze_bound(vis_img, vis_id, obj, mask, 384, device,
                                                                              debug)

    # Determine color properties: Histograms, Color Slices and
    # Pseudocolored Images, output color analyzed images (optional)
    device, color_header, color_data, color_img = pcv.analyze_color(vis_img, vis_id, mask, 256, device, debug,
                                                                    None, 'v', 'img', 300)

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
                                                                           device, False, debug)
    device, nshape_header, nshape_data, nir_shape = pcv.analyze_object(nir_cv2, nir_id, nir_combined, nir_combinedmask,
                                                                       device, debug)

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

    return [vis_traits, nir_traits]


# PlantCV code for processing top-view images
def process_tv_images_core(vis_id, vis_img, nir_id, nir_rgb, nir_cv2, brass_mask, traits, debug=None):
    device = 0

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
    device, shape_header, shape_data, shape_img = pcv.analyze_object(vis_img, vis_id, obj, mask, device, debug)

    # Determine color properties
    device, color_header, color_data, color_img = pcv.analyze_color(vis_img, vis_id, mask, 256, device, debug, None,
                                                                    'v', 'img', 300)

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
                                                                           device, False, debug)
    device, nshape_header, nshape_data, nir_shape = pcv.analyze_object(nir_cv2, nir_id, nir_combined, nir_combinedmask,
                                                                       device, debug)

    nir_traits = {}
    for i in range(1, len(nshape_header)):
        nir_traits[nshape_header[i]] = nshape_data[i]
    for i in range(2, len(nhist_header)):
        nir_traits[nhist_header[i]] = serialize_color_data(nhist_data[i])

    # Add data to traits table
    traits['tv_area'] = vis_traits['area']

    return [vis_traits, nir_traits]


# Calculate the average value of a trait list
def average_trait(input_list):
    total = sum(input_list)
    if len(input_list) > 0:
        average = total / len(input_list)
    else:
        average = -1

    return average
