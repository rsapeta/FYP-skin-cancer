
import numpy as np  
from PIL import Image, ImageOps
from skimage import morphology
from skimage.color import rgba2rgb
from skimage.segmentation import slic



def apply_mask(image, mask):
    '''Applies a mask to an image'''
    #Check if the image has an aplha channel
    if image.shape[2] == 4:
        image = rgba2rgb(image)

    # Apply mask on the image
    masked_image = np.copy(image)
    masked_image[mask == 0] = 0
    return masked_image

def fix_mask(mask):
    '''Turns a mask into only black(0) or white(255) pixel values. Put the leison in the
     center of the image '''

    #Make it purely black and white
    mask = mask.astype(int)
    mask_pic = Image.fromarray(np.uint8(mask * 255))
    
    #Crop the leison 
    row_index =  np.where(np.sum(mask, axis=1)>0)[0] #all the rows with at least one white element 
    first_row , last_row = row_index[0] , row_index[-1]  #first and last row 
    if (last_row - first_row) %2 != 0:
        last_row += 1 #one extra row to make it even and able to halve it

    col_index =  np.where(np.sum(mask, axis=0)>0)[0] #all the col with at least one white element 
    first_col , last_col = col_index[0] , col_index[-1]  #first and last col 
    if (last_col - first_col) %2 != 0:
        last_col += 1

    cropped_mask = mask_pic.crop((first_col,first_row,last_col,last_row))

    #Add borders
    old_width , old_height = cropped_mask.size 
    fixed_mask = ImageOps.expand(cropped_mask, border = int(old_width/2))

    return fixed_mask

def test_asymmetry(mask):
    '''Takes a mask image, halves it vertically, compares both sides. Returns an index of asymmetry 
    as the proportion of leisure that differs on both sides over the total leison'''

    width,height = mask.size

    #Cut in half
    left = mask.crop((0, 0, int(width/2), height)) #left part of picture (left, top, right, bottom)
    right = mask.crop((int(width/2), 0, width, height)) #right part of picture
    right = right.transpose(Image.FLIP_LEFT_RIGHT) #flip right part to compare

    #Compairing both sides
    asym = np.sum(np.where(np.array(left) != np.array(right), 1, 0))
    total_white = np.sum(np.where(np.array(mask)==255, 1, 0))
   
    return round((asym/total_white), 3)

def get_asymmetry(mask):
    '''Returns the asymmetry for a given leison by rotating the mask image by several angles, measuring 
     the proportion of asymmetry on each, and returning the minimum index. '''
    
    #Leison in the center of the image. Expand black borders to give freedom when rotating image. 
    mask = fix_mask(mask)

    asym = [test_asymmetry(mask.rotate(angle)) for angle in [0,15,30,45,60,75,90]]

    return round(np.min(asym), 3)


def get_average_color(image, mask):
    # Apply mask to image
    masked_image = apply_mask(image, mask)

    r = masked_image[:, :, 0]
    g = masked_image[:, :, 1]
    b = masked_image[:, :, 2]
    
    return [(np.mean(r[r > 0])), (np.mean(g[g > 0])), (np.mean(b[b > 0]))]

def get_color_variability(image, mask, measure='variance'):
    
    # Apply mask on the image
    masked_image = apply_mask(image, mask)
 
    # Find the non-black pixels (i.e., the lesion pixels)
    non_black_pixels = np.where(np.any(masked_image > 0, axis=None))

    # Extract the color values of the non-black pixels
    r = masked_image[non_black_pixels][:, 0]
    g = masked_image[non_black_pixels][:, 1]
    b = masked_image[non_black_pixels][:, 2]

    # Divide the lesion pixels into segments of similar color
    segments_slic = slic(masked_image[non_black_pixels], n_segments=10, compactness=1, sigma=3, start_label=1)

    # Compute the color variability for each segment that is within the lesion
    segment_color_variability_r = []
    segment_color_variability_g = []
    segment_color_variability_b = []
    for i in range(1, np.max(segments_slic) + 1):
        segment_pixels = np.where(segments_slic == i)[0]
        segment_r = r[segment_pixels]
        segment_g = g[segment_pixels]
        segment_b = b[segment_pixels]

        if measure == 'variance':
            rgb_variability = (np.var(segment_r), np.var(segment_g), np.var(segment_b))
        elif measure == 'standard_deviation':
            rgb_variability = (np.std(segment_r), np.std(segment_g), np.std(segment_b))
        else:
            return None

        segment_color_variability_r.append(rgb_variability[0])
        segment_color_variability_g.append(rgb_variability[1])
        segment_color_variability_b.append(rgb_variability[2])

    return [np.mean(segment_color_variability_r), np.mean(segment_color_variability_g), np.mean(segment_color_variability_b)]

def area_perimeter(mask): 
    '''Measures the area and perimeter of a mask image'''

    mask = np.where(mask==1, 1, 0)

    #area: the sum of all white pixels in the mask image
    area = np.sum(mask)

    #perimeter: first find which pixels belong to the perimeter.
    struct_el = morphology.disk(1)
    mask_eroded = morphology.binary_erosion(mask, struct_el)
    image_perimeter = mask - mask_eroded
    perimeter = np.sum(image_perimeter)

    return area, perimeter

 
def get_compactness(mask):
    '''Computes and returns the compactness of a figure'''

    area, perimeter = area_perimeter(mask)

    return round( (perimeter ** 2) /(4* np.pi *area), 4)

def extract_features(image,mask):
    '''Extracts the features of a given mask and image'''
    features = []

    features.append(get_asymmetry(mask))
    # features.append(get_color_variability(image,mask)) split into 3
    features.append((get_color_variability(image,mask)[0]))
    features.append((get_color_variability(image,mask)[1]))
    features.append((get_color_variability(image,mask)[2]))
    features.append(get_compactness(mask))
    # features.append(get_average_color(image,mask)) split into 3
    features.append((get_average_color(image,mask)[0]))
    features.append((get_average_color(image,mask)[1]))
    features.append((get_average_color(image,mask)[2]))

    return features


