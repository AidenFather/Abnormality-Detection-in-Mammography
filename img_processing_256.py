import re
import numpy as np
import os
import PIL
import random
import shutil
import matplotlib.pyplot as plt
import PIL
import sys
import cv2
from PIL import Image, ImageMath

def rename(name):
	patient_id = re.findall("(P_[\d]+)_", name)
	
	image_side = re.findall("_(LEFT|RIGHT)_", name)

	if len(image_side) > 0:
		image_side = image_side[0]
	else:
		print("Side error")
		return name

	image_type = re.findall("(CC|MLO)", name)
	if len(image_type) > 0:
		image_type = image_type[0]
	else:
		return name
	
	return patient_id + "_" + image_side + "_" + image_type  



def mask_img(mask_path, full_image_arr, slice_size=512, return_size=False, half=True, output=True):
	"""
	input: path to mask image PNG
	opens the mask, reduces its size by half, finds the borders of the mask and returns the center of the mass
	if the mass is bigger than the slice it returns the upper left and lower right corners of the mask as tuples
	which will be used to create multiple slices
	returns: center_row - int with center row of mask, or tuple with edges of the mask if the mask is bigger than the slice
			 center_col - idem
			 too_big - boolean indicating if the mask is bigger than the slice
	"""
	mask = PIL.Image.open(mask_path)    
	if half:
		h, w = mask.size
		new_size = ( h // 2, w // 2)
		mask.thumbnail(new_size, PIL.Image.ANTIALIAS)

	mask_arr = np.array(mask)
	mask_arr = mask_arr[:,:,0]
		
	if np.sum(np.sum(full_image_arr >= 225)) > 20000:
		full_image_arr = remove_margins(full_image_arr)
		mask_arr = remove_margins(mask_arr)
		if output:
			print("Trimming borders", mask_path)
			
	# The maks size must be same as the full image size
	if mask_arr.shape != full_image_arr.shape:
		# see if the ratios are the same
		mask_ratio = mask_arr.shape[0] / mask_arr.shape[1]
		image_ratio = full_image_arr.shape[0] / full_image_arr.shape[1]
		
		if abs(mask_ratio - image_ratio) <=  1e-03:
			if output:
				print("Mishaped mask, resizing mask", mask_path)
			
			# reshape the mask to match the image
			#mask_arr = imresize(mask_arr, full_image_arr.shape)
			mask_arr = np.array(Image.fromarray(mask_arr).resize(full_image_arr.shape))
		else:
			if output:
				print("Mask shape:", mask_arr.shape)
				print("Image shape:", full_image_arr.shape)
			print("Mask shape doesn't match image!", mask_path)
			return 0, 0, False, full_image_arr, 0
	
	# find the borders
	mask_mask = mask_arr == 255

	# check whether each row or column have a white pixel
	cols = np.sum(mask_mask, axis=0)
	rows = np.sum(mask_mask, axis=1)

	# check corners
	first_col = np.argmax(cols > 0)
	last_col = mask_arr.shape[1] - np.argmax(np.flip(cols, axis=0) > 0)
	center_col = int((first_col + last_col) / 2)

	first_row = np.argmax(rows > 0)
	last_row = mask_arr.shape[0] - np.argmax(np.flip(rows, axis=0) > 0)
	center_row = int((first_row + last_row) / 2)
	
	col_size = last_col - first_col
	row_size = last_row - first_row
	
	mask_size = [row_size, col_size]
	
	# When a mask size is bigger than a slice
	too_big = False
	
	if (last_col - first_col > slice_size + 30) or (last_row - first_row > slice_size + 30):
		too_big = True
	  
	return center_row, center_col, too_big, full_image_arr, mask_size


def remove_margins(image_arr, margin=20):
	"""
	function to trim plxels off all sides of an image
	"""
	h, w = image_arr.shape
	new_image = image_arr[margin:h-margin,margin:w-margin]
	return new_image



def random_flip_img_train(img):
    fliplr = np.random.binomial(1,0.5)
    flipud = np.random.binomial(1,0.5)
    
    if fliplr:
        img = np.flip(img, 1)
    if flipud:
        img = np.flip(img, 0)
        
    return random_rotate_image_train(img)


def crop_img(img):
	slice_size=512
	tile_size=256
    img_h = img.shape[0]
    img_w = img.shape[1]
    
    # make sure the image is big enough to use
    if (img_h < slice_size) or (img_w < slice_size):
        print("Error - image is wrong size!", img.shape)
        return np.array([0])
    
    # pick a random place to start the crop so that the crop will be the right size
    start_row = np.random.randint(low=0, high=(img_h - slice_size))
    start_col = np.random.randint(low=0, high=(img_w - slice_size))
    
    end_row = start_row + slice_size
    end_col = start_col + slice_size
    
    # crop the image and randomly rotate it
    cropped_img = random_flip_img_train(img[start_row:end_row, start_col:end_col])
    
    # make sure the image is the right size
    if cropped_img.shape[0] == cropped_img.shape[1]:
        # resize it and return it
        cropped_img = cv2.resize(cropped_img, dsize=(tile_size, tile_size), interpolation=cv2.INTER_CUBIC) 
        return cropped_img.reshape((tile_size, tile_size, 1))
    
    # else repeat until the image is the right size
    else:
        return crop_img(img)


def random_rotate_img_train(img):
    rotations = np.random.randint(low=-3, high=3)
    return np.rot90(img, rotations)