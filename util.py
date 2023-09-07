import pickle

from skimage.transform import resize
import numpy as np
import cv2


EMPTY = True
NOT_EMPTY = False

MODEL = pickle.load(open("model.p", "rb"))

def empty_or_not(spot_bgr):
    # single dimensional arr, store flattened data of resized parking spot image.
    flat_data = []
    # resizes image
    img_resized = resize(spot_bgr, (15, 15, 3))
    # append the flattened image to the arr
    flat_data.append(img_resized.flatten())
    # convert the arr to numpy array
    flat_data = np.array(flat_data)
    # use the pre-trained model to make predictions with the flat data
    y_output = MODEL.predict(flat_data)

    if y_output == 0:
        return EMPTY
    else:
        return NOT_EMPTY

# function that grabs the parking spots, creates bounding boxes coordinates
def get_parking_spots_bboxes(connected_components):
    (totalLabels, label_ids, values, centroid) = connected_components
    # initialize empty list of slots
    slots = []
    # coefficient is used to scale the coordinates of the bounding boxes to match original image
    coef = 1
    # iterates through each labels component skipping label 0
    for i in range(1, totalLabels):
        # Now extract the coordinate points
        x1 = int(values[i, cv2.CC_STAT_LEFT] * coef)
        y1 = int(values[i, cv2.CC_STAT_TOP] * coef)
        w = int(values[i, cv2.CC_STAT_WIDTH] * coef)
        h = int(values[i, cv2.CC_STAT_HEIGHT] * coef)

        slots.append([x1, y1, w, h])

    return slots
