import cv2
import matplotlib.pyplot as plt
import numpy as np

from util import get_parking_spots_bboxes, empty_or_not

# Define a function to calculate the absolute difference in mean values between two images
def calc_diff(im1, im2):
    return np.abs(np.mean(im1) - np.mean(im2))

# Define file paths for the mask and video
mask = 'data/mask_1920_1080.png'
video_path = 'data/parking_1920_1080_loop.mp4'

# Read the mask image and store it in grayscale
mask = cv2.imread(mask, 0)

# Open the video file for reading
cap = cv2.VideoCapture(video_path)

# Extract connected components from the mask image
connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)

# Get bounding boxes for parking spots
spots = get_parking_spots_bboxes(connected_components)

# Initialize lists to store spot statuses and differences
spots_status = [None for j in spots]
diffs = [None for j in spots]

# Initialize variables for previous frame, frame number, and retrieval status
previous_frame = None
frame_nmr = 0
ret = True

# Define the step interval for frame processing
step = 30

# Process frames from the video
while ret:
    ret, frame = cap.read()

    # Process frames at specific intervals and when a previous frame is available
    if frame_nmr % step == 0 and previous_frame is not None:
        for spot_indx, spot in enumerate(spots):
            x1, y1, w, h = spot

            # Crop the frame to the parking spot region
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]

            # Calculate the difference in mean values between spot_crop and the previous frame region
            diffs[spot_indx] = calc_diff(spot_crop, previous_frame[y1:y1 + h, x1:x1 + w, :])

        # Print the differences in descending order
        print([diffs[j] for j in np.argsort(diffs)][::-1])

    # Process frames at specific intervals
    if frame_nmr % step == 0:
        if previous_frame is None:
            arr_ = range(len(spots))
        else:
            # Sort spot indices based on differences and a threshold
            arr_ = [j for j in np.argsort(diffs) if diffs[j] / np.amax(diffs) > 0.4]
        for spot_indx in arr_:
            spot = spots[spot_indx]
            x1, y1, w, h = spot

            # Crop the frame to the parking spot region
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]

            # Determine if the parking spot is empty or not
            spot_status = empty_or_not(spot_crop)

            spots_status[spot_indx] = spot_status

    # Process frames at specific intervals
    if frame_nmr % step == 0:
        previous_frame = frame.copy()

    # Draw rectangles and text on the frame based on spot statuses
    for spot_indx, spot in enumerate(spots):
        spot_status = spots_status[spot_indx]
        x1, y1, w, h = spots[spot_indx]

        # Draw rectangles around parking spots based on their status
        if spot_status:
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
        else:
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)

    # Draw a black rectangle and text displaying available spots information
    cv2.rectangle(frame, (80, 20), (550, 80), (0, 0, 0), -1)
    cv2.putText(frame, 'Available Spots: {} / {}'.format(str(sum(spots_status)), str(len(spots_status))), (100, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Create a named window and display the frame
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.imshow('frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    frame_nmr += 1

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
