# Parking Spot Detection

This project is a Python application that uses OpenCV to detect available parking spots in a video feed.
Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.
Prerequisites

You need to have Python installed along with the following libraries:

- OpenCV
- NumPy
- Matplotlib

You can install these using pip:
matplotlib
Running the Application

To run the application, navigate to the directory containing main.py and run the following command:

`python main.py`

## How It Works

The application reads a video file and a mask image. The mask image is used to identify the regions of interest (parking spots) in the video frames. The application then processes the video frames at specific intervals, calculating the difference in mean values between the current frame and the previous frame for each parking spot. If the difference exceeds a certain threshold, the spot is considered occupied; otherwise, it's considered available.

The application also provides a visual output, drawing rectangles around the parking spots and displaying the number of available spots.

## License

This script is open-source and is licensed under the MIT License. For more information, consult the [LICENSE](LICENSE) file.
