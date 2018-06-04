# OpenCV_Object Detection
ball tracking camera using TensorFlow and OpenCV

The process by which this was done was by:

1. gathering a data set of images.
2. drawing a bounding box around the images.
3. generating individual .xml files with bounding box location data as well as image name for each image.
4. generating .csv file with merged xml data of all images in one file.
5. generating .tfrecord binary file.
6. train model using TensorFlow.
7. use trained model with OpenCV.
8. distinguish between the pixels on the screen as a x and y coordinate system.
