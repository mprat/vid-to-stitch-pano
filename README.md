# vid-to-stitch-pano
Inspired by a video of Robin's paper / capacitor airplane. The goal is to stitch a panorama view from the video to show the little plane flight.


To run:
----
You need to have OpenCV installed. Right now this only works with OpenCV 2.9.11 (tested), not with OpenCV 3.0.


Improvement ideas:
----
[] Need to improve how good the homographies are, because they are terrible. The images are not good quality, and lots of them are fuzzy clouds. It would be nice to specify image regions from which to extract key points, or at least weight them in some way according to the "niceness" of that part of the image.