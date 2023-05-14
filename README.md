# Triangles-CV
The goal is to automatically detect equilateral triangles in natural images using a variant of the Hough Transform.
All the code is written in Python.

# Explanations of the process
First we ran the CANNY algorithm. We blurred the image in order to get a relatively clean edge map, and calculated the gradient.
Next, we went through each edge point found, and scored for all the centers of mass of the triangles it could participate in, given its angle.
The pointing was done with the help of calculating the slope, finding equations of a straight line, and the fact that this point is at most a distance of half a side from the center of the side, from which the distance to the center of mass is h/3.
In the next picture you can see all the votes (threshold=10) for a certain picture, you can clearly see that the center of mass of the triangle will receive the most votes.
![‏‏תמונה לתרגיל משולשים 1 1 1](https://github.com/EinatLasry/Triangles-CV/assets/82314695/ad73e9b4-1798-4997-8bc6-e9ea9037037a)

In the next step, we deleted points that did not pass the threshold of votes. We left one local maximum point for each window, the size of the window was adjusted to the size of the given triangle.
For each point we identified as the center of mass of the triangle, we kept the angle of the sides from which the votes came. Note that all sides have the same angle, if we apply modulo 120.
Using the angle, we reproduced the vertices of the triangle, and thus drew the triangle.

# Examples and explanations of incorrect results:
# image001:
Winsize=30,30 Threshold=27  Canny:100-200 Length_side=26

![image](https://github.com/EinatLasry/Triangles-CV/assets/82314695/dab51f74-9fe0-4238-aadf-6ceaca3edcaa)

Discussion: All triangles are correct. Many triangles are missing, because of the threshold and because of the defined window size. The threshold and window size can be lowered, but then wrong triangles will be inserted.

# image002:
Winsize=30,30 Threshold=2 Canny:100-200 Length_side=sqrt(68)

![image](https://github.com/EinatLasry/Triangles-CV/assets/82314695/a20b16f6-735c-4a82-adbb-d7331624a544)

Discussion: Some of the triangles were identified, but one of the cartoon animals was identified as a triangle. Sometimes there was a strong pointing outside the triangle and there the strongest center of mass point in the window was determined. In addition, there are triangles that overlap each other. Possible solutions: stronger cleaning of the image, increasing the defined window and threshold, but then the number of detected triangles will be reduced as we saw earlier.

