# **Edge Detector Comparison using Pratts Figure of Merit**

# Introduction

Edge detectors are image processing techniques used to identify and
locate boundaries or changes in intensity within an image. They are used
to highlight features of interest, such as object boundaries or texture
changes, by producing a binary or gray-scale representation of the image
that emphasizes edges. This information can then be used for a variety
of applications, such as object recognition, image segmentation, and
feature extraction. Edge detection is a crucial step in many computer
vision and image processing algorithms and is used in a wide range of
fields, including robotics, surveillance, medical imaging, and more.

Each edge detector operates based on different assumptions and
mathematical models, resulting in different strengths and weaknesses for
different types of images and applications.

## Subject

The objective of this report is to perform a comparative analysis of
edge detectors using Pratt’s figure of merit. Pratt’s figure of merit is
a quantitative measure used to evaluate the performance of edge
detectors in terms of both detection accuracy and localization error.
The report aims to compare different edge detectors, including Canny,
Sobel, prewitt, Roberts and Gaussian, using Pratt’s figure of merit. The
results of this comparison will provide insights into the strengths and
weaknesses of each edge detector, as well as their suitability for
different image processing tasks. Ultimately, this report aims to
provide guidance on the selection of the most appropriate edge detector
for a particular application, based on the desired balance between
detection accuracy and localization error.


# Background

## Pratt’s Figure of Merit

Pratt’s figure of merit is a metric used to compare different edge
detectors. It is a quantitative measure that evaluates the performance
of edge detectors in terms of both detection accuracy and localization
error.  
The figure of merit is calculated by taking into account both the number
of true edge pixels detected and the localization error, which is
defined as the distance between the detected edge pixels and the true
edge pixels in the image. A higher value of Pratt’s figure of merit
indicates better performance, as it indicates that a higher proportion
of true edges have been detected with minimal localization error.  
Pratt’s figure of merit is a useful tool for comparing different edge
detectors, as it provides a comprehensive evaluation of the performance
of each detector, taking into account both detection accuracy and
localization error. This allows for a more comprehensive comparison of
edge detectors and provides valuable insights into the strengths and
weaknesses of each method, which can inform the selection of the most
appropriate edge detector for a particular application.  
Mathematically, it is given by:
$$R = \frac{1}{Max(N_i,N_a)}\sum\_{k=1}^{N_a}\frac{1}{1 + md^2(k)}$$
Where:

-   *N*<sub>*I*</sub> is the number of actual edges

-   *N*<sub>*A*</sub> is the number of detected edges

-   *m* is a scaling constant set to $\frac{1}{9}$.

-   *d*(*k*) denotes the distance from the actual edge to the
    corresponding detected edge


# Methodology

In order to compare the performance of various edge detectors, a Python
implementation was carried out using the built-in functions of the
OpenCV library. To facilitate the comparison, a binary image was created
using the Numpy library, with half the image being black and half white.
The purpose of this was to ensure that the image had a clear edge
between the black and white parts for easier analysis.  
Additionally, Gaussian noise was added to the image to simulate
real-world conditions. The Canny, Sobel, Prewitt, Roberts, and Gaussian
edge detectors were then applied to the image. The results of each
detector were subsequently evaluated using Pratt’s Model of Figure,
which provided a numerical score between 0 and 1, with a score closer to
1 indicating better performance.  
   
The code and results of the analysis are presented in the following
chapter.

# Development and Results

## Image Creation

I first created an Image which is half black and half white using Numpy:

``` python
def create_image():
    # create an 100x100 array of zeros with uint8 data type
    img = np.zeros((100, 100), dtype=np.uint8)
    
    # set all elements in the first 50 columns of the array to 255
    img[:, :50] = 255
    
    # display the image using matplotlib, with a 'gray' colormap
    plt.imshow(img, cmap='gray')
    plt.title("Original Image")
    plt.show()
    
    # return the original image
    return img
```

The image created is as following:

<figure id="fig:Orignal_Image">
<img src="Images/Original Image.png" />
<figcaption>Original Image</figcaption>
</figure>

## Adding noise to the image

Then I added some Gaussian noise to the image:

``` python
    # Function to add Gaussian noise to an image
def add_gaussian_noise(image, mean=0, sigma=1):
    # Generate Gaussian noise with the specified mean and standard deviation
    noise = np.random.normal(mean, sigma, image.shape)
    # Add the noise to the image
    noisy_image = image + noise
    # Clip the values in the noisy image to be within [0, 255]
    noisy_image = np.clip(noisy_image, 0, 255)
    # Return the noisy image as an 8-bit unsigned integer array
    return noisy_image.astype(np.uint8)
```

The resulted image is:

<figure id="fig:Noisy Image">
<img src="Images/noisy.png" />
<figcaption>Noisy Image</figcaption>
</figure>

## Edge Detectors

Next, I defined various edge detectors and passed the noisy image
through them:

``` python
        # Function to apply the Canny edge detector to an image
def canny_detector(img):
    # Apply Canny edge detection to the input image, 
    #with the lower and upper threshold values
    #set to 50 and 150, respectively
    
    canny = cv2.Canny(img, 50, 150)
    # Return the result
    return canny

# Function to apply the Sobel edge detector to an image
def sobel_detector(img):
    # Compute the gradient in the x direction 
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    # Compute the gradient in the y direction 
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    # Calculate the magnitude of the gradient
    sobel = np.sqrt(np.square(sobelx) + np.square(sobely))
    # Threshold the magnitude
    #to keep only values
    #greater than 150 and set the rest to 0
    sobel = np.where(sobel > 150, 255, 0)
    # Return the result
    return sobel

# Function to apply the Prewitt edge detector to an image
def prewitt_detector(img):
    # Compute the gradient in the x direction 
    prewittx = cv2.Sobel(img, cv2.CV_64F, 1, 0,
                         ksize=3, scale=1, delta=0,
                         borderType=cv2.BORDER_DEFAULT)
    # Compute the gradient in the y direction 
    prewitty = cv2.Sobel(img, cv2.CV_64F, 0, 1,
                        ksize=3, scale=1, delta=0, 
                        borderType=cv2.BORDER_DEFAULT)
    # Calculate the magnitude of the gradient
    prewitt = np.sqrt(np.square(prewittx) + np.square(prewitty))
    # Threshold the magnitude
    prewitt = np.where(prewitt > 150, 255, 0)
    # Return the result
    return prewitt


def roberts_detector(img):
    # Apply the Roberts cross edge detection
    # The Sobel function calculates the gradient of the image intensity
    # The "robertsx" is the gradient in the x direction
    robertsx = cv2.Sobel(img, cv2.CV_64F, 1, 0,
                        ksize=3, scale=1, delta=0, 
                        borderType=cv2.BORDER_DEFAULT)
    # The "robertsy" is the gradient in the y direction
    robertsy = cv2.Sobel(img, cv2.CV_64F, 0, 1, 
                        size=3,scale=1, delta=0, 
                        borderType=cv2.BORDER_DEFAULT)
    # Combine the x and y gradients
    roberts = np.abs(robertsx) + np.abs(robertsy)
    # Threshold the result 
    roberts = np.where(roberts > 150, 255, 0)
    # Return the resulting image
    return roberts

# Define a function "Gaussian" that takes an image as input
def GaussianBlur(img):
    GaussianBlur_ = cv2.GaussianBlur(img, (3,3), 0)
    # Return the blurred image
    return GaussianBlur_
```

The Resulting images from the edge detectors are as following:

<figure id="fig:Edges">
<img src="Images/Edge detection.png" />
<figcaption>Edge Detection</figcaption>
</figure>

## Prett’s Figure of Merit

Next, I pass the above attained images into the Prett’s Figure of Merit
function:

``` python

DEFAULT_ALPHA = 1.0 / 9

def fom(img, original, alpha = DEFAULT_ALPHA):
    """
    This function calculates Pratt's Figure of Merit (FOM)
    for a given image "img".
    """
    
    # Compute the distance transform for the original image
    # using Euclidean distance transform
    dist = distance_transform_edt(np.invert(original))

    # Initialize FOM with a value proportional
    #to the inverse of the maximum 
    # between the number of non-zero pixels in "img" and "original"
    fom = 1.0 / np.maximum(
        np.count_nonzero(img),
        np.count_nonzero(original))

    # Get the shape of the image
    N, M = img.shape

    # Loop through all the pixels in the image
    for i in range(0, N):
        for j in range(0, M):
            # If the current pixel has a non-zero value
            if (img[i, j]).any():
                fom += 1.0 / ( 1.0 + dist[i, j] * dist[i, j] * alpha)

    # Normalize the FOM by
    # dividing it with the maximum between the number of 
    # non-zero pixels in "img" and "original"
    fom /= np.maximum(
        np.count_nonzero(img),
        np.count_nonzero(original))    

    # Return the calculated FOM
    return fom
```

## Results

The results that are I get are:

|     Edge detection algorithm      | PFOM |
|:---------------------------------:|:----:|
|  Sobel edge detection algorithm   | 0.66 |
| Prewitt edge detection algorithm  | 0.62 |
|  Robert edge detection algorithm  | 0.66 |
| Gaussian edge detection algorithm | 0.70 |
|  Canny edge detection algorithm   | 0.76 |

Based on the results, it can be concluded that the Canny algorithm
produced the best outcome. This is attributed to its use of dual
thresholding and hysteresis, which effectively reduce edge detection
errors. Additionally, the results for Sobel, Prewitt, and Roberts
algorithms were found to be similar.  
   
Edge detection algorithms, such as Canny, Sobel, Roberts, Prewitt, and
Gaussian, have different strengths and weaknesses. When choosing an edge
detection algorithm, it is important to consider the type of image, the
desired output, and the computational resources available.  
Canny is considered to be the most advanced and robust algorithm,
providing accurate results with minimum errors. It is often used in
tasks such as object recognition, image segmentation, and edge-based
registration. On the other hand, Sobel, Prewitt, and Roberts are simpler
and faster algorithms, and are commonly used in real-time video
processing and robot navigation where speed is more important than
accuracy. Gaussian Edge Detection uses a Gaussian filter to smooth the
image before applying the edge detection algorithm, and is well suited
for images with high levels of noise and small variations, such as
medical imaging and satellite imagery.  
It is important to note that the choice of algorithm will depend on the
specific requirements and trade-offs of the application at hand. In some
cases, pre-processing steps may be required to enhance weak edges or
reduce the effect of noise in the image before edge detection.

# Conclusion

In conclusion, the comparison of Canny, Sobel, Roberts, Prewitt, and
Gaussian edge detectors using the Pratt’s Figure of Merit showed that
Canny performed the best, while Sobel, Roberts, and Prewitt performed
similarly. The comparison was carried out using Python 3 and a
self-created image which was half white and half black. These results
suggest that Canny provides the most accurate results in terms of
detecting edges with high precision and minimum errors, while Sobel,
Roberts, and Prewitt may be more suitable for applications that require
faster processing times.
