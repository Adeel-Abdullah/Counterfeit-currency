import cv2
import numpy as np

from IPython.display import clear_output, Image, display
from io import BytesIO
import PIL.Image
from collections import defaultdict

from skimage.filters import threshold_local
import argparse
import imutils

from itertools import combinations 
from shapely.geometry import Polygon

def showarray(a, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    f = BytesIO()
    if len(a.shape)<3:
            PIL.Image.fromarray(a).save(f, fmt)
    else:
            PIL.Image.fromarray(cv2.cvtColor(a,cv2.COLOR_RGB2BGR)).save(f, fmt)
    display(Image(data=f.getvalue()))
    

def segment_by_angle_kmeans(lines, k=2, **kwargs):
    """Groups lines based on angle with k-means.

    Uses k-means on the coordinates of the angle on the unit circle 
    to segment `k` angles inside `lines`.
    """

    # Define criteria = (type, max_iter, epsilon)
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))
    flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
    attempts = kwargs.get('attempts', 10)

    # returns angles in [0, pi] in radians
    angles = np.array([line[0][1] for line in lines])
    # multiply the angles by two and find coordinates of that angle
    pts = np.array([[np.cos(2*angle), np.sin(2*angle)]
                    for angle in angles], dtype=np.float32)

    # run kmeans on the coords
    labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]
    labels = labels.reshape(-1)  # transpose to row vec

    # segment lines based on their kmeans label
    segmented = defaultdict(list)
    for i, line in zip(range(len(lines)), lines):
        segmented[labels[i]].append(line)
    segmented = list(segmented.values())
    return segmented

def intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [[x0, y0]]

def segmented_intersections(lines):
    """Finds the intersections between groups of lines."""

    intersections = []
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i+1:]:
            for line1 in group:
                for line2 in next_group:
                    intersections.append(intersection(line1, line2)) 

    return intersections

def scan(image):
    # load the image and compute the ratio of the old height
    # to the new height, clone it, and resize it
    #image = cv2.imread(args["image"])
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = imutils.resize(image, height = 500)
    img = image.copy()

    # convert the image to grayscale, blur it, and find edges
    # in the image
    blur = cv2.bilateralFilter(image,12,150,150)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
#    gray = cv2.GaussianBlur(blur, (9, 15), 0)
    # blur = cv2.bilateralFilter(gray,9,75,75)
    # edged = cv2.Canny(gray, 50, 200,apertureSize=3,L2gradient = True)
    edged = cv2.Canny(gray,50,120,L2gradient = True)
    lines = cv2.HoughLines(edged,1,np.pi/180, 90)
    seg_lines = segment_by_angle_kmeans(lines=lines)
    for line in seg_lines[0]:
        for r,theta in line: 
            # Stores the value of cos(theta) in a 
            a = np.cos(theta)
            # Stores the value of sin(theta) in b 
            b = np.sin(theta)
            # x0 stores the value rcos(theta) 
            x0 = a*r 
            # y0 stores the value rsin(theta) 
            y0 = b*r 
            
            # x1 stores the rounded off value of (rcos(theta)-1000sin(theta)) 
            x1 = int(x0 + 1000*(-b))
            # y1 stores the rounded off value of (rsin(theta)+1000cos(theta)) 
            y1 = int(y0 + 1000*(a))
            # x2 stores the rounded off value of (rcos(theta)+1000sin(theta)) 
            x2 = int(x0 - 1000*(-b))
            # y2 stores the rounded off value of (rsin(theta)-1000cos(theta)) 
            y2 = int(y0 - 1000*(a)) 
            
            cv2.line(image,(x1,y1), (x2,y2), (0,255,255),2) 

    for line in seg_lines[1]:
        for r,theta in line: 
            # Stores the value of cos(theta) in a 
            a = np.cos(theta) 
            # Stores the value of sin(theta) in b 
            b = np.sin(theta) 
            # x0 stores the value rcos(theta) 
            x0 = a*r 
            # y0 stores the value rsin(theta) 
            y0 = b*r 
            # x1 stores the rounded off value of (rcos(theta)-1000sin(theta)) 
            x1 = int(x0 + 1000*(-b)) 
            # y1 stores the rounded off value of (rsin(theta)+1000cos(theta)) 
            y1 = int(y0 + 1000*(a)) 
            # x2 stores the rounded off value of (rcos(theta)+1000sin(theta)) 
            x2 = int(x0 - 1000*(-b)) 
            # y2 stores the rounded off value of (rsin(theta)-1000cos(theta)) 
            y2 = int(y0 - 1000*(a)) 
            cv2.line(image,(x1,y1), (x2,y2), (0,0,255),2)
            
    inters = segmented_intersections(seg_lines)
    comb = combinations(inters, 4)

    a_dict = {Polygon(np.array(i).reshape(4,2)).area: np.array(i)  for i in comb}
    largest_area = sorted(a_dict.keys(),reverse = True)[0]
    pts = cv2.convexHull(np.array(a_dict[largest_area]).reshape(4,2),returnPoints = True)
    img = cv2.polylines(img,[pts],True,(0,255,255),2,lineType=cv2.LINE_AA)
    warped = four_point_transform(orig, pts.reshape(4, 2) * ratio)
    return warped

def segment_by_color_kmeans(warped, k=5):
    pixels = np.float32(warped.reshape(-1, 3))    
    median = np.median(np.median(warped, axis=0),axis=0)
    n_colors = k
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS

    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)

    indices = np.argsort(counts)[::-1]   
    freqs = np.cumsum(np.hstack([[0], counts[indices]/counts.sum()]))
    rows = np.int_(warped.shape[0]*freqs)

    dom_patch = np.zeros(shape=warped.shape, dtype=np.uint8)
    for i in range(len(rows) - 1):
        dom_patch[rows[i]:rows[i + 1], :, :] += np.uint8(palette[indices[i]])
    
    return median,palette, dom_patch

def order_points(pts):
    
	rect = np.zeros((4, 2), dtype = "float32")
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	return rect

def four_point_transform(image, pts):
	rect = order_points(pts)
	(tl, tr, br, bl) = rect

	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

	return warped