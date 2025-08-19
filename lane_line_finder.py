import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import pandas as pd

def remove_small_components(binary, min_area):
    """Remove small white regions from a binary mask."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    output = np.zeros_like(binary)
    for i in range(1, num_labels):  # skip background (0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            output[labels == i] = 255
    return output

def compare_slopes(s1, s2, threshold):
    if abs(s2) > 1:
        if 1 / threshold <= s2 / s1 <= threshold:
            return True
        else:
            return False
    else:
        if abs(s2 - s1) <= threshold - 1:
            return True
        else:
            return False
image = cv2.imread('90000_w.png')
y_thresh = 400
#''' Pre-processing
# Read in the image
gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

# Define a kernel size and apply Gaussian smoothing
kernel_size = 5
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
gray_thresh = 214
binary = blur_gray

for i in range(len(binary)):
    for j in range(len(binary[i])):
        if i < y_thresh - 10:
            binary[i][j] = 0
        elif binary[i][j] < gray_thresh:
            binary[i][j] = 0
        else:
            binary[i][j] = 255
cv2.imwrite('bw.png', blur_gray)
#'''

#''' Removing white specks and filling black holes
# Step 1: Remove small white specks
#binary = cv2.imread("bw.png", cv2.IMREAD_GRAYSCALE)
cleaned_white = remove_small_components(binary, min_area=100)

# Step 2: Remove small black holes (invert, clean, invert back)
inverted = cv2.bitwise_not(cleaned_white)
cleaned_black = remove_small_components(inverted, min_area=1000)
final_result = cv2.bitwise_not(cleaned_black)
cv2.imwrite('final.png', final_result)
#'''

#''' Canny edge detection
low_thresh = 1
high_thresh = 254
edges = cv2.Canny(final_result, low_thresh, high_thresh)
cv2.imwrite('edge.png', edges)
#'''

#''' Centerline Finding
#edges = cv2.imread('edge.png', cv2.IMREAD_GRAYSCALE)
img_height = edges.shape[0]
img_width  = edges.shape[1]
y_space = 5
y_list = list(range(y_thresh, img_height+1, y_space))
points_all = []
points_plot = []
for i in range(y_thresh - 1, img_height, y_space):
    beg = 0 # Beginning of lane line
    last_beg = 0 #end of lane line
    points = []
    points_2 = []
    for j in range(img_width):
        if edges[i][j] == 255:
            if beg == 0:
                beg = j
            else:
                if j - beg == 1:
                    continue
                elif last_beg != 0 and j - last_beg < 100:
                    points[-1] = int((j+last_beg)/2)
                    points_2[-1] = (int((j+last_beg)/2), i)
                else: # Line found
                    points.append(int((j+beg)/2))
                    points_2.append((int((j+beg)/2), i)) 
                    last_beg = beg
                    beg = 0
                    continue
        if j - beg > 100 and beg != 0:
            beg = 0
            last_beg = 0
    points_all.append(points)
    points_plot.append(points_2)

#print(points_all)
#image = cv2.imread('90000_w.png')
for points_2 in points_plot:
    for point in points_2:
        cv2.circle(image, point, radius=5, color=(255,0,0), thickness=-1)
#cv2.imshow("Points detected", image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
cv2.imwrite('points.png', image)
#'''

#'''
lines = []
hor_thresh = 50
slope_thresh = 1.2
for i in range(len(points_all)):
    if i == 0:
        for x in points_all[i]:
            lines.append([(x, y_list[i])])
    else:
        used = [False] * len(points_all[i])
        for line in lines:
            if len(line) >= 2:
                last_x1, last_y1 = line[-1]
                last_x2 = False
                for further_back in line:
                    if 0 < last_y1 - further_back[1] <= 50:
                        last_x2, last_y2 = further_back
                        break
                #print(last_x1, last_x2, last_y1, last_y2)
                if not last_x2:
                    last_x2, last_y2 = line[-2]
                last_slope = (last_x2 - last_x1) / (last_y2 - last_y1)
                pred_x = last_x1 + last_slope * (y_list[i] - last_y1)
            else:
                pred_x = line[-1][0]
            dists = [abs(x - pred_x) for x in points_all[i]]
            if dists:
                idx = int(np.argmin(dists))
                if not used[idx] and dists[idx] <= hor_thresh:  # assign point if not already used
                    if len(line) >= 2:
                        slope_cur = (points_all[i][idx] - last_x1) / (y_list[i] - last_y1)
                        if compare_slopes(last_slope, slope_cur, slope_thresh):
                            line.append((points_all[i][idx], y_list[i]))
                            used[idx] = True
                        else:
                            used[idx] = True
                    else:
                        line.append((points_all[i][idx], y_list[i]))
                        used[idx] = True
        for j in range(len(used)):
            if not used[j]:
                lines.append([(points_all[i][j], y_list[i])])

for i in range(len(lines)-1):
    for j in range(i, len(lines)):
        if lines[j][0][0] < lines[i][0][0]:
            temp = lines[i]
            lines[i] = lines[j]
            lines[j] = temp

for line in lines:
    if line[0][1] > y_thresh:
        x1, y1 = line[0]
        x2 = False
        for j in range(len(line)):
            if line[j][1] - y1 > 50:
                x2, y2 = line[j-1]
        if not x2:
            x2, y2 = line[1]
        x0 = int(x1 - (x2 - x1) / (y2 - y1) * (y1-y_thresh))
        line.insert(0, (x0, y_thresh))
    if line[-1][1] < img_width:
        x1, y1 = line[-1]
        for j in range(len(line)):
            if line[j][1] - y1 < 50:
                x2, y2 = line[j]
                break
        x0 = int(x1 + (x2 - x1) / (y2 - y1) * (img_width-y1))
        y0 = img_width
        if x0 < 0:
            x0 = 0
            y0 = int(y1 - (y2 - y1) / (x2 - x1) * x1)
        elif x0 > img_width:
            x0 = img_width
            y0 = int(y1 + (y2 - y1) / (x2 - x1) * (img_width-x1))
        line.append((x0, y0))

lines_clean = []
for i in range(len(lines)):
    if len(lines[i]) > 5:
        lines_clean.append(lines[i])

lines_out = np.array([[None]*(len(lines_clean)*2)]*len(y_list))
headers = []
for i in range(len(lines_clean)):
    headers.append(f'Line {i+1} X')
    headers.append(f'Line {i+1} Y')
    for j in range(len(lines_clean[i])):
        lines_out[j][i*2:(i+1)*2] = lines_clean[i][j]
        #lines_out[j][i] = lines[i][j]

df1 = pd.DataFrame(lines_out)
df1.to_excel('lane_lines.xlsx', index=False, header=headers)


for i in range(len(lines_clean)):
    for j in range(len(lines_clean[i])-1):
        cv2.line(image, lines_clean[i][j], lines_clean[i][j+1],(0,i/len(lines_clean)*255,0),4)
#cv2.imshow("lines", image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
cv2.imwrite('lane_lines.png', image)
#'''
