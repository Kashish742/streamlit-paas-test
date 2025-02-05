import subprocess
import sys

# Try installing OpenCV if it's not installed
try:
    import cv2
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python"])
    import cv2

import numpy as np
import os
import matplotlib.pyplot as plt
import streamlit as st

def detect_rectangular_box_lines(image):
    if image is None:
        raise ValueError("Image could not be loaded.")
   
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=5)
   
    if lines is None:
        raise ValueError("No lines detected in the image.")
   
    vertical_lines = []
    horizontal_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x1 - x2) < abs(y1 - y2):  # Vertical line
            vertical_lines.append((x1, y1, x2, y2))
        else:  # Horizontal line
            horizontal_lines.append((x1, y1, x2, y2))
   
    min_x, max_x = float('inf'), -float('inf')
    min_y, max_y = float('inf'), -float('inf')
   
    for line in vertical_lines:
        x1, y1, x2, y2 = line
        min_x = min(min_x, x1, x2)
        max_x = max(max_x, x1, x2)
        min_y = min(min_y, y1, y2)
        max_y = max(max_y, y1, y2)

    for line in horizontal_lines:
        x1, y1, x2, y2 = line
        min_y = min(min_y, y1, y2)
        max_y = max(max_y, y1, y2)
   
    output_image_with_lines = image.copy()
    for line in vertical_lines + horizontal_lines:
        x1, y1, x2, y2 = line
        cv2.line(output_image_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cropped_image = image[min_y:max_y, min_x:max_x]
   
    return cropped_image, min_x, min_y, max_x, max_y, output_image_with_lines

def find_black_squares_in_cropped_image(cropped_image):
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 50, 250, cv2.THRESH_BINARY_INV)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    min_y, max_y = float('inf'), -float('inf')
   
    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append((x, y, w, h))
        min_y = min(min_y, y)
        max_y = max(max_y, y + h)
   
    bounding_boxes.sort(key=lambda box: box[0])
    graph_height = max_y - min_y
    normalized_y_positions = []
    output_image = cropped_image.copy()

    for i, box in enumerate(bounding_boxes):
        x, y, w, h = box
        if w > 5 and h > 5:  # Adjust threshold values
            y_center = y + h // 2
            normalized_y = 1 - ((y_center - min_y) / graph_height)
            tolerance = 0.05
            if not (0.5 - tolerance < normalized_y < 0.5 + tolerance):
                normalized_y_positions.append(normalized_y)
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            bin_label = f"Bin {i + 1}"
   
    return normalized_y_positions, output_image

def check_efficiency_with_bin_info(y_positions, efficiency_threshold, total_inefficiencies_count):
    bin_statuses = []
    failed_bins = []
    inefficiency_count = 0  

    for i, y in enumerate(y_positions):
        inefficiency = 1 - y  

        if y < efficiency_threshold:
            bin_status = f"Bin {i + 1}: Below threshold (y = {y:.2f})"
            failed_bins.append(i + 1)
            inefficiency_count += 1  
            total_inefficiencies_count[i] += 1  
        else:
            bin_status = f"Bin {i + 1}: Above threshold (y = {y:.2f})"
       
        bin_statuses.append(bin_status)
   
    for status in bin_statuses:
        st.write(status)
   
    if failed_bins:
        st.write("\nBins below the threshold:", ', '.join([f"Bin {bin}" for bin in failed_bins]))
    else:
        st.write("\nAll bins are above the threshold!")

    if inefficiency_count > 3:
        st.write("Inefficient")

    return inefficiency_count > 3

def main():
    st.title("Image Processing App")
    st.write("Upload an image to detect rectangular boxes and black squares.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        st.image(image, caption="Uploaded Image", use_column_width=True)

        cropped_image, min_x, min_y, max_x, max_y, output_image_with_lines = detect_rectangular_box_lines(image)
        st.image(output_image_with_lines, caption="Image with Detected Lines", use_column_width=True)

        y_positions, output_image = find_black_squares_in_cropped_image(cropped_image)
        st.image(output_image, caption="Cropped Image with Black Squares", use_column_width=True)

        efficiency_threshold = st.slider("Efficiency Threshold", 0.0, 1.0, 0.8)
        total_inefficiencies_count = [0] * len(y_positions)
        if check_efficiency_with_bin_info(y_positions, efficiency_threshold, total_inefficiencies_count):
            st.write("Inefficient Image Detected!")

if __name__ == "__main__":
    main()
