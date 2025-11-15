import cv2
import numpy as np
from matplotlib import pyplot as plt

def extract_polygon_boundaries(image_path, output_path='cleaned_boundaries.png'):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    # Erode to separate text close to boundaries (1-2 pixel gap)
    kernel_erode = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(binary, kernel_erode, iterations=1)
    
    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(eroded, connectivity=8)
    
    # Keep only the largest connected component
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    largest_component = (labels == largest_label).astype(np.uint8) * 255
    
    # Dilate back to restore original line thickness
    dilated = cv2.dilate(largest_component, kernel_erode, iterations=1)
    
    # # Do another open-close to remove small noise
    # kernel_open = np.ones((3, 3), np.uint8)
    # cleaned_largest_component = cv2.morphologyEx(largest_component, cv2.MORPH_OPEN, kernel_open, iterations=1)
    
    # Optional: Close small gaps
    kernel_close = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw on white background
    output = np.ones_like(gray) * 255
    cv2.drawContours(output, contours, -1, (0, 0, 0), 1)
    
    cv2.imwrite(output_path, output)
    cv2.imwrite('largest_component.png', largest_component)
    # cv2.imwrite('cleaned_largest_component.png', cleaned_largest_component)
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 4, 1)
    plt.imshow(binary, cmap='gray')
    plt.title('Binary')
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.imshow(eroded, cmap='gray')
    plt.title('After Erosion')
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.imshow(largest_component, cmap='gray')
    plt.title('Largest Component')
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.imshow(output, cmap='gray')
    plt.title('Final Output')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('steps.png', dpi=150)
    plt.show()
    
    return output

if __name__ == "__main__":
    image_path = "D:\\S\\ANTS\\Data\\BADC\\Data\\mouza_maps\\2\\1.jpg"
    result = extract_polygon_boundaries(image_path)