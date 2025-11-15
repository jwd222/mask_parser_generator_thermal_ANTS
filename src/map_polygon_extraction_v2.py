import cv2
import numpy as np
from matplotlib import pyplot as plt
from collections import deque

def get_neighbors(y, x, h, w):
    """Get 8-connected neighbors"""
    neighbors = []
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                neighbors.append((ny, nx))
    return neighbors

def is_edge_pixel(y, x, h, w):
    """Check if pixel is on image edge"""
    return y == 0 or y == h-1 or x == 0 or x == w-1

def find_branch_endpoints(skeleton):
    """Find all endpoints and junction points in skeleton"""
    h, w = skeleton.shape
    endpoints = []
    junctions = []
    
    for y in range(h):
        for x in range(w):
            if skeleton[y, x] > 0:
                neighbors = get_neighbors(y, x, h, w)
                neighbor_count = sum(1 for ny, nx in neighbors if skeleton[ny, nx] > 0)
                
                if neighbor_count == 1:
                    endpoints.append((y, x))
                elif neighbor_count > 2:
                    junctions.append((y, x))
    
    return endpoints, junctions

def trace_branch(skeleton, start, visited):
    """Trace a branch from endpoint to junction or edge"""
    h, w = skeleton.shape
    path = [start]
    current = start
    visited.add(current)
    
    while True:
        y, x = current
        neighbors = get_neighbors(y, x, h, w)
        next_pixels = [
            (ny, nx) for ny, nx in neighbors 
            if skeleton[ny, nx] > 0 and (ny, nx) not in visited
        ]
        
        if not next_pixels:
            break
            
        current = next_pixels[0]
        path.append(current)
        visited.add(current)
        
        # Check if we hit a junction (more than 2 neighbors)
        neighbors = get_neighbors(current[0], current[1], h, w)
        neighbor_count = sum(1 for ny, nx in neighbors if skeleton[ny, nx] > 0)
        if neighbor_count > 2:
            break
    
    return path

def prune_internal_branches(skeleton):
    """Remove branches that end inside the image (not on edges) - recursively"""
    h, w = skeleton.shape
    pruned = skeleton.copy()
    
    max_iterations = 100
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        changed = False
        
        endpoints, junctions = find_branch_endpoints(pruned)
        
        # Filter endpoints that are NOT on image edges
        internal_endpoints = [ep for ep in endpoints if not is_edge_pixel(ep[0], ep[1], h, w)]
        
        if not internal_endpoints:
            break
        
        visited = set()
        
        for endpoint in internal_endpoints:
            if endpoint in visited:
                continue
                
            # Trace branch from this endpoint
            path = trace_branch(pruned, endpoint, visited)
            
            # Check if path ends at image edge
            last_point = path[-1]
            if not is_edge_pixel(last_point[0], last_point[1], h, w):
                # This branch ends internally, remove it
                for y, x in path:
                    pruned[y, x] = 0
                changed = True
        
        if not changed:
            break
    
    return pruned

def extract_polygon_boundaries(image_path, output_path='cleaned_boundaries.png', vis=False):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Threshold at 150
    _, binary_150 = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    # Erode to separate text close to boundaries (1-2 pixel gap)
    kernel_erode = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(binary_150, kernel_erode, iterations=1)
    
    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(eroded, connectivity=8)
    
    # Keep only the largest connected component
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    largest_component = (labels == largest_label).astype(np.uint8) * 255
    
    # Dilate back to restore original line thickness
    dilated = cv2.dilate(largest_component, kernel_erode, iterations=1)
    
    # Optional: Close small gaps
    kernel_close = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    
    # Thin to single pixel width
    thinned = cv2.ximgproc.thinning(cleaned)
    
    # Prune internal branches
    pruned = prune_internal_branches(thinned)
    
    # Save outputs
    cv2.imwrite('thinned_before_prune.png', thinned)
    cv2.imwrite(output_path, pruned)
    
    if vis:
        # Visualization
        plt.figure(figsize=(20, 5))
        
        plt.subplot(1, 5, 1)
        plt.imshow(gray, cmap='gray')
        plt.title('Original')
        plt.axis('off')
        
        plt.subplot(1, 5, 2)
        plt.imshow(binary_150, cmap='gray')
        plt.title('Threshold 150')
        plt.axis('off')
        
        plt.subplot(1, 5, 3)
        plt.imshow(largest_component, cmap='gray')
        plt.title('Largest Component')
        plt.axis('off')
        
        plt.subplot(1, 5, 4)
        plt.imshow(thinned, cmap='gray')
        plt.title('Thinned (Before Prune)')
        plt.axis('off')
        
        plt.subplot(1, 5, 5)
        plt.imshow(pruned, cmap='gray')
        plt.title('After Pruning')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('steps.png', dpi=150)
        plt.show()
    
    return pruned

if __name__ == "__main__":
    image_path = "D:\\S\\ANTS\\Data\\BADC\\Data\\mouza_maps\\2\\1.jpg"
    result = extract_polygon_boundaries(image_path, vis=False)