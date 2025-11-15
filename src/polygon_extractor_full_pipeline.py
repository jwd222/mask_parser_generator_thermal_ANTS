import cv2
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt

from sam2_wrapper_auto import SAM2Wrapper, SAM2PATH

# =================== GLOBAL INITIALIZATON OF SAM2 ====================
# Create the SAM2 Wrapper instance
config_path = Path(SAM2PATH / "sam2/configs/sam2.1/sam2.1_hiera_l.yaml")
checkpoint_path = Path(SAM2PATH / "checkpoints/sam2.1_hiera_large.pt")

sam = SAM2Wrapper(
    checkpoint_path=checkpoint_path,
    config_path=config_path,
)

# ==================== HELPER FUNCTIONS ====================

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

def prune_internal_branches(skeleton, aggressive=True):
    """
    Remove branches that end inside the image (not on edges) - recursively
    If aggressive=True, also removes disconnected components not touching edges
    """
    h, w = skeleton.shape
    pruned = skeleton.copy()
    
    # First pass: Standard branch pruning
    max_iterations = 200
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
    
    # Second pass: Remove completely disconnected components that don't touch edges
    if aggressive:
        # Find all connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pruned, connectivity=8)
        
        # Identify edge-connected components
        edge_components = set()
        
        # Check all four edges
        for x in range(w):
            if pruned[0, x] > 0:
                edge_components.add(labels[0, x])
            if pruned[h-1, x] > 0:
                edge_components.add(labels[h-1, x])
        
        for y in range(h):
            if pruned[y, 0] > 0:
                edge_components.add(labels[y, 0])
            if pruned[y, w-1] > 0:
                edge_components.add(labels[y, w-1])
        
        edge_components.discard(0)  # Remove background
        
        # Keep only edge-connected components
        final_pruned = np.zeros_like(pruned)
        for component_id in edge_components:
            final_pruned[labels == component_id] = 255
        
        pruned = final_pruned
    
    return pruned

def get_edge_connected_components(binary_img):
    """Find all connected components that touch the image edges."""
    h, w = binary_img.shape
    
    # Connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_img, connectivity=8
    )
    
    edge_components = set()

    # Top & bottom edges
    for x in range(w):
        if binary_img[0, x] > 0:
            edge_components.add(labels[0, x])
        if binary_img[h-1, x] > 0:
            edge_components.add(labels[h-1, x])

    # Left & right edges
    for y in range(h):
        if binary_img[y, 0] > 0:
            edge_components.add(labels[y, 0])
        if binary_img[y, w-1] > 0:
            edge_components.add(labels[y, w-1])

    edge_components.discard(0)  # remove background
    
    return edge_components, labels, num_labels, stats

def get_largest_connected_component(stats):
    """Return label of largest component (excluding background)."""
    if stats.shape[0] <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]   # ignore background (index 0)
    largest = 1 + areas.argmax()
    return largest

# ==================== CORE PROCESSING ====================

def process_patch(patch, debug=True):
    """Process a single patch to extract boundaries - using edge-connected components"""
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if len(patch.shape) == 3 else patch
    
    # Threshold at 150
    _, binary_130 = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
    
    # Erode to separate text close to boundaries (1-2 pixel gap)
    kernel_erode = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(binary_130, kernel_erode, iterations=1)
    # eroded = cv2.erode(eroded, kernel_erode, iterations=1)
    
    if debug:
        # View Eroded image for debugging
        plt.imshow(eroded, cmap='gray')
        plt.show()
        
    # Find edge-connected components and full CC metadata
    edge_components, labels, num_labels, stats = get_edge_connected_components(eroded)

    # Also get largest CC
    largest_cc = get_largest_connected_component(stats)

    # Merge: edge-connected + largest
    components_to_keep = set(edge_components)
    if largest_cc is not None:
        components_to_keep.add(largest_cc)

    # Create merged mask
    merged_mask = np.zeros_like(eroded)
    for comp_id in components_to_keep:
        merged_mask[labels == comp_id] = 255
        
    if debug:
        # View merged mask for debugging
        plt.imshow(merged_mask, cmap='gray')
        plt.show()

    # Dilate back to restore thickness
    dilated = cv2.dilate(merged_mask, kernel_erode, iterations=1)
    
    # Optional: Close small gaps
    kernel_close = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    
    # Thin to single pixel width
    thinned = cv2.ximgproc.thinning(cleaned)
    
    # Prune internal branches
    pruned = prune_internal_branches(thinned)
    
    return pruned

def segment_patch(boundary_patch):
    """
    Dummy segmentation function - to be implemented later
    Input: boundary image (single channel)
    Output: segmented image (with labeled regions)
    """
    sam.image = boundary_patch

    # Generate masks for the current image
    segmemted_mask = sam.generate_automatic_masks()
    all_masks = sam.masks_to_binary(segmemted_mask)
    return all_masks


def remove_disconnected_components(merged_image, min_size=100, connectivity=8):
    """
    Remove small disconnected components from merged image
    """
    # Ensure binary image
    if merged_image.dtype != np.uint8:
        merged_image = merged_image.astype(np.uint8)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        merged_image, connectivity=connectivity
    )
    
    # Create mask to keep only components above minimum size
    cleaned = np.zeros_like(merged_image)
    
    for label in range(1, num_labels):  # Skip background (label 0)
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_size:
            cleaned[labels == label] = 255
    
    return cleaned

def remove_edge_connected_components_only(merged_image, connectivity=8):
    """
    Keep only components that touch the image edges
    """
    if merged_image.dtype != np.uint8:
        merged_image = merged_image.astype(np.uint8)
    
    h, w = merged_image.shape
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        merged_image, connectivity=connectivity
    )
    
    # Find components that touch image edges
    edge_components = set()
    
    # Check all four edges
    for x in range(w):
        if merged_image[0, x] > 0:
            edge_components.add(labels[0, x])
        if merged_image[h-1, x] > 0:
            edge_components.add(labels[h-1, x])
    
    for y in range(h):
        if merged_image[y, 0] > 0:
            edge_components.add(labels[y, 0])
        if merged_image[y, w-1] > 0:
            edge_components.add(labels[y, w-1])
    
    # Remove background label (0)
    edge_components.discard(0)
    
    # Keep only edge-connected components
    cleaned = np.zeros_like(merged_image)
    for component_id in edge_components:
        cleaned[labels == component_id] = 255
    
    return cleaned

def remove_small_components_and_clean(merged_image, min_size=50, keep_only_edge_connected=True):
    """
    Comprehensive cleaning: remove small components and optionally keep only edge-connected ones
    """
    # First remove very small components
    cleaned = remove_disconnected_components(merged_image, min_size=min_size)
    
    # Optionally keep only edge-connected components
    if keep_only_edge_connected:
        cleaned = remove_edge_connected_components_only(cleaned)
    
    return cleaned

def prune_merged_image(merged_image, min_component_size=100):
    """
    Comprehensive cleaning for merged image:
    1. Remove small disconnected components
    2. Prune internal branches (like we do for patches)
    3. Optional: Keep only edge-connected components
    """
    # Ensure binary image
    if merged_image.dtype != np.uint8:
        merged_image = merged_image.astype(np.uint8)
    
    # Step 1: Remove very small components first
    cleaned = remove_disconnected_components(merged_image, min_size=min_component_size)
    
    # Step 2: Apply the same branch pruning we use for patches
    pruned = prune_internal_branches(cleaned)
    
    # Step 3: Optional - remove any small components that might have been created during pruning
    final_cleaned = remove_disconnected_components(pruned, min_size=min_component_size//2)
    
    return final_cleaned

def prune_merged_image_keep_edge_connected(merged_image, min_component_size=100):
    """
    Alternative: Keep only edge-connected components after pruning
    """
    # Ensure binary image
    if merged_image.dtype != np.uint8:
        merged_image = merged_image.astype(np.uint8)
    
    # Step 1: Remove very small components first
    cleaned = remove_disconnected_components(merged_image, min_size=min_component_size)
    
    # Step 2: Apply branch pruning
    pruned = prune_internal_branches(cleaned)
    
    # Step 3: Keep only edge-connected components
    final_cleaned = remove_edge_connected_components_only(pruned)
    
    # Step 4: One final branch pruning to clean up any new internal branches
    final_pruned = prune_internal_branches(final_cleaned)
    
    return final_pruned
# ==================== PATCH EXTRACTION & MERGING ====================

def extract_patches(image, patch_size=512, overlap=0.5):
    """
    Extract overlapping patches from image
    
    Args:
        image: Input image
        patch_size: Size of each patch (height and width)
        overlap: Overlap ratio (0.5 = 50% overlap)
    
    Returns:
        patches: List of image patches
        positions: List of (y, x, h, w) for each patch
    """
    h, w = image.shape[:2]
    patches = []
    positions = []
    
    # Calculate step size based on overlap
    step = int(patch_size * (1 - overlap))
    
    for y in range(0, h, step):
        for x in range(0, w, step):
            y_end = min(y + patch_size, h)
            x_end = min(x + patch_size, w)
            
            # Adjust start position if we're at the edge to maintain patch_size
            y_start = y_end - patch_size if y_end == h and h - y < patch_size else y
            x_start = x_end - patch_size if x_end == w and w - x < patch_size else x
            
            # Ensure we don't go negative
            y_start = max(0, y_start)
            x_start = max(0, x_start)
            
            patch = image[y_start:y_end, x_start:x_end]
            
            # Pad if needed (only for very small images)
            if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                if len(image.shape) == 3:
                    padded = np.ones((patch_size, patch_size, image.shape[2]), dtype=image.dtype) * 255
                else:
                    padded = np.ones((patch_size, patch_size), dtype=image.dtype) * 255
                padded[:patch.shape[0], :patch.shape[1]] = patch
                patch = padded
            
            patches.append(patch)
            positions.append((y_start, x_start, patch.shape[0], patch.shape[1]))
    
    return patches, positions

def merge_patches(patches, positions, original_shape, patch_size=512):
    """
    Merge overlapping patches back to original shape using averaging in overlap regions
    
    Args:
        patches: List of processed patches
        positions: List of (y, x, h, w) for each patch
        original_shape: Shape of original image
        patch_size: Size of patches
    
    Returns:
        merged: Merged image
    """
    merged = np.zeros(original_shape[:2], dtype=np.float32)
    count = np.zeros(original_shape[:2], dtype=np.float32)
    
    for patch, (y, x, h, w) in zip(patches, positions):
        # Add patch values
        merged[y:y+h, x:x+w] += patch[:h, :w].astype(np.float32)
        count[y:y+h, x:x+w] += 1
    
    # Average overlapping regions
    count[count == 0] = 1  # Avoid division by zero
    merged = merged / count
    
    # Convert back to uint8 and threshold for binary images
    merged = (merged > 127).astype(np.uint8) * 255
    
    return merged

def analyze_components(merged_image, cleaned_image):
    """Analyze what components were removed"""
    original_components = cv2.connectedComponentsWithStats(merged_image)[0] - 1  # exclude background
    cleaned_components = cv2.connectedComponentsWithStats(cleaned_image)[0] - 1
    
    print(f"Component analysis:")
    print(f"  Original: {original_components} components")
    print(f"  After cleaning: {cleaned_components} components")
    print(f"  Removed: {original_components - cleaned_components} components")
    
    # Show removed components (for debugging)
    removed = merged_image - cleaned_image
    removed[removed < 0] = 0  # Remove negative values
    
    return removed
# ==================== MAIN PIPELINE ====================

def process_image_pipeline(input_path, output_dir, patch_size=512, final_cleanup=True, cleanup_mode='prune_only'):
    """
    Main pipeline: 
    1. Load image
    2. Extract patches
    3. Process each patch (boundary extraction)
    4. Segment each patch (dummy for now)
    5. Merge patches back
    """
    
    # Setup directories
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    patches_dir = output_dir / "patches"
    patches_dir.mkdir(exist_ok=True)
    
    boundaries_dir = output_dir / "boundaries"
    boundaries_dir.mkdir(exist_ok=True)
    
    segmented_dir = output_dir / "segmented"
    segmented_dir.mkdir(exist_ok=True)
    
    # Load image
    print(f"Loading image: {input_path}")
    image = cv2.imread(str(input_path))
    original_shape = image.shape
    
    # Extract patches
    print(f"Extracting {patch_size}x{patch_size} patches...")
    patches, positions = extract_patches(image, patch_size)
    print(f"Total patches: {len(patches)}")
    
    # Save patch info
    patch_info = {
        "original_shape": original_shape,
        "patch_size": patch_size,
        "num_patches": len(patches),
        "positions": positions
    }
    with open(output_dir / "patch_info.json", "w") as f:
        json.dump(patch_info, f, indent=2)
    
    # Process each patch
    processed_boundaries = []
    processed_segmented = []
    
    for i, (patch, pos) in enumerate(zip(patches, positions)):
        print(f"Processing patch {i+1}/{len(patches)}...")
        
        # Save original patch
        cv2.imwrite(str(patches_dir / f"patch_{i:04d}.png"), patch)
        
        # Process boundary extraction
        boundary = process_patch(patch)
        processed_boundaries.append(boundary)
        cv2.imwrite(str(boundaries_dir / f"boundary_{i:04d}.png"), boundary)
        
        # Segment (dummy for now)
        segmented = segment_patch(boundary)
        processed_segmented.append(segmented)
        cv2.imwrite(str(segmented_dir / f"segmented_{i:04d}.png"), segmented.astype(np.uint8))
    
    # Merge patches
    print("Merging boundaries...")
    merged_boundaries = merge_patches(processed_boundaries, positions, original_shape, patch_size)
    
    # Apply final comprehensive cleanup
    if final_cleanup:
        print("Applying comprehensive cleanup to merged image...")
        
        if cleanup_mode == 'prune_only':
            merged_boundaries = prune_merged_image(merged_boundaries, min_component_size=100)
            print("  - Removed small components and pruned internal branches")
            
        elif cleanup_mode == 'edge_connected':
            merged_boundaries = prune_merged_image_keep_edge_connected(merged_boundaries, min_component_size=30)
            print("  - Kept only edge-connected components after pruning")
        
        print("  Cleanup complete!")
    
    cv2.imwrite(str(output_dir / "merged_boundaries.png"), merged_boundaries)    
    print("Merging segmented patches...")
    merged_segmented = merge_patches(processed_segmented, positions, original_shape, patch_size)
    cv2.imwrite(str(output_dir / "merged_segmented.png"), merged_segmented.astype(np.uint8))
    
    print(f"\nProcessing complete! Output saved to: {output_dir}")
    print(f"- Patches: {patches_dir}")
    print(f"- Boundaries: {boundaries_dir}")
    print(f"- Segmented: {segmented_dir}")
    print(f"- Merged boundaries: {output_dir / 'merged_boundaries.png'}")
    print(f"- Merged segmented: {output_dir / 'merged_segmented.png'}")
    
    return merged_boundaries, merged_segmented


def visualize_pruning_effect(original, pruned, output_path):
    """Create a visualization showing what was pruned"""
    # Create color visualization
    if len(original.shape) == 2:
        original_bgr = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        pruned_bgr = cv2.cvtColor(pruned, cv2.COLOR_GRAY2BGR)
    else:
        original_bgr = original
        pruned_bgr = pruned
    
    # Find differences
    removed = original - pruned
    removed[removed < 0] = 0
    
    # Create visualization: green=kept, red=removed
    visualization = pruned_bgr.copy()
    visualization[removed > 0] = [0, 0, 255]  # Red for removed parts
    
    # Add some kept parts in green for contrast
    visualization[pruned > 0] = [0, 255, 0]  # Green for kept parts
    
    cv2.imwrite(str(output_path), visualization)
    return visualization
# ==================== USAGE ====================

if __name__ == "__main__":
    input_image = "D:\\S\\ANTS\\Data\\BADC\\Data\\mouza_maps\\2\\rangpur_mougach_sheet_5_cleaned_cropped.jpg"
    output_directory = "badc_mouza_map_output"
    patch_size = 512  # or 256
    
    merged_boundaries, merged_segmented = process_image_pipeline(
        input_image, 
        output_directory, 
        patch_size=patch_size,
        final_cleanup=True,
        cleanup_mode='edge_connected'
    )
    
    # Optional: Analyze what was removed
    original_merged = cv2.imread(str(Path(output_directory) / "merged_boundaries_before_cleanup.png"), 0)
    if original_merged is not None:
        visualization = visualize_pruning_effect(original_merged, merged_boundaries, 
                                               str(Path(output_directory) / "pruning_visualization.png"))
        print("Created pruning visualization: pruning_visualization.png")
        
        removed_components = analyze_components(original_merged, merged_boundaries)
        cv2.imwrite(str(Path(output_directory) / "removed_components.png"), removed_components)
