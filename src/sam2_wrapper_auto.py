import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import sys
import os
from pathlib import Path

# Setup paths
current_dir = Path(__file__).parent.resolve()
base_dir = current_dir.parent.resolve()
thirdparty_dir = base_dir / "thirdparty"
SAM2PATH = thirdparty_dir / "sam2"
sys.path.insert(0, str(SAM2PATH))

# Sam import
from pathlib import Path

# Import SAM2 wrapper
try:
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False
    print("[WARNING] SAM2 not available. Install required packages.")

class SAM2Wrapper:
    def __init__(self, checkpoint_path, config_path, device=None):
        """
        Initialize the SAM2 model, predictor, and automatic mask generator.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # It's better to work with absolute paths
        sam_base_path = SAM2PATH
        full_checkpoint_path = (sam_base_path / checkpoint_path).resolve()
        full_config_path = (sam_base_path / config_path).resolve()
        
        print(f"Loading SAM2 model on {self.device}...")
        self.model = build_sam2(str(full_config_path), str(full_checkpoint_path), device=self.device)
        self.predictor = SAM2ImagePredictor(self.model)
        self.mask_generator = SAM2AutomaticMaskGenerator(self.model)
        self.image = None

    def set_image(self, image_path):
        """
        Load and set an image for prediction.
        """
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)
        self.predictor.set_image(image)
        self.image = image
        print(f"Image loaded: {image_path} | Shape: {image.shape}")

    # ======================= Automatic Mask Generation ======================= #
    def generate_automatic_masks(self):
        if self.image is None:
            raise ValueError("No image loaded. Please call set_image() first.")
        
        print("Generating automatic masks...")
        # Convert image of (H,W) shape into (H, W, C) where C=1 for grayscale and C=3 for RGB
        if self.image.ndim == 2:              # (H, W)
            self.image = np.stack([self.image]*3, axis=2)

        elif self.image.shape[2] == 1:        # (H, W, 1)
            self.image = np.repeat(self.image, 3, axis=2)
                
        masks = self.mask_generator.generate(self.image)
        return masks

    # ======================= Visualization and Saving Functions ======================= #
    @staticmethod
    def save_combined_mask(anns, output_path):
        """
        Combines all masks into a single image with random colors and saves it.
        """
        if not anns:
            print("No masks to save.")
            return
            
        # Sort masks by area
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        
        # Create a blank RGBA image
        h, w = sorted_anns[0]['segmentation'].shape
        combined_mask_img = np.zeros((h, w, 4), dtype=np.uint8)

        # Draw each mask with a random color
        for ann in sorted_anns:
            m = ann['segmentation']
            # Generate a random color with 50% opacity
            color = np.concatenate([np.random.randint(0, 256, 3), [128]])
            combined_mask_img[m] = color

        # Convert to PIL Image and save
        pil_img = Image.fromarray(combined_mask_img, 'RGBA')
        pil_img.save(output_path)
        print(f"Saved combined mask to {output_path}")

    @staticmethod
    def show_anns(anns, borders=True):
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)

        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
        img[:, :, 3] = 0
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.5]])
            img[m] = color_mask
            if borders:
                contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
                cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=2)

        ax.imshow(img)
    
    def show_generated_masks(self, masks):
        if self.image is None:
            raise ValueError("No image loaded. Please call set_image() first.")
        
        plt.figure(figsize=(20, 20))
        plt.imshow(self.image)
        self.show_anns(masks)
        plt.axis('off')
        plt.show()
        
    @staticmethod
    def masks_to_binary(anns, height=None, width=None):
        """
        Converts a list of SAM masks into a single binary mask.
        Output: uint8 mask with values 0 or 255.
        """
        if not anns:
            raise ValueError("No masks received from mask generator.")

        # Determine output size
        if height is None or width is None:
            height, width = anns[0]["segmentation"].shape

        # Create empty binary mask
        merged = np.zeros((height, width), dtype=np.uint8)

        # Combine all segmentations
        for ann in anns:
            seg = ann["segmentation"].astype(bool)
            merged[seg] = 255

        return merged

# ======================= Main Execution Logic ======================= #
if __name__ == "__main__":
    # --- USER: PLEASE UPDATE THESE PATHS ---
    input_folder = "D:\S\ANTS\Repo\mask_masker\badc_mouza_map_output_03"  # Folder containing your source images
    output_folder = "D:\S\ANTS\Repo\mask_masker\badc_mouza_map_output_03\merged_masks" # Folder where masks will be saved
    # -----------------------------------------

    # Create the SAM2 Wrapper instance
    config_path = Path(SAM2PATH / "sam2/configs/sam2.1/sam2.1_hiera_l.yaml")
    checkpoint_path = Path(SAM2PATH / "checkpoints/sam2.1_hiera_large.pt")

    sam = SAM2Wrapper(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
    )

    # Create the output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Define supported image extensions
    supported_extensions = ['.jpg', '.jpeg', '.png']

    # Process each image in the input folder
    for filename in os.listdir(input_folder):
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext in supported_extensions:
            image_path = os.path.join(input_folder, filename)
            print(f"\n--- Processing {filename} ---")
            
            # Set the image
            sam.set_image(image_path)

            # Generate masks for the current image
            all_masks = sam.generate_automatic_masks()
            print(f"Found {len(all_masks)} masks for {filename}.")
            
            # Define the output path for the mask image
            mask_filename = f"{os.path.splitext(filename)[0]}_mask.png"
            output_path = os.path.join(output_folder, mask_filename)
            
            # Save the combined mask
            sam.save_combined_mask(all_masks, output_path)

    print("\n--- Folder processing complete! ---")