import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import sys
import os
from pathlib import Path

# current_dir = os.path.dirname(os.path.abspath(__file__))
# active_dir = os.getcwd()
# PIX2POLY_PATH = r'thirdparty/sam2'
# sam_dir = os.path.join(current_dir, PIX2POLY_PATH)
# sys.path.insert(0, str(sam_dir))

current_dir = Path(__file__).parent.resolve()
base_dir = current_dir.parent.parent.resolve()
thridparty_dir = base_dir / "thirdparty"
PIX2POLY_PATH = thridparty_dir / "sam2"
sys.path.insert(0, str(PIX2POLY_PATH))
active_dir = str(base_dir)

# Sam import

# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )
from pathlib import Path
config_path = Path("/home/wslhdsl36/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml").resolve()
checkpoint_path = Path("/home/wslhdsl36/sam2/sam2/checkpoints/sam2.1_hiera_l.pth").resolve()

# Sam plot functions
np.random.seed(3)

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


class SAM2Wrapper:
    def __init__(self, checkpoint_path=checkpoint_path, config_path=config_path, device=None):
        """
        Initialize the SAM2 model and predictor.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        os.chdir(PIX2POLY_PATH)
        print(f"Loading SAM2 model on {self.device}...")
        self.model = build_sam2(config_path, (PIX2POLY_PATH / checkpoint_path).resolve(), device=self.device)
        self.predictor = SAM2ImagePredictor(self.model)
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

    
        
    # ======================= Prediction Functions ======================= #

    def predict_with_points(self, input_points, input_labels, multimask_output=True):
        masks, scores, logits = self.predictor.predict(
            point_coords=np.array(input_points),
            point_labels=np.array(input_labels),
            multimask_output=multimask_output,
        )

        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]
        return masks, scores, logits

    def refine_with_additional_points(self, prev_logits, new_points, new_labels, multimask_output=False):
        masks, scores, _ = self.predictor.predict(
            point_coords=np.array(new_points),
            point_labels=np.array(new_labels),
            mask_input=prev_logits[None, :, :],
            multimask_output=multimask_output,
        )
        return masks, scores

    def predict_with_box(self, input_box, multimask_output=False):
        masks, scores, logits = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=np.array(input_box)[None, :],
            multimask_output=multimask_output,
        )
        return masks, scores, logits

    def predict_with_box_and_points(self, input_box, input_points, input_labels, multimask_output=False):
        masks, scores, logits = self.predictor.predict(
            point_coords=np.array(input_points),
            point_labels=np.array(input_labels),
            box=np.array(input_box),
            multimask_output=multimask_output,
        )
        return masks, scores, logits

    def predict_batch(self, pts_batch, labels_batch, box_batch=None, multimask_output=True):
        masks_batch, scores_batch, _ = self.predictor.predict_batch(
            pts_batch, labels_batch, box_batch=box_batch, multimask_output=multimask_output
        )

        best_masks = []
        for masks, scores in zip(masks_batch, scores_batch):
            best_masks.append(masks[range(len(masks)), np.argmax(scores, axis=-1)])
        return best_masks, scores_batch

    def get_best_mask(self, masks, scores):
        best_idx = np.argmax(scores)
        return masks[best_idx], scores[best_idx]

    # ======================= Visualization Functions ======================= #

    @staticmethod
    def show_mask(mask, ax, random_color=False, borders=True):
        np.random.seed(3)
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])

        h, w = mask.shape[-2:]
        mask = mask.astype(np.uint8)
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

        if borders:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
        ax.imshow(mask_image)

    @staticmethod
    def show_points(coords, labels, ax, marker_size=375):
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*',
                   s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*',
                   s=marker_size, edgecolor='white', linewidth=1.25)

    @staticmethod
    def show_box(box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green',
                                   facecolor=(0, 0, 0, 0), lw=2))

    def show_masks(self, masks, scores, point_coords=None, box_coords=None,
                   input_labels=None, borders=True):
        """
        Display masks (and optionally, points or boxes) over the loaded image.
        """
        if self.image is None:
            raise ValueError("No image loaded. Please call set_image() first.")

        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.figure(figsize=(10, 10))
            plt.imshow(self.image)
            self.show_mask(mask, plt.gca(), borders=borders)
            if point_coords is not None and input_labels is not None:
                self.show_points(point_coords, input_labels, plt.gca())
            if box_coords is not None:
                self.show_box(box_coords, plt.gca())
            if len(scores) > 1:
                plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
            plt.axis('off')
            plt.show()


# ======================= Example Usage ======================= #
if __name__ == "__main__":
    sam = SAM2Wrapper(
        checkpoint_path="checkpoints/sam2.1_hiera_large.pt",
        config_path="configs/sam2.1/sam2.1_hiera_l.yaml",
    )
    os.chdir(active_dir)
    sam.set_image("D:\\S\\ANTS\\Data\\tmp\\A\\IRX_4577.JPG")

    # Example 1: Points only
    input_points = np.array([[200, 375]])
    input_labels = np.array([1])
    masks, scores, logits = sam.predict_with_points(input_points, input_labels)
    sam.show_masks(masks, scores, point_coords=input_points, input_labels=input_labels)

    # Example 2: Box
    input_box = np.array([425, 600, 700, 875])
    masks, scores, _ = sam.predict_with_box(input_box)
    sam.show_masks(masks, scores, box_coords=input_box)

    # Example 3: Batch
    image1_pts = np.array([[[500, 375]], [[650, 750]]])
    image1_labels = np.array([[1], [1]])
    image2_pts = np.array([[[400, 300]], [[630, 300]]])
    image2_labels = np.array([[1], [1]])

    pts_batch = [image1_pts, image2_pts]
    labels_batch = [image1_labels, image2_labels]

    best_masks, scores_batch = sam.predict_batch(pts_batch, labels_batch)
    print(f"Predicted {len(best_masks)} batched masks successfully.")
