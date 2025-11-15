import os
import sys
import cv2
import numpy as np
from pathlib import Path

from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog, QPushButton, 
                              QLabel, QListWidget, QVBoxLayout, QWidget, QHBoxLayout,
                              QMessageBox)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QPoint

# Add SAM2 to path
current_dir = Path(__file__).parent.resolve()
base_dir = current_dir.parent.resolve()
thirdparty_dir = base_dir / "thirdparty"
PIX2POLY_PATH = thirdparty_dir / "sam2"
sys.path.insert(0, str(PIX2POLY_PATH))

# Import SAM2 wrapper
try:
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False
    print("[WARNING] SAM2 not available. Install required packages.")


class SAM2Wrapper:
    def __init__(self, checkpoint_path, config_path, device=None):
        """Initialize the SAM2 model and predictor."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        print(f"Loading SAM2 model on {self.device}...")
        saved_cwd = os.getcwd()
        os.chdir(PIX2POLY_PATH)
        
        self.model = build_sam2(str(config_path), str(checkpoint_path), device=self.device)
        self.predictor = SAM2ImagePredictor(self.model)
        self.image = None
        
        os.chdir(saved_cwd)
        print("SAM2 model loaded successfully!")

    def set_image(self, image):
        """Set image for prediction (accepts numpy array)."""
        self.predictor.set_image(image)
        self.image = image

    def predict_with_points(self, input_points, input_labels, multimask_output=True):
        """Predict masks using point prompts."""
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
    
    @staticmethod
    def clean_mask(mask, kernel_size=5, min_area=100):
        """
        Clean mask using morphological operations.
        
        Args:
            mask: Binary mask (0 or 1)
            kernel_size: Size of morphological kernel
            min_area: Minimum contour area to keep
        """
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Morphological operations to clean noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        
        # Close small holes
        mask_closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
        
        # Remove small noise
        mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)
        
        # Find contours and filter by area
        contours, _ = cv2.findContours(mask_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create clean mask with only large contours
        clean_mask = np.zeros_like(mask_uint8)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                cv2.drawContours(clean_mask, [contour], -1, 255, -1)
        
        return (clean_mask > 0).astype(np.uint8)
    
    @staticmethod
    def get_bounding_boxes(mask, rotated=False):
        """
        Extract bounding boxes from mask as corner coordinates.

        Args:
            mask: Binary mask
            rotated: If True, use minimum area rotated rectangles

        Returns:
            List of 4-point rectangles: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        """
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bboxes = []
        for contour in contours:
            if rotated:
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
            else:
                x, y, w, h = cv2.boundingRect(contour)
                box = np.array([
                    [x,     y],
                    [x + w, y],
                    [x + w, y + h],
                    [x,     y + h]
                ], dtype=np.float32)
            bboxes.append(box.tolist())

        return bboxes
    
    @staticmethod
    def create_rectangle_mask(shape, bboxes):
        mask = np.zeros(shape, dtype=np.uint8)
        for box in bboxes:
            pts = np.array(box, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 1)
        return mask


class ImageLabel(QLabel):
    """Custom QLabel that handles mouse clicks for point annotation"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.points = []
        self.labels = []
        self.original_pixmap = None
        self.setMouseTracking(True)
        
    def set_image(self, pixmap):
        self.original_pixmap = pixmap
        self.update_display()
        
    def mousePressEvent(self, event):
        if not self.original_pixmap or event.button() != Qt.LeftButton:
            return
            
        # Get click position relative to the label widget
        pos = event.pos()
        
        # Get the actual displayed pixmap size (scaled)
        displayed_pixmap = self.pixmap()
        if not displayed_pixmap:
            return
            
        # Calculate offset (image might be centered in label)
        label_width = self.width()
        label_height = self.height()
        pixmap_width = displayed_pixmap.width()
        pixmap_height = displayed_pixmap.height()
        
        offset_x = (label_width - pixmap_width) / 2
        offset_y = (label_height - pixmap_height) / 2
        
        # Adjust click position by offset
        adjusted_x = pos.x() - offset_x
        adjusted_y = pos.y() - offset_y
        
        # Check if click is within the displayed image
        if adjusted_x < 0 or adjusted_y < 0 or adjusted_x >= pixmap_width or adjusted_y >= pixmap_height:
            return
        
        # Scale to original image coordinates
        scale_x = self.original_pixmap.width() / pixmap_width
        scale_y = self.original_pixmap.height() / pixmap_height
        
        x = int(adjusted_x * scale_x)
        y = int(adjusted_y * scale_y)
        
        # Check if Ctrl is pressed
        ctrl_pressed = event.modifiers() & Qt.ControlModifier
        label = 0 if ctrl_pressed else 1
        
        self.points.append([x, y])
        self.labels.append(label)
        
        label_text = "negative" if ctrl_pressed else "positive"
        print(f"Point ({x}, {y}), label={label} ({label_text})")
        
        self.update_display()
        
    def update_display(self):
        if not self.original_pixmap:
            return
            
        # Create a copy to draw on
        pixmap = self.original_pixmap.copy()
        painter = QPainter(pixmap)
        
        # Draw all points
        scale_factor = max(pixmap.width(), pixmap.height()) / 1000.0
        radius = max(5, int(8 * scale_factor))
        
        for point, label in zip(self.points, self.labels):
            x, y = point
            color = QColor(255, 0, 0) if label == 0 else QColor(0, 255, 0)
            
            # Draw filled circle
            painter.setBrush(color)
            painter.setPen(QPen(Qt.black, 2))
            painter.drawEllipse(QPoint(x, y), radius, radius)
        
        painter.end()
        
        # Scale to fit label while maintaining aspect ratio
        self.setPixmap(pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
    
    def reset_points(self):
        self.points = []
        self.labels = []
        self.update_display()
        
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.original_pixmap:
            self.update_display()


class ImageAnnotator(QWidget):
    """Window for annotating a single image"""
    
    def __init__(self, image_path, sam_wrapper, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.sam_wrapper = sam_wrapper
        self.img_cv = cv2.imread(image_path)
        
        # Convert BGR to RGB for SAM2
        self.img_rgb = cv2.cvtColor(self.img_cv, cv2.COLOR_BGR2RGB)
        
        self.setWindowTitle(f"Annotate - {os.path.basename(image_path)}")
        self.setGeometry(100, 100, 1200, 800)
        
        # Main layout
        layout = QVBoxLayout()
        
        # Instructions
        instructions = QLabel(
            "Left click = positive point (green) | "
            "Ctrl + Left click = negative point (red)"
        )
        instructions.setStyleSheet("padding: 5px; background-color: #f0f0f0;")
        layout.addWidget(instructions)
        
        # Image display
        self.image_label = ImageLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(800, 600)
        layout.addWidget(self.image_label)
        
        # Convert OpenCV image to QPixmap
        h, w, ch = self.img_cv.shape
        bytes_per_line = ch * w
        q_img = QImage(self.img_cv.data, w, h, bytes_per_line, QImage.Format_BGR888)
        self.image_label.set_image(QPixmap.fromImage(q_img))
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.reset_btn = QPushButton("Reset Points (R)")
        self.reset_btn.clicked.connect(self.reset_points)
        button_layout.addWidget(self.reset_btn)
        
        self.save_btn = QPushButton("Generate & Save Mask (S)")
        self.save_btn.clicked.connect(self.save_mask)
        self.save_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        button_layout.addWidget(self.save_btn)
        
        self.save_rect_btn = QPushButton("Save as Rectangles (Shift+S)")
        self.save_rect_btn.clicked.connect(self.save_rectangles)
        self.save_rect_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        button_layout.addWidget(self.save_rect_btn)
        
        self.save_rotated_btn = QPushButton("Save Rotated Rectangles (Ctrl+S)")
        self.save_rotated_btn.clicked.connect(self.save_rotated_rectangles)
        self.save_rotated_btn.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold;")
        button_layout.addWidget(self.save_rotated_btn)
        
        self.close_btn = QPushButton("Close (Q)")
        self.close_btn.clicked.connect(self.close)
        button_layout.addWidget(self.close_btn)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_S:
            if event.modifiers() & Qt.ControlModifier:
                self.save_rotated_rectangles()
            elif event.modifiers() & Qt.ShiftModifier:
                self.save_rectangles()
            else:
                self.save_mask()
        elif event.key() == Qt.Key_R:
            self.reset_points()
        elif event.key() == Qt.Key_Q or event.key() == Qt.Key_Escape:
            self.close()
            
    def reset_points(self):
        print("[INFO] Resetting points...")
        self.image_label.reset_points()
        
    def save_mask(self, clean=True, save_rectangles=False, rotated=False):
        if not self.image_label.points:
            QMessageBox.warning(self, "No Points", "Please add at least one point before generating mask.")
            return
        
        try:
            print(f"[INFO] Generating masks for {len(self.image_label.points)} points...")
            
            # Set image in SAM2
            self.sam_wrapper.set_image(self.img_rgb)
            image_name = Path(self.image_path).stem
            
            # Predict masks
            masks, scores, logits = self.sam_wrapper.predict_with_points(
                np.array(self.image_label.points), 
                np.array(self.image_label.labels),
                multimask_output=True
            )
            
            print(f"[INFO] Generated {len(masks)} masks with scores: {scores}")
            
            # Get the best mask (highest score)
            best_idx = np.argmax(scores)
            best_mask = masks[best_idx]
            best_score = scores[best_idx]
            
            print(f"[INFO] Selected mask {best_idx} with score: {best_score:.3f}")
            
            # Clean mask if requested
            if clean:
                print("[INFO] Cleaning mask...")
                best_mask = self.sam_wrapper.clean_mask(best_mask, kernel_size=5, min_area=100)
            
            # Create output directory
            output_dir = os.path.join("output")
            os.makedirs(output_dir, exist_ok=True)
            
            if save_rectangles:
                # Extract bounding boxes and create rectangle mask
                bboxes = self.sam_wrapper.get_bounding_boxes(best_mask, rotated=rotated)
                print(f"[INFO] Found {len(bboxes)} bounding boxes")
                
                # Save bounding boxes to text file
                bbox_path = os.path.join(output_dir, "bboxes.txt")
                with open(bbox_path, 'w') as f:
                    for i, box in enumerate(bboxes):
                        flat = [coord for pt in box for coord in pt]  # flatten [[x1,y1],..] → [x1,y1,x2,y2,x3,y3,x4,y4]
                        f.write(f"{i}," + ",".join(map(str, flat)) + "\n")
                print(f"[SAVED] Bounding boxes saved to {bbox_path}")
                
                # Create and save rectangle mask
                rect_mask = self.sam_wrapper.create_rectangle_mask(best_mask.shape, bboxes)
                rect_mask_path = os.path.join(output_dir, f"{image_name}_mask.png")
                cv2.imwrite(rect_mask_path, (rect_mask * 255).astype(np.uint8))
                print(f"[SAVED] Rectangle mask saved to {rect_mask_path}")
                
                # Save visualization with bounding boxes
                vis_img = self.img_cv.copy()
                for box in bboxes:
                    pts = np.array(box, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(vis_img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                vis_path = os.path.join(output_dir, f"{image_name}_vis.png")
                cv2.imwrite(vis_path, vis_img)
                print(f"[SAVED] Visualization saved to {vis_path}")
                
                QMessageBox.information(
                    self, "Success", 
                    f"Saved successfully!\n"
                    f"Score: {best_score:.3f}\n"
                    f"Bounding boxes: {len(bboxes)}\n"
                    f"Path: {output_dir}"
                )
            else:
                # Save cleaned mask
                mask_path = os.path.join(output_dir, "mask_cleaned.png")
                cv2.imwrite(mask_path, (best_mask * 255).astype(np.uint8))
                print(f"[SAVED] Cleaned mask saved to {mask_path}\n")
                
                QMessageBox.information(
                    self, "Success", 
                    f"Mask saved successfully!\n"
                    f"Score: {best_score:.3f}\n"
                    f"Path: {mask_path}"
                )
            
        except Exception as e:
            print(f"[ERROR] Failed to generate mask: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Failed to generate mask:\n{str(e)}")
    
    def save_rotated_rectangles(self):
        """Save mask as minimum area rotated rectangles"""
        self.save_mask(clean=True, save_rectangles=True, rotated=True)
        
    def save_rectangles(self):
        """Save mask as clean rectangles with bounding boxes"""
        self.save_mask(clean=True, save_rectangles=True)


class PVSegApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PV Cell SAM2 Segmentation")
        self.setGeometry(300, 200, 500, 500)

        layout = QVBoxLayout()
        
        # SAM2 status
        self.status_label = QLabel("Initializing SAM2...")
        self.status_label.setStyleSheet("padding: 10px; background-color: #fff3cd; border-radius: 5px;")
        layout.addWidget(self.status_label)
        
        self.folder_label = QLabel("Select image folder:")
        layout.addWidget(self.folder_label)

        self.select_btn = QPushButton("Open Folder")
        self.select_btn.clicked.connect(self.select_folder)
        self.select_btn.setEnabled(False)
        layout.addWidget(self.select_btn)

        self.image_list = QListWidget()
        layout.addWidget(self.image_list)

        self.open_img_btn = QPushButton("Open Selected Image")
        self.open_img_btn.clicked.connect(self.open_selected_image)
        self.open_img_btn.setEnabled(False)
        layout.addWidget(self.open_img_btn)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.image_paths = []
        self.annotator_window = None
        self.sam_wrapper = None
        
        # Initialize SAM2 in background
        self.init_sam2()

    def init_sam2(self):
        """Initialize SAM2 model"""
        if not SAM2_AVAILABLE:
            self.status_label.setText("❌ SAM2 not available. Please install required packages.")
            self.status_label.setStyleSheet("padding: 10px; background-color: #f8d7da; border-radius: 5px;")
            return
        
        try:
            # Update these paths to match your setup
            config_path = Path(PIX2POLY_PATH / "sam2/configs/sam2.1/sam2.1_hiera_l.yaml")
            checkpoint_path = Path(PIX2POLY_PATH / "checkpoints/sam2.1_hiera_large.pt")
            
            if not config_path.exists():
                raise FileNotFoundError(f"Config not found: {config_path}")
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
            self.sam_wrapper = SAM2Wrapper(checkpoint_path, config_path)
            self.status_label.setText("✓ SAM2 loaded successfully!")
            self.status_label.setStyleSheet("padding: 10px; background-color: #d4edda; border-radius: 5px;")
            self.select_btn.setEnabled(True)
            self.open_img_btn.setEnabled(True)
            
        except Exception as e:
            self.status_label.setText(f"❌ Failed to load SAM2: {str(e)}")
            self.status_label.setStyleSheet("padding: 10px; background-color: #f8d7da; border-radius: 5px;")
            print(f"[ERROR] SAM2 initialization failed: {e}")

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder:
            self.image_paths = [os.path.join(folder, f) for f in os.listdir(folder)
                                if f.lower().endswith(('.jpg', '.png', '.jpeg', '.tif', '.tiff'))]
            self.image_list.clear()
            for img_path in self.image_paths:
                self.image_list.addItem(os.path.basename(img_path))
            print(f"[INFO] Loaded {len(self.image_paths)} images from {folder}")

    def open_selected_image(self):
        if not self.sam_wrapper:
            QMessageBox.warning(self, "SAM2 Not Ready", "SAM2 model is not loaded yet.")
            return
            
        selected_item = self.image_list.currentItem()
        if selected_item:
            idx = self.image_list.row(selected_item)
            image_path = self.image_paths[idx]
            
            # Close previous annotator if open
            if self.annotator_window:
                self.annotator_window.close()
            
            self.annotator_window = ImageAnnotator(image_path, self.sam_wrapper)
            self.annotator_window.show()


if __name__ == "__main__":
    app = QApplication([])
    window = PVSegApp()
    window.show()
    app.exec_()