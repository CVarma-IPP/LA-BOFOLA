import sys, os
import numpy as np
import cv2
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout,
    QWidget, QFileDialog, QLabel, QSlider, QGroupBox, QSizePolicy, QSpacerItem,
    QComboBox, QListWidget, QSpinBox, QMessageBox
)
from PyQt6.QtCore import Qt, QSettings, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QPalette, QColor
import pyqtgraph as pg
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from BPM_functions import (load_calibration_file, input_for_proj_mat, 
                           calc_perspective_transform_matrix, image_to_BP)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class BatchWorker(QThread):
    image_processed = pyqtSignal(str)
    def __init__(self, file_list, parent=None):
        super().__init__(parent)
        self.file_list = file_list
        self.parent = parent
        self._running = True
    
    def run(self):
        import time
        for file in self.file_list:
            if not self._running:
                break
            self.image_processed.emit(file)
            time.sleep(1)  # 1 Hz
    
    def stop(self):
        self._running = False

class BeamProfileMonitor(QMainWindow):
    """
    Main GUI class for monitoring and analyzing electron beam profile images.
    This class handles all user interactions, file selections, image processing,
    and display of results. Most users will only need to edit this class to
    change the GUI layout, add/remove buttons, or adjust how results are shown.
    """
    def __init__(self):
        super().__init__()
        # Set up the main window properties
        self.setWindowTitle("Beam Profile Monitor")
        self.setMinimumSize(1300, 900)  # Minimum window size
        self.resize(1300, 900)
        
        # Initialize important variables
        self.directory = ""                 # Directory with images to analyze
        self.output_directory = ""          # Where to save analysis results
        self.calibration_data = {}          # Calibration info loaded from file
        self.projmat, self.maxWid, self.maxLen = None, None, None  # Calibration transform
        self.threshold = 0.37               # Default threshold for analysis (0.37 is standard)
        self.observer = None                # For monitoring a directory for new files
        self.activity_log = []              # Stores log messages for the user
        self.colormap = 'gray'              # Default colormap for image display
        
        # Persistent settings (remembers last used folders, etc.)
        self.settings = QSettings('BeamProfileMonitor', 'UserSettings')
        self.restore_last_session()         # Restore previous session info
        self.initUI()                       # Build the GUI
        self.apply_styles()                 # Apply custom look

    def restore_last_session(self):
        # Restore last used directories and calibration file
        self.directory = self.settings.value('last_directory', "")
        self.output_directory = self.settings.value('last_output_directory', "")
        self.last_calib_file = self.settings.value('last_calib_file', "")

    def save_session(self):
        # Save current directories and calibration file for next time
        self.settings.setValue('last_directory', self.directory)
        self.settings.setValue('last_output_directory', self.output_directory)
        self.settings.setValue('last_calib_file', getattr(self, 'last_calib_file', ""))

    def initUI(self):
        """
        Build the GUI layout and add all widgets (buttons, sliders, plots, etc).
        Most users can add/remove controls here to change the interface.
        """
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(18)

        # --- Sidebar for controls ---
        sidebar = QVBoxLayout()
        sidebar.setSpacing(8)  # Reduced spacing for minimalism
        sidebar.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Status label
        self.statusLabel = QLabel("Select a directory and calibration file to begin.")
        self.statusLabel.setFont(QFont("Segoe UI", 10, QFont.Weight.Normal))  # Smaller font
        self.statusLabel.setStyleSheet("color: #333; margin-bottom: 4px;")
        sidebar.addWidget(self.statusLabel)

        # File selection group
        file_group = QGroupBox("File Selection")
        file_group.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))  # Smaller font
        file_layout = QVBoxLayout()
        selectDirBtn = QPushButton("Select Image Directory")
        selectDirBtn.setFont(QFont("Segoe UI", 9))
        selectDirBtn.clicked.connect(self.select_directory)
        file_layout.addWidget(selectDirBtn)
        selectCalibBtn = QPushButton("Select Calibration File")
        selectCalibBtn.setFont(QFont("Segoe UI", 9))
        selectCalibBtn.clicked.connect(self.select_calibration)
        file_layout.addWidget(selectCalibBtn)
        file_group.setLayout(file_layout)
        sidebar.addWidget(file_group)

        # Output directory selection
        output_group = QGroupBox("Output Directory")
        output_group.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        output_layout = QVBoxLayout()
        selectOutputBtn = QPushButton("Select Output Directory")
        selectOutputBtn.setFont(QFont("Segoe UI", 9))
        selectOutputBtn.clicked.connect(self.select_output_directory)
        output_layout.addWidget(selectOutputBtn)
        self.outputDirLabel = QLabel(self.output_directory or "Not selected")
        self.outputDirLabel.setFont(QFont("Segoe UI", 8))
        output_layout.addWidget(self.outputDirLabel)
        output_group.setLayout(output_layout)
        sidebar.addWidget(output_group)

        # Actions group
        action_group = QGroupBox("Actions")
        action_group.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        action_layout = QVBoxLayout()
        monitor_row = QHBoxLayout()
        self.monitorBtn = QPushButton("Start Monitoring")
        self.monitorBtn.setFont(QFont("Segoe UI", 9))
        self.monitorBtn.setCheckable(True)
        self.monitorBtn.clicked.connect(self.toggle_monitoring)
        monitor_row.addWidget(self.monitorBtn)
        self.pauseMonitorBtn = QPushButton("Pause Monitoring")
        self.pauseMonitorBtn.setFont(QFont("Segoe UI", 9))
        self.pauseMonitorBtn.setCheckable(True)
        self.pauseMonitorBtn.setEnabled(False)
        self.pauseMonitorBtn.clicked.connect(self.toggle_pause_monitoring)
        monitor_row.addWidget(self.pauseMonitorBtn)
        action_layout.addLayout(monitor_row)
        analyzeExistingBtn = QPushButton("Analyze Existing Images")
        analyzeExistingBtn.setFont(QFont("Segoe UI", 9))
        analyzeExistingBtn.clicked.connect(self.analyze_existing_images)
        action_layout.addWidget(analyzeExistingBtn)
        action_group.setLayout(action_layout)
        sidebar.addWidget(action_group)

        # Threshold group
        threshold_group = QGroupBox("Threshold Adjustment")
        threshold_group.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        threshold_layout = QVBoxLayout()
        threshold_label_row = QHBoxLayout()
        # Removed inner label
        threshold_label_row.addStretch()
        self.thresholdValueLabel = QLabel(f"{self.threshold:.2f}")
        self.thresholdValueLabel.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        threshold_label_row.addWidget(self.thresholdValueLabel)
        threshold_layout.addLayout(threshold_label_row)
        slider_row = QHBoxLayout()
        min_label = QLabel("0.00")
        min_label.setFont(QFont("Segoe UI", 8))
        slider_row.addWidget(min_label)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(int(self.threshold * 100))
        self.slider.setFixedWidth(120)  # Smaller slider
        self.slider.valueChanged.connect(self.update_threshold)
        slider_row.addWidget(self.slider)
        max_label = QLabel("1.00")
        max_label.setFont(QFont("Segoe UI", 8))
        slider_row.addWidget(max_label)
        threshold_layout.addLayout(slider_row)
        threshold_group.setLayout(threshold_layout)
        sidebar.addWidget(threshold_group)

        # Colormap selection
        cmap_group = QGroupBox("Colormap")
        cmap_group.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        cmap_layout = QHBoxLayout()
        # Removed inner label
        self.cmapCombo = QComboBox()
        self.cmapCombo.setFont(QFont("Segoe UI", 8))
        self.cmapCombo.addItems(['gray', 'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'jet', 'hot'])
        self.cmapCombo.setCurrentText(self.colormap)
        self.cmapCombo.currentTextChanged.connect(self.change_colormap)
        cmap_layout.addWidget(self.cmapCombo)
        cmap_group.setLayout(cmap_layout)
        sidebar.addWidget(cmap_group)

        # Bit depth selection
        bitdepth_group = QGroupBox("Bit Depth")
        bitdepth_group.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        bitdepth_layout = QHBoxLayout()
        # Removed inner label
        self.bitDepthCombo = QComboBox()
        self.bitDepthCombo.setFont(QFont("Segoe UI", 8))
        self.bitDepthCombo.addItems(["8-bit", "14-bit"])
        self.bitDepthCombo.setCurrentText("8-bit")
        self.bitDepthCombo.currentTextChanged.connect(self.change_bit_depth)
        bitdepth_layout.addWidget(self.bitDepthCombo)
        bitdepth_group.setLayout(bitdepth_layout)
        sidebar.addWidget(bitdepth_group)

        # Activity log
        log_group = QGroupBox("Activity Log")
        log_group.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        log_layout = QVBoxLayout()
        self.logWidget = QListWidget()
        self.logWidget.setFont(QFont("Segoe UI", 8))
        self.logWidget.setMinimumHeight(60)  # Smaller log
        log_layout.addWidget(self.logWidget)
        log_group.setLayout(log_layout)
        sidebar.addWidget(log_group)

        # Average profile controls
        avg_group = QGroupBox("Average Profile")
        avg_group.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        avg_layout = QHBoxLayout()
        self.nSpin = QSpinBox()
        self.nSpin.setFont(QFont("Segoe UI", 8))
        self.nSpin.setMinimum(1)
        self.nSpin.setMaximum(100)
        self.nSpin.setValue(10)
        self.nSpin.setPrefix("N=")
        avg_layout.addWidget(self.nSpin)
        avgBtn = QPushButton("Show Average Profile")
        avgBtn.setFont(QFont("Segoe UI", 8))
        avgBtn.clicked.connect(self.show_average_profile)
        avg_layout.addWidget(avgBtn)
        avg_group.setLayout(avg_layout)
        sidebar.addWidget(avg_group)

        sidebar.addStretch()

        # --- Main image display: two matplotlib canvases side by side ---
        from matplotlib import pyplot as plt
        image_layout = QHBoxLayout()
        image_layout.setSpacing(10)  # Less spacing
        # Left: Raw image
        self.raw_fig = Figure(figsize=(5,5))  # Slightly larger
        self.raw_canvas = FigureCanvas(self.raw_fig)
        image_layout.addWidget(self.raw_canvas, 1)
        # Right: Processed image
        self.proc_fig = Figure(figsize=(5,5))
        self.proc_canvas = FigureCanvas(self.proc_fig)
        image_layout.addWidget(self.proc_canvas, 1)

        # --- Spot size variation plot and stats panel below images ---
        plot_stats_layout = QHBoxLayout()
        plot_stats_layout.setSpacing(10)
        # Spot size variation plot (2/3 width)
        self.variationPlot = pg.PlotWidget()
        self.variationPlot.setMinimumHeight(171)  # Reduced by 5%
        self.variationPlot.setMaximumHeight(228)  # Reduced by 5%
        self.variationPlot.setBackground('w')
        self.variationPlot.showGrid(x=True, y=True)
        self.variationPlot.setLabel('left', 'Spot Size (%)', **{'font-size': '8pt'})
        self.variationPlot.setLabel('bottom', 'Image #', **{'font-size': '8pt'})
        plot_stats_layout.addWidget(self.variationPlot, 2)
        # Stats panel (1/3 width)
        self.statsPanel = QWidget()
        stats_layout = QHBoxLayout()  # Change to horizontal to fit button and stats
        stats_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        # Left: stats labels (vertical)
        stats_labels_layout = QVBoxLayout()
        self.meanVarLabel = QLabel("Mean Variation: -- %")
        self.meanVarLabel.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        stats_labels_layout.addWidget(self.meanVarLabel)
        self.maxVarLabel = QLabel("Max Variation: -- %")
        self.maxVarLabel.setFont(QFont("Segoe UI", 10, QFont.Weight.Normal))
        stats_labels_layout.addWidget(self.maxVarLabel)
        self.centroidStdLabelX = QLabel("Centroid Motion X: -- mm")
        self.centroidStdLabelX.setFont(QFont("Segoe UI", 10, QFont.Weight.Normal))
        stats_labels_layout.addWidget(self.centroidStdLabelX)
        self.centroidStdLabelY = QLabel("Centroid Motion Y: -- mm")
        self.centroidStdLabelY.setFont(QFont("Segoe UI", 10, QFont.Weight.Normal))
        stats_labels_layout.addWidget(self.centroidStdLabelY)
        stats_layout.addLayout(stats_labels_layout)
        # Right: reset button
        self.resetVarBtn = QPushButton("Reset Calcs")
        self.resetVarBtn.setFont(QFont("Segoe UI", 9))
        self.resetVarBtn.setFixedHeight(32)
        self.resetVarBtn.clicked.connect(self.reset_variation_data)
        stats_layout.addWidget(self.resetVarBtn)
        self.statsPanel.setLayout(stats_layout)
        plot_stats_layout.addWidget(self.statsPanel, 1)
        plot_group = QGroupBox("Spot Size Variation")
        plot_group.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        plot_group.setLayout(plot_stats_layout)

        # --- Compose the right column ---
        right_column = QVBoxLayout()
        right_column.addLayout(image_layout, 4)  # Give more weight to images
        right_column.addWidget(plot_group, 1)
        right_column.addStretch()

        # Add sidebar and right column to main layout
        main_layout.addLayout(sidebar, 0)
        main_layout.addLayout(right_column, 1)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # For storing last N processed images (projective transformed)
        self.last_n_proc_imgs = []
        self.last_n_raw_imgs = []
        # Bit depth mode (default 8-bit)
        self.bit_depth_mode = "8-bit"
        # For centroid tracking
        self.centroid_positions = []

    def apply_styles(self):
        """
        Set the color scheme and style for the GUI. Edit this to change the look.
        """
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f6fa;
            }
            QGroupBox {
                border: 1.5px solid #b2bec3;
                border-radius: 8px;
                margin-top: 10px;
                background-color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
                color: #636e72;
            }
            QPushButton {
                background-color: #0984e3;
                color: white;
                border-radius: 6px;
                padding: 7px 18px;
                font-size: 11pt;
            }
            QPushButton:checked {
                background-color: #00b894;
            }
            QPushButton:hover {
                background-color: #74b9ff;
            }
            QLabel {
                color: #2d3436;
            }
            QSlider::groove:horizontal {
                border: 1px solid #b2bec3;
                height: 8px;
                background: #dfe6e9;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #0984e3;
                border: 1px solid #636e72;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
                           """)

    def select_directory(self):
        """
        Open a dialog for the user to select the folder containing images.
        """
        self.directory = QFileDialog.getExistingDirectory(self, "Select Image Directory", self.directory or "")
        self.statusLabel.setText(f"Directory selected: {self.directory}")
        self.save_session()
        self.log_action(f"Image directory set: {self.directory}")

    def select_calibration(self):
        """
        Open a dialog to select a calibration file, then load and check it.
        Calibration is required for correct analysis.
        """
        calib_file, _ = QFileDialog.getOpenFileName(self, "Select Calibration File", self.last_calib_file or "")
        if calib_file:
            self.last_calib_file = calib_file
            self.save_session()
            self.calibration_data = load_calibration_file(calib_file)
            required_keys = [
                'Corner_TL_x', 'Corner_TL_y', 'Corner_TR_x', 'Corner_TR_y',
                'Corner_BR_x', 'Corner_BR_y', 'Corner_BL_x', 'Corner_BL_y', 'Square_flag'
            ]
            missing = [k for k in required_keys if k not in self.calibration_data]
            if missing:
                self.statusLabel.setText(f"Calibration file missing keys: {', '.join(missing)}")
                self.projmat = None
                return
            try:
                corners_TL = [self.calibration_data['Corner_TL_x'], self.calibration_data['Corner_TL_y']]
                corners_TR = [self.calibration_data['Corner_TR_x'], self.calibration_data['Corner_TR_y']]
                corners_BR = [self.calibration_data['Corner_BR_x'], self.calibration_data['Corner_BR_y']]
                corners_BL = [self.calibration_data['Corner_BL_x'], self.calibration_data['Corner_BL_y']]
                square_flag = str(self.calibration_data['Square_flag']).lower() in ['true', '1', 'yes']
                source_pts, destin_pts, self.maxWid, self.maxLen, _ = input_for_proj_mat(
                    corners_TL, corners_TR, corners_BR, corners_BL, square_flag)
                self.projmat = calc_perspective_transform_matrix(source_pts, destin_pts)
                self.statusLabel.setText("Calibration loaded successfully.")
            except Exception as e:
                self.statusLabel.setText(f"Error loading calibration: {str(e)}")
                self.projmat = None

    def toggle_pause_monitoring(self):
        """
        Pause or resume the directory monitoring observer.
        """
        if not self.observer:
            return
        if self.pauseMonitorBtn.isChecked():
            self.observer.stop()
            self.pauseMonitorBtn.setText("Resume Monitoring")
            self.log_action("Monitoring paused.")
        else:
            self.observer.start()
            self.pauseMonitorBtn.setText("Pause Monitoring")
            self.log_action("Monitoring resumed.")

    def toggle_monitoring(self):
        """
        Start or stop monitoring the selected folder for new images.
        When monitoring, new images will be analyzed automatically.
        """
        if self.monitorBtn.isChecked():
            if not self.directory:
                self.statusLabel.setText("Please select a directory first.")
                self.monitorBtn.setChecked(False)
                return
            self.start_monitoring()
            self.monitorBtn.setText("Stop Monitoring")
            self.pauseMonitorBtn.setEnabled(True)
            self.pauseMonitorBtn.setChecked(False)
            self.pauseMonitorBtn.setText("Pause Monitoring")
        else:
            self.stop_monitoring()
            self.monitorBtn.setText("Start Monitoring")
            self.pauseMonitorBtn.setEnabled(False)
            self.pauseMonitorBtn.setChecked(False)
            self.pauseMonitorBtn.setText("Pause Monitoring")

    def start_monitoring(self):
        """
        Begin watching the folder for new image files (TIFFs only).
        """
        from watchdog.events import FileSystemEventHandler
        class TiffOnlyHandler(FileSystemEventHandler):
            def on_created(self, event):
                if event.src_path.lower().endswith(('.tif', '.tiff')):
                    self.outer.process_image(event.src_path)
        handler = TiffOnlyHandler()
        handler.outer = self
        self.observer = Observer()
        self.observer.schedule(handler, self.directory, recursive=False)
        self.observer.start()
        self.statusLabel.setText("Monitoring started...")

    def stop_monitoring(self):
        """
        Stop watching the folder for new files.
        """
        if self.observer:
            self.observer.stop()
            self.observer.join()
        self.statusLabel.setText("Monitoring stopped.")

    def analyze_existing_images(self):
        """
        Analyze all existing TIFF images in the selected folder (batch mode).
        Uses a QThread worker to keep the UI responsive and safe.
        """
        if not self.directory:
            self.statusLabel.setText("Please select a directory first.")
            return
        files = sorted([
            os.path.join(self.directory, f)
            for f in os.listdir(self.directory)
            if f.lower().endswith(('.tif', '.tiff'))
        ])
        if not files:
            self.statusLabel.setText("No TIFF images found in the selected directory.")
            return
        # Stop any previous worker
        if hasattr(self, 'batch_worker') and self.batch_worker is not None:
            self.batch_worker.stop()
            self.batch_worker.wait()
        self.batch_worker = BatchWorker(files)
        self.batch_worker.image_processed.connect(self.process_image)
        self.batch_worker.start()

    def _round_floats(self, obj):
        """
        Recursively round all floats in a dict/list to two decimal places for output JSON.
        """
        if isinstance(obj, float):
            return round(obj, 2)
        elif isinstance(obj, dict):
            return {k: self._round_floats(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._round_floats(v) for v in obj]
        else:
            return obj

    def process_image(self, filepath):
        """
        Main function to process and analyze a single image file.
        Loads the image, applies calibration and threshold, fits an ellipse,
        displays the result, and saves analysis output (JSON).
        """
        import json
        from datetime import datetime
        from skimage.measure import EllipseModel
        # Check if calibration data is loaded
        if self.projmat is None:
            self.statusLabel.setText("Please load calibration first.")
            QMessageBox.critical(self, "Calibration Error", "Please load calibration first.")
            return
        # Load and preprocess the image
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            self.statusLabel.setText(f"Failed to load image: {os.path.basename(filepath)}")
            QMessageBox.critical(self, "Image Error", f"Failed to load image: {os.path.basename(filepath)}")
            self.log_action(f"Failed to load image: {filepath}")
            return
        img = img.astype(np.float32)
        # Apply beam profile analysis
        try:
            processed_img, contour = image_to_BP(
                img, self.threshold, self.projmat, self.maxWid, self.maxLen)
            processed_img = np.round(processed_img, 1)
            # Fit ellipse to contour
            ellipse_params = None
            spot_area = None
            ellipse_fit = None
            if contour is not None and len(contour) >= 5:
                ellipse = EllipseModel()
                if ellipse.estimate(contour):
                    xc, yc, a, b, theta = ellipse.params
                    ellipse_params = {
                        'center_x': float(xc), 'center_y': float(yc),
                        'axis_a': float(a), 'axis_b': float(b), 'angle_rad': float(theta)
                    }
                    spot_area = np.pi * a * b
                    ellipse_fit = (xc, yc, a, b, theta)
                else:
                    ellipse_params = None
            else:
                ellipse_params = None
        except Exception as e:
            self.statusLabel.setText(f"Error processing {os.path.basename(filepath)}: {str(e)}")
            QMessageBox.critical(self, "Processing Error", f"Error processing {os.path.basename(filepath)}: {str(e)}")
            self.log_action(f"Error processing {filepath}: {str(e)}")
            return
        # --- Plot raw image (left) ---
        self.raw_fig.clf()
        ax_raw = self.raw_fig.add_subplot(111)
        vmax = 255 if self.bit_depth_mode == "8-bit" else 16383
        im_raw = ax_raw.imshow(img, cmap=self.colormap, origin='lower', vmin=0, vmax=vmax)
        ax_raw.set_title('Raw Image')
        ax_raw.set_xlabel('X (pixels)')
        ax_raw.set_ylabel('Y (pixels)')
        # Make colorbar same height as image
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider_raw = make_axes_locatable(ax_raw)
        cax_raw = divider_raw.append_axes("right", size="4%", pad=0.05)
        self.raw_fig.colorbar(im_raw, cax=cax_raw, orientation='vertical')
        self.raw_canvas.draw()
        # --- Plot processed image (right) ---
        self.proc_fig.clf()
        ax_proc = self.proc_fig.add_subplot(111)
        extent = [0, self.maxLen, 0, self.maxWid]  # axes in mm if calibration is correct
        im_proc = ax_proc.imshow(processed_img, cmap=self.colormap, origin='lower', extent=extent, vmin=0, vmax=vmax)
        ax_proc.set_title('Processed Image (Transformed)')
        ax_proc.set_xlabel('X (mm)')
        ax_proc.set_ylabel('Y (mm)')
        # Make colorbar same height as image
        divider_proc = make_axes_locatable(ax_proc)
        cax_proc = divider_proc.append_axes("right", size="4%", pad=0.05)
        self.proc_fig.colorbar(im_proc, cax=cax_proc, orientation='vertical')
        # Overlay ellipse if available
        if ellipse_fit is not None:
            from matplotlib.patches import Ellipse
            xc, yc, a, b, theta = ellipse_fit
            # Convert angle from radians to degrees
            angle_deg = np.degrees(theta)
            # Swap xc/yc and a/b for correct display orientation
            e = Ellipse((yc, xc), 2*b, 2*a, angle=angle_deg, edgecolor='red', facecolor='none', lw=2, alpha=0.7, linestyle='--')
            ax_proc.add_patch(e)
        self.proc_canvas.draw()
        self.statusLabel.setText(f"Processed: {os.path.basename(filepath)}")
        # Save analysis output file
        if self.output_directory:
            # Contour area (in pixels, or mm^2 if calibration is available)
            if contour is not None:
                from skimage.measure import regionprops, label
                # Create a mask from the contour
                from skimage.draw import polygon2mask
                mask = polygon2mask(processed_img.shape, contour)
                contour_area_px = float(np.sum(mask))
                # If calibration is available, convert to mm^2
                px_to_mm = None
                if self.maxWid and self.maxLen:
                    px_to_mm = (self.maxWid * self.maxLen) / (processed_img.shape[0] * processed_img.shape[1])
                    contour_area_mm2 = contour_area_px * px_to_mm
                else:
                    contour_area_mm2 = None
                # Contour centroid
                yx = np.argwhere(mask)
                if len(yx) > 0:
                    contour_centroid = [float(np.mean(yx[:,1])), float(np.mean(yx[:,0]))]
                else:
                    contour_centroid = None
                # Spot intensity stats
                spot_pixels = processed_img[mask]
                spot_max_intensity = float(np.max(spot_pixels)) if spot_pixels.size > 0 else None
                spot_mean_intensity = float(np.mean(spot_pixels)) if spot_pixels.size > 0 else None
                # Bounding box
                if len(yx) > 0:
                    min_x, max_x = int(np.min(yx[:,1])), int(np.max(yx[:,1]))
                    min_y, max_y = int(np.min(yx[:,0])), int(np.max(yx[:,0]))
                    spot_bbox = [min_x, min_y, max_x, max_y]
                else:
                    spot_bbox = None
            else:
                contour_area_px = None
                contour_area_mm2 = None
                contour_centroid = None
                spot_max_intensity = None
                spot_mean_intensity = None
                spot_bbox = None
            # --- Ellipse/variation/metadata as before ---
            if hasattr(self, 'spot_areas') and len(self.spot_areas) > 1 and spot_area is not None:
                mean_area = np.mean(self.spot_areas)
                spot_size_var = (spot_area - mean_area) / mean_area * 100
            else:
                spot_size_var = None
            if len(self.centroid_positions) >= 2:
                centroids = np.array(self.centroid_positions)
                centroid_std_x = float(np.nanstd(centroids[:, 0]))
                centroid_std_y = float(np.nanstd(centroids[:, 1]))
            else:
                centroid_std_x = None
                centroid_std_y = None
            out_data = {
                'image_file': filepath,
                'calibration_file': getattr(self, 'last_calib_file', ''),
                'analysis_time': datetime.now().isoformat(),
                'ellipse_fit': ellipse_params,  # dict or None
                'spot_area_ellipse': float(spot_area) if spot_area is not None else None,
                'centroid_ellipse': [float(ellipse_fit[0]), float(ellipse_fit[1])] if ellipse_fit is not None else None,
                'spot_size_variation_percent': float(spot_size_var) if spot_size_var is not None else None,
                'centroid_motion_std_x_mm': centroid_std_x,
                'centroid_motion_std_y_mm': centroid_std_y,
                'fit_success': ellipse_params is not None,
                'error_message': None if ellipse_params is not None else 'Ellipse fit failed',
                'threshold': self.threshold,
                'bit_depth': self.bit_depth_mode,
                # Direct beam profile data (no contour points):
                'contour_area_px': contour_area_px,
                'contour_area_mm2': contour_area_mm2,
                'contour_centroid': contour_centroid,
                'spot_max_intensity': spot_max_intensity,
                'spot_mean_intensity': spot_mean_intensity,
                'spot_bbox': spot_bbox,
            }
            out_data = self._round_floats(out_data)
            out_name = os.path.splitext(os.path.basename(filepath))[0] + '_analysis.json'
            out_path = os.path.join(self.output_directory, out_name)
            try:
                with open(out_path, 'w') as f:
                    json.dump(out_data, f, indent=2)
                self.log_action(f"Saved analysis output: {out_path}")
            except Exception as e:
                self.log_action(f"Failed to save output for {filepath}: {str(e)}")
        else:
            self.log_action("No output directory set. Analysis output not saved.")
        # Store for average profile
        N = self.nSpin.value()
        self.last_n_proc_imgs.append(processed_img)
        self.last_n_raw_imgs.append(img)
        self.last_n_proc_imgs = self.last_n_proc_imgs[-N:]
        self.last_n_raw_imgs = self.last_n_raw_imgs[-N:]
        # Store centroid for stats (if ellipse fit available)
        if ellipse_fit is not None:
            xc, yc, *_ = ellipse_fit
            self.centroid_positions.append((xc, yc))
        else:
            self.centroid_positions.append((np.nan, np.nan))
        self.centroid_positions = self.centroid_positions[-50:]
        # Update spot size variation plot and stats
        self.update_spot_size_variation(spot_area)
        self.update_stats_panel()

    def update_threshold(self):
        """
        Update the threshold value for analysis when the slider is moved.
        Shows the value with two decimal places.
        """
        self.threshold = round(self.slider.value() / 100.0, 2)
        self.thresholdValueLabel.setText(f"{self.threshold:.2f}")

    def select_output_directory(self):
        """
        Open a dialog to select where analysis results (JSON) will be saved.
        """
        dir_ = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_:
            self.output_directory = dir_
            self.outputDirLabel.setText(self.output_directory)
            self.save_session()
            self.log_action(f"Output directory set: {self.output_directory}")

    def log_action(self, msg):
        """
        Add a message to the activity log (bottom left of GUI).
        """
        from datetime import datetime
        timestamp = datetime.now().strftime('%H:%M:%S')
        entry = f"[{timestamp}] {msg}"
        self.activity_log.append(entry)
        self.logWidget.addItem(entry)
        self.logWidget.scrollToBottom()

    def change_colormap(self, cmap):
        """
        Change the colormap used for displaying images.
        """
        self.colormap = cmap
        self.log_action(f"Colormap changed to: {cmap}")
        # Optionally, refresh the current image display

    def change_bit_depth(self, mode):
        """
        Change the bit depth mode for image display (affects colorbar scaling).
        """
        self.bit_depth_mode = mode
        self.log_action(f"Bit depth changed to: {mode}")
        # Optionally, refresh the current image display

    def show_average_profile(self):
        """
        Show a popup window with the average of the last N processed images (projective transformed),
        including lineouts in x (top) and y (left).
        """
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        N = self.nSpin.value()
        if len(self.last_n_proc_imgs) == 0:
            QMessageBox.information(self, "No Data", "No processed images to average.")
            return
        num_used = min(N, len(self.last_n_proc_imgs))
        avg_img = np.mean(self.last_n_proc_imgs[-num_used:], axis=0)
        # Compute lineouts
        x_lineout = np.mean(avg_img, axis=0)
        y_lineout = np.mean(avg_img, axis=1)
        # Normalize for display
        x_lineout = x_lineout / np.max(x_lineout) if np.max(x_lineout) > 0 else x_lineout
        y_lineout = y_lineout / np.max(y_lineout) if np.max(y_lineout) > 0 else y_lineout
        # Set up gridspec
        fig = plt.figure(figsize=(7,7))
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 4], height_ratios=[1, 4], wspace=0.05, hspace=0.05)
        ax_y = fig.add_subplot(gs[1,0])  # left
        ax_img = fig.add_subplot(gs[1,1])  # main image
        ax_x = fig.add_subplot(gs[0,1], sharex=ax_img)  # top
        # Main image
        im = ax_img.imshow(avg_img, cmap=self.colormap, origin='lower', extent=[0, self.maxLen, 0, self.maxWid])
        ax_img.set_xlabel('X (mm)')
        ax_img.set_ylabel('Y (mm)')
        ax_img.set_title(f'Average of Last {num_used} Processed Images')
        # X lineout (top)
        ax_x.plot(np.linspace(0, self.maxLen, avg_img.shape[1]), x_lineout, color='k')
        ax_x.set_ylabel('Norm. X Lineout')
        ax_x.set_xlim(0, self.maxLen)
        ax_x.set_xticks([])
        ax_x.grid(True, which='both', linestyle='--', alpha=0.5)
        # Y lineout (left)
        ax_y.plot(y_lineout, np.linspace(0, self.maxWid, avg_img.shape[0]), color='k')
        ax_y.set_xlabel('Norm. Y Lineout')
        ax_y.set_ylim(0, self.maxWid)
        ax_y.set_yticks([])
        ax_y.invert_xaxis()
        ax_y.grid(True, which='both', linestyle='--', alpha=0.5)
        # Hide spines for cleaner look
        for ax in [ax_x, ax_y]:
            for spine in ax.spines.values():
                spine.set_visible(False)
        # Colorbar
        cbar = fig.colorbar(im, ax=ax_img, orientation='vertical', fraction=0.046, pad=0.04)
        plt.show()
        self.log_action(f"Displayed average profile of last {num_used} images with lineouts.")

    def update_spot_size_variation(self, spot_area):
        """
        Update the plot showing spot size variation (bottom of GUI).
        Each point is the percent change from the mean spot size.
        Only the latest 50 points are shown.
        """
        if not hasattr(self, 'spot_areas'):
            self.spot_areas = []
        if spot_area is not None:
            self.spot_areas.append(spot_area)
        # Keep only the last 50 datapoints
        self.spot_areas = self.spot_areas[-50:]
        # Calculate percentage variation from mean
        if len(self.spot_areas) > 1:
            mean_area = np.mean(self.spot_areas)
            percent_var = [(a - mean_area) / mean_area * 100 for a in self.spot_areas]
        else:
            percent_var = [0 for _ in self.spot_areas]
        self.variationPlot.clear()
        self.variationPlot.plot(percent_var, pen=pg.mkPen('b', width=2), symbol='o', symbolBrush='b')
        self.variationPlot.setTitle('Spot Size Variation (%)')

    def update_stats_panel(self):
        """
        Update the stats panel with mean/max variation and centroid motion for last 30 shots.
        """
        # Mean and max variation
        if hasattr(self, 'spot_areas') and len(self.spot_areas) >= 2:
            last = self.spot_areas[-30:]
            mean_area = np.mean(last)
            percent_var = [(a - mean_area) / mean_area * 100 for a in last]
            mean_var = np.mean(np.abs(percent_var))
            max_var = np.max(np.abs(percent_var))
            self.meanVarLabel.setText(f"Mean Variation: {mean_var:.2f} %")
            self.maxVarLabel.setText(f"Max Variation: {max_var:.2f} %")
        else:
            self.meanVarLabel.setText("Mean Variation: -- %")
            self.maxVarLabel.setText("Max Variation: -- %")
        # Centroid std
        if len(self.centroid_positions) >= 2:
            arr = np.array(self.centroid_positions[-30:])
            std_x = np.nanstd(arr[:,0])
            std_y = np.nanstd(arr[:,1])
            self.centroidStdLabelX.setText(f"Centroid Motion X: {std_x:.2f} mm")
            self.centroidStdLabelY.setText(f"Centroid Motion Y: {std_y:.2f} mm")
        else:
            self.centroidStdLabelX.setText("Centroid Motion X: -- mm")
            self.centroidStdLabelY.setText("Centroid Motion Y: -- mm")

    def reset_variation_data(self):
        """
        Reset the spot size variation and centroid motion data for a new dataset.
        """
        self.spot_areas = []
        self.centroid_positions = []
        self.variationPlot.clear()
        self.update_stats_panel()
        self.log_action("Spot size variation and centroid motion data reset.")

if __name__ == '__main__':
    # Create application instance and run the GUI
    app = QApplication(sys.argv)
    window = BeamProfileMonitor()
    window.show()
    sys.exit(app.exec())

    
# The GUI is now complete and can be run by executing the script. The user can interact with the GUI to select the image directory, calibration file, and adjust the threshold value. The beam profile images will be processed and displayed in real-time or batch mode based on the user's selection. The GUI provides a convenient way to monitor and analyze electron beam profile images for various applications.

# --- File selection group ---
# "Select Image Directory" button:
#   - Opens a dialog for the user to select a directory containing images.
#   - Sets self.directory and updates the status label.

# "Select Calibration File" button:
#   - Opens a dialog for the user to select a calibration file.
#   - Loads calibration data, computes transformation parameters, and updates the status label.

# --- Actions group ---
# "Start Monitoring" button (toggle):
#   - When pressed, starts monitoring the selected directory for new image files.
#   - When pressed again, stops monitoring.
#   - Updates the button text and status label accordingly.

# "Analyze Existing Images" button:
#   - Processes all existing image files in the selected directory (batch mode).
#   - For each image, runs the beam profile analysis and displays the result.

# --- Threshold group ---
# Slider ("Adjust Threshold:"):
#   - Lets the user adjust the threshold value for image analysis.
#   - Updates self.threshold, which affects how images are processed.

# --- Image display ---
# The processed image is shown in the "Processed Image" section after analysis.

# --- Output directory selection ---
# "Select Output Directory" button:
#   - Opens a dialog for the user to select a directory for saving analysis results.
#   - Sets self.output_directory and updates the label.

# --- Colormap selection ---
# Dropdown ("Colormap:"):
#   - Lets the user select a colormap for image display.
#   - Updates self.colormap, affecting the color mapping of processed images.

# --- Activity log ---
# Displays a log of key actions and events with timestamps.
# Logs are updated in real-time as actions are performed in the GUI.

# --- Spot Size Variation Plot ---
# Displays the variation of spot size (from ellipse fit) as a percentage.
# Updated with each processed image, showing the percentage variation from the mean spot size.

# --- Average Profile ---
# New controls for showing the average profile of the last N images.
# Uses matplotlib to display the average image in the processed image area.

# --- Bit Depth Selection ---
# New dropdown to select between 14-bit and 8-bit display modes.
# Affects the scaling of the colorbar and the displayed image intensity range.

# --- Monitoring Controls ---
# "Pause Monitoring" button (next to "Start Monitoring"):
#   - Pauses the monitoring of the directory when clicked.
#   - Resumes monitoring when clicked again.
#   - Updates its text and enabled state accordingly.

# --- Stats Panel ---
# Shows statistics about the spot size variation and centroid motion.
# Updated after each image is processed, showing real-time analysis stats.

# --- Reset Variation Button ---
# New "Reset Variation" button in the stats panel section.
# Resets the spot size and centroid data, clears the plot, and updates the stats when clicked.
