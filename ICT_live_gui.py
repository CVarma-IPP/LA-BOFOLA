import sys               # For application startup/exit
import os                # For file path manipulations
import time              # To pause briefly while files are written
from datetime import datetime      # To timestamp output files
from collections import deque      # Efficient fixed-length queue for plotting

# PyQt5 widgets and layouts
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTextEdit, QDoubleSpinBox,
    QGroupBox, QFormLayout, QSizePolicy
)
from PyQt5.QtCore import Qt  # For alignment and orientation flags

# Watch a folder for new files
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Your analysis routine for each new ICT data file
import ICT_live_analysis as ict_analysis

# Embed a Matplotlib figure inside Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class PlotCanvas(FigureCanvas):
    """
    A plot area embedded in the Qt window,
    showing “Charge vs. Shot Number” in real time.
    """
    def __init__(self, parent=None, maxshots: int = 50):
        # Create a Matplotlib Figure and attach it
        self.fig = Figure(figsize=(5, 3))
        super().__init__(self.fig)
        self.setParent(parent)

        # One subplot for our line‐plot
        self.ax = self.fig.add_subplot(111)

        # Keep only the last `maxshots` points for performance
        self.maxshots = maxshots
        self.shots = deque(maxlen=maxshots)    # shot indexes
        self.charges = deque(maxlen=maxshots)  # corresponding charges

        # Label axes and show grid lines
        self.ax.set_xlabel('Shot Number')
        self.ax.set_ylabel('Charge (pC)')
        self.ax.grid(True)
        self.fig.tight_layout()

    def update_plot(self, shot_number: int, charge: float):
        """
        Called whenever a new charge measurement arrives.
        - Append the new (shot, charge) pair
        - Redraw the line plot showing only the last N shots
        """
        self.shots.append(shot_number)
        self.charges.append(charge)

        # Clear old drawing and plot updated data
        self.ax.clear()
        self.ax.plot(list(self.shots), list(self.charges), marker='o')

        # Adjust x-axis to exactly cover the recent shots
        if self.shots:
            self.ax.set_xlim(self.shots[0], self.shots[-1])

        # Reapply labels and grid then render
        self.ax.set_xlabel('Shot Number')
        self.ax.set_ylabel('Charge (pC)')
        self.ax.grid(True)
        self.draw()


class NewFileHandler(FileSystemEventHandler):
    """
    Watches a directory for new data files. When one appears:
     1. Reads the new file
     2. Runs your ICT analysis routine
     3. Logs success or error to the text box
     4. Saves a small _analysis_ text file with the computed charge
     5. Tells the plot to update with the new point
    """
    def __init__(self, out_dir: str, log_widget: QTextEdit,
                 plot_canvas: PlotCanvas, param_getter):
        self.out_dir = out_dir                # where to write analysis results
        self.log = log_widget                 # text box for status messages
        self.plot_canvas = plot_canvas        # live-update plot
        self.param_getter = param_getter      # function returning two calibration numbers
        self.shot_counter = 0                 # how many files we’ve processed

    def on_created(self, event):
        # Ignore directories
        if event.is_directory:
            return

        path = event.src_path
        # Only care about data files with these extensions
        if not path.lower().endswith(('.csv', '.txt', '.dat', '.wave')):
            return

        # Start analysis in a separate thread for responsiveness
        from threading import Thread
        Thread(target=self._analyze_file_thread, args=(path,)).start()

    def _analyze_file_thread(self, path):
        import shutil
        # Wait until file is stable (not growing) before processing
        stable = False
        last_size = -1
        for _ in range(20):  # up to 2 seconds
            try:
                size = os.path.getsize(path)
            except Exception:
                time.sleep(0.05)
                continue
            if size == last_size and size > 0:
                stable = True
                break
            last_size = size
            time.sleep(0.1)
        if not stable:
            self.log.append(f"[ERROR] File not stable: {os.path.basename(path)}")
            return

        # Pull current calibration values from the GUI and apply them
        val1, val2 = self.param_getter()
        ict_analysis.VAL1 = val1
        ict_analysis.VAL2 = val2

        # Try to analyze the new data file
        try:
            charge = ict_analysis.analyze_file(path)
        except Exception as e:
            # On error, show a red “ERROR” line in the log and move file to error folder
            self.log.append(f"[ERROR] analyzing {os.path.basename(path)}: {e}")
            error_dir = os.path.join(self.out_dir, "error_files")
            os.makedirs(error_dir, exist_ok=True)
            try:
                shutil.move(path, os.path.join(error_dir, os.path.basename(path)))
            except Exception as move_err:
                self.log.append(f"[ERROR] Could not move file to error folder: {move_err}")
            return

        # Successful: increment our shot count
        self.shot_counter += 1

        # Build an output filename with timestamp
        base = os.path.basename(path)
        name, _ = os.path.splitext(base)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name = f"{name}_analysis_{timestamp}.txt"
        out_path = os.path.join(self.out_dir, out_name)

        # Write the analysis result (charge in pC) to that file
        try:
            with open(out_path, 'w') as f:
                f.write(f"charge_pC: {charge:.6f}\n")
        except Exception as e:
            self.log.append(f"[ERROR] Could not write analysis file: {e}")
            return

        # Log success and update the live plot
        self.log.append(f"✔ {base} → {out_name} (Charge: {charge:.2f} pC)")
        self.plot_canvas.update_plot(self.shot_counter, charge)


class MainWindow(QMainWindow):
    """
    The main application window:
     - Lets you choose an input folder to watch
     - Lets you choose an output folder for results
     - Shows your two calibration spin-boxes
     - Embeds the live plot and the activity log
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live ICT Monitor")
        self.resize(900, 600)

        # Track which folders are selected
        self.in_dir = ''
        self.out_dir = ''

        # --- Main layout: horizontal split ---
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(18)

        # --- Sidebar for controls ---
        sidebar = QVBoxLayout()
        sidebar.setSpacing(12)
        sidebar.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Calibration group
        calib_group = QGroupBox("Calibration Parameters")
        calib_group.setStyleSheet("font-weight: bold;")
        form = QFormLayout()
        self.spin_val1 = QDoubleSpinBox()
        self.spin_val1.setDecimals(6)
        self.spin_val1.setRange(0.0, 100.0)
        self.spin_val1.setValue(ict_analysis.VAL1)
        self.spin_val2 = QDoubleSpinBox()
        self.spin_val2.setDecimals(6)
        self.spin_val2.setRange(0.0, 100.0)
        self.spin_val2.setValue(ict_analysis.VAL2)
        form.addRow("VAL1:", self.spin_val1)
        form.addRow("VAL2:", self.spin_val2)
        calib_group.setLayout(form)
        sidebar.addWidget(calib_group)

        # Folder selection group
        folder_group = QGroupBox("Folder Selection")
        folder_layout = QVBoxLayout()
        self.btn_in = QPushButton("Select Input Folder…")
        self.lbl_in = QLabel("<i>No input folder selected</i>")
        self.btn_out = QPushButton("Select Output Folder…")
        self.lbl_out = QLabel("<i>No output folder selected</i>")
        folder_layout.addWidget(self.btn_in)
        folder_layout.addWidget(self.lbl_in)
        folder_layout.addWidget(self.btn_out)
        folder_layout.addWidget(self.lbl_out)
        folder_group.setLayout(folder_layout)
        sidebar.addWidget(folder_group)

        # Activity log group
        log_group = QGroupBox("Activity Log")
        log_layout = QVBoxLayout()
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        log_layout.addWidget(self.log)
        log_group.setLayout(log_layout)
        sidebar.addWidget(log_group)

        sidebar.addStretch(1)

        # --- Main plot area ---
        plot_group = QGroupBox("Charge vs. Shot Number")
        plot_layout = QVBoxLayout()
        self.plot_canvas = PlotCanvas(self, maxshots=50)
        self.plot_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        plot_layout.addWidget(self.plot_canvas)
        plot_group.setLayout(plot_layout)

        # --- Compose main layout ---
        main_layout.addLayout(sidebar, 0)
        main_layout.addWidget(plot_group, 1)

        # --- Set color scheme and style ---
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
            QPushButton:hover {
                background-color: #74b9ff;
            }
            QLabel {
                color: #2d3436;
            }
            QTextEdit {
                background: #f8fafd;
                border-radius: 6px;
                font-size: 10pt;
            }
        """)

        # --- Set main widget ---
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # --- Connect button clicks to folder-choosing methods ---
        self.btn_in.clicked.connect(self.choose_input)
        self.btn_out.clicked.connect(self.choose_output)

        # Placeholder for the folder-watch observer
        self.observer = None

    def choose_input(self):
        """Open a dialog to pick the folder where new data appears."""
        folder = QFileDialog.getExistingDirectory(self, "Select input folder")
        if folder:
            self.in_dir = folder
            self.lbl_in.setText(folder)
            self.restart_observer()

    def choose_output(self):
        """Open a dialog to pick where to save analysis results."""
        folder = QFileDialog.getExistingDirectory(self, "Select output folder")
        if folder:
            self.out_dir = folder
            self.lbl_out.setText(folder)
            self.restart_observer()

    def restart_observer(self):
        """
        Stop any existing folder-watcher, then if both
        input+output folders are set, start watching for new files.
        """
        if self.observer:
            self.observer.stop()
            self.observer.join()

        if self.in_dir and self.out_dir:
            handler = NewFileHandler(
                self.out_dir,
                self.log,
                self.plot_canvas,
                # param_getter returns the two spinbox values
                param_getter=lambda: (self.spin_val1.value(), self.spin_val2.value())
            )
            self.observer = Observer()
            self.observer.schedule(handler, self.in_dir, recursive=False)
            self.observer.start()
            self.log.append(f"▶ Watching {self.in_dir}\n  saving analyses to {self.out_dir}")

    def closeEvent(self, event):
        """Cleanly stop the folder-watcher when the window closes."""
        if self.observer:
            self.observer.stop()
            self.observer.join()
        super().closeEvent(event)


if __name__ == '__main__':
    # Standard Qt application bootstrap
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())