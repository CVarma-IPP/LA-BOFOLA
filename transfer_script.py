import easygui
import shutil
import os
import time

def main():
    # Ask user to select files
    files = easygui.fileopenbox(
        msg="Select files to copy",
        title="Select Files",
        multiple=True
    )
    if not files:
        easygui.msgbox("No files selected. Exiting.")
        return

    # Ask user to select output folder
    output_folder = easygui.diropenbox(
        msg="Select output folder",
        title="Select Output Folder"
    )
    if not output_folder:
        easygui.msgbox("No output folder selected. Exiting.")
        return

    # Copy files at 1 Hz
    for file_path in files:
        filename = os.path.basename(file_path)
        dest_path = os.path.join(output_folder, filename)
        shutil.copy2(file_path, dest_path)
        time.sleep(2)  # 1 Hz = 1 file per second

    easygui.msgbox("All files copied successfully.")

if __name__ == "__main__":
    main()