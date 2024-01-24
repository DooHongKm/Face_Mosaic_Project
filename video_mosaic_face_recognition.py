"""
Team8_IamImage_'video_mosaic_face_recognition.py'

To run the provided program, you need to install the required Python libraries. 
You can use the following command to install the necessary packages using pip:

pip install opencv-python
pip install numpy
pip install imageio
pip install face-recognition
pip install PyQt5
pip install pillow
pip install moviepy
"""

import sys
import os
import cv2
import numpy as np
import imageio
import face_recognition
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QApplication,QLabel,QMainWindow,QVBoxLayout,QWidget,QPushButton,QFileDialog,QHBoxLayout,)
from PIL import Image
from moviepy.editor import ImageSequenceClip

# Get a list of file paths in the specified folder
def get_files_in_folder(folder_path):
    try:
        file_list = os.listdir(folder_path)
        file_paths = [os.path.join(folder_path, file) for file in file_list]
        return file_paths
    except Exception as e:
        print(f"Error: {e}")
        return None

# The subjects in the image are not the mosaic targets
def user_mosaic(image, location):
    
    return image
# Apply the mosaic effect to regions other than the identified person in the image
def others_mosaic(image, location):
    """
    This function takes an input image and the location of an identified person.
    It applies a mosaic effect to regions other than the identified person's location and returns the modified image.
    """
    result_image = image.copy()
    top, right, bottom, left = location
    width = right - left + 1
    height = bottom - top + 1
    
    window_size = width // 20
    xstep = width // window_size
    ystep = height // window_size
    
    for j in range(ystep):
        for i in range(xstep):
            window = result_image[top + j * window_size:top + (j + 1) * window_size, left + i * window_size:left + (i + 1) * window_size, :]
            channel_sums = np.sum(window, axis=(0, 1))
            count = window_size * window_size
            averages = channel_sums // count
            result_image[top + j * window_size:top + (j + 1) * window_size, left + i * window_size:left + (i + 1) * window_size, :] = averages
            
    return result_image

# Analysis results of a single frame and apply them to the remaining two frames
def process_point_frame(image, folder_path):
    """
    Analyzes the input frame, identifies faces, and applies mosaic effects based on known face encodings.
    Args:
        image (numpy.ndarray): The input frame image.
        folder_path (str): The path to the folder containing known face images.
    Returns:
        tuple: A tuple containing the processed image, face locations, 
        and a list of booleans indicating whether each face was identified.
    """
    result_image = image.copy()
    unknown_face_locations = face_recognition.face_locations(image)

    known_face_encodings = []
    paths = get_files_in_folder(folder_path)
    for path in paths:
        img = face_recognition.load_image_file(path)
        encoding = face_recognition.face_encodings(img)[0]
        known_face_encodings.append(encoding)

    unkown_face_encodings = face_recognition.face_encodings(image, unknown_face_locations)

    similarities = []
    for unknown_face_location, unknown_face_encoding in zip(unknown_face_locations, unkown_face_encodings):
        distance = face_recognition.face_distance(known_face_encodings, unknown_face_encoding)
        minimum = min(distance)

        if minimum < 0.6:
            result_image = user_mosaic(result_image, unknown_face_location)
            similarities.append(True)
        else:
            result_image = others_mosaic(result_image, unknown_face_location)
            similarities.append(False)

    return result_image, unknown_face_locations, similarities

# Transform the analyzed results into images for the remaining two frames
def process_other_frame(image, face_locations, similarities):
    result_image = image.copy()

    for face_location, similarity in zip(face_locations, similarities):
        if similarity:
            result_image = user_mosaic(result_image, face_location)
        else:
            result_image = others_mosaic(result_image, face_location)

    return result_image

# Convert a video to a list of frames along with its frames per second (fps) information
def video_to_frames(video_path):
    frames = []
    video_reader = imageio.get_reader(video_path)
    fps = video_reader.get_meta_data()['fps']

    for frame in video_reader:
        frames.append(frame)

    return frames, fps

# Process a list of frames by analyzing every third frame and applying the results to the remaining two frames
def process_frames(frames, folder_path):
    result = []
    face_locations = []
    similarities = []

    for idx, frame in enumerate(frames):
        if idx % 3 == 0:
            f, face_locations, similarities = process_point_frame(frame, folder_path)
            result.append(f)
        else:
            f = process_other_frame(frame, face_locations, similarities)
            result.append(f)

    return result

# Conver a list of frames to a video
def frames_to_video(frames, output_path, fps):
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(output_path, codec='libx264', audio=False)

class ExifOrientation:
    @staticmethod
    def apply_orientation(image):
        try:
            exif_orientation_tag = 0x0112
            orientation = image._getexif().get(exif_orientation_tag, 1)

            if orientation == 3:
                return image.transpose(Image.ROTATE_180)
            elif orientation == 6:
                return image.transpose(Image.ROTATE_270)
            elif orientation == 8:
                return image.transpose(Image.ROTATE_90)
        except (AttributeError, KeyError, IndexError):
            pass

        return image

class MediaViewer(QMainWindow):
    image_folder_path_changed = pyqtSignal(str)
    video_folder_path_changed = pyqtSignal(str)

    def __init__(self):
        super().__init__()

        self.current_image_path = ""
        self.current_video_path = ""

        self.setWindowTitle("Video Mosaic")
        self.setGeometry(100, 100, 1500, 900)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

        self.labels_layout = QHBoxLayout()

        # image folder path label
        self.image_folder_path_label = QLabel("Selected image folder: Empty", self.central_widget)
        self.image_folder_path_label.setStyleSheet("font-size: 25px;")
        self.layout.addWidget(self.image_folder_path_label)

        # Video before mosaic label
        self.video_label = QLabel("The video to be mosaiced will appear here", self.central_widget)
        self.video_label.setStyleSheet("font-size: 20px;")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setFixedSize(800, 600)
        self.labels_layout.addWidget(self.video_label)

        # Video after mosaic label
        self.processed_video_label = QLabel("The mosaiced video will appear here", self.central_widget)
        self.processed_video_label.setStyleSheet("font-size: 20px;")
        self.processed_video_label.setAlignment(Qt.AlignCenter)
        self.processed_video_label.setFixedSize(800, 600)
        self.labels_layout.addWidget(self.processed_video_label)

        self.layout.addLayout(self.labels_layout)

        # Button
        self.buttons_layout = QHBoxLayout()

        self.load_image_button = QPushButton("Select image folder", self.central_widget)
        self.load_image_button.clicked.connect(self.load_and_display_image)
        font = self.load_image_button.font()
        font.setPointSize(14)  
        self.load_image_button.setFont(font)
        self.buttons_layout.addWidget(self.load_image_button)

        self.load_video_button = QPushButton("Load Video", self.central_widget)
        self.load_video_button.clicked.connect(self.load_and_display_video)
        font = self.load_video_button.font()
        font.setPointSize(14)  
        self.load_video_button.setFont(font)
        self.buttons_layout.addWidget(self.load_video_button)

        self.convert_button = QPushButton("Convert", self.central_widget)
        self.convert_button.clicked.connect(self.convert_function)

        font = self.convert_button.font()
        font.setPointSize(14)
        self.convert_button.setFont(font)
        self.buttons_layout.addWidget(self.convert_button)

        self.load_image_button.setFixedSize(150, 30)
        self.load_video_button.setFixedSize(150, 30)
        self.convert_button.setFixedSize(150, 30)
        self.load_image_button.setFixedSize(200, 40)
        self.load_video_button.setFixedSize(200, 40)
        self.convert_button.setFixedSize(200, 40)

        self.buttons_layout.setAlignment(Qt.AlignCenter)

        self.layout.addLayout(self.buttons_layout)

        self.video_capture = None
        self.timer = QTimer(self)

        self.frames = []

    # Load and display an image folder path
    def load_and_display_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        options |= QFileDialog.DontUseNativeDialog

        folder_path = QFileDialog.getExistingDirectory(
            self, "Select image folder", options=options
        )

        if folder_path:
            self.image_folder_path_label.setText(f"Selected image folder: {folder_path}")
            print("Complete selecting image folder:", folder_path)

        self.image_paths = get_files_in_folder(folder_path)

        self.image_folder_path_changed.emit(folder_path)

    # Load and display a avideo
    def load_and_display_video(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly

        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mkv);;All Files (*)", options=options
        )

        if file_name:
            _, file_extension = os.path.splitext(file_name)
            if file_extension.lower() not in ('.mp4', '.avi', '.mkv'):
                print("Error: The selected file is not a supported video file.")
                return

            print("Selected video name:", file_name)
            if self.video_capture and self.video_capture.isOpened():
                self.video_capture.release()

            self.video_capture = cv2.VideoCapture(file_name)

            if not self.video_capture.isOpened():
                print("Error: Unable to open video capture.") 
                return
            
            folder_path = os.path.dirname(file_name)
            self.video_paths = get_files_in_folder(folder_path)

            self.video_folder_path_changed.emit(file_name) 

            # Read the first frame of the video
            ret, frame = self.video_capture.read()
            if not ret:
                print("Error: Unable to read video frame.")
                return

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Define the target display size
            target_width, target_height = 800, 600
            current_height, current_width, _ = frame_rgb.shape
            aspect_ratio = current_width / current_height
            new_width = int(target_height * aspect_ratio)
            new_height = target_height
            frame_rgb_resized = cv2.resize(frame_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)

            # Resize the image to fit the target display size
            frame_rgb_resized = cv2.resize(frame_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)

            # Convert the resized frame to QImage and display it
            height, width, channel = frame_rgb_resized.shape
            bytes_per_line = 3 * width
            q_image = QImage(frame_rgb_resized.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.video_label.setPixmap(pixmap.scaled(target_width, target_height, Qt.KeepAspectRatio))

            self.timer.stop()

    # Convert face to mosaic face
    def convert_function(self):
        """
        Perform the conversion process by applying mosaic to the faces in the video.
        """
        global video_path, folder_path, output_video_path

        if not video_path or not folder_path:
            error_message = "Error: Both video and image folder paths must be correctly specified."
            self.show_conversion_progress(error_message)
            print(error_message)
            return

        self.show_conversion_progress("Conversion in progress")

        frames, fps = video_to_frames(video_path)
        processed_frames = process_frames(frames, folder_path)

        # 수정된 코드: 처리된 비디오로 변환 및 output_video_path 출력
        frames_to_video(processed_frames, output_video_path, fps)
        self.show_conversion_progress(f"Conversion completed - Output Video Path: {output_video_path}")

        # 첫 번째 프레임 저장
        self.save_first_frame(output_video_path)

        print("Conversion completed!")

    # Save and display the first frame
    def save_first_frame(self, video_path):
        # Read first frame from the video
        video_capture = cv2.VideoCapture(video_path)
        ret, frame = video_capture.read()

        # Save first frame
        if ret:
            cv2.imwrite('first_frame.jpg', frame)
            print("First frame saved successfully.")

            # Convert the first frame to QImage and display it in the QLabel
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = frame_rgb.shape
            bytes_per_line = 3 * width
            q_image = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.processed_video_label.setPixmap(pixmap.scaled(800, 600, Qt.KeepAspectRatio))
        else:
            print("Error: Unable to read the frame.")

        video_capture.release()

    # Display a progress of conversion
    def show_conversion_progress(self, message):
        self.statusBar().showMessage(message)


video_path = ''
folder_path = ""
output_video_path = 'result_video.mp4'

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = MediaViewer()
    viewer.show()

    def on_image_folder_path_changed(image_folder_path):
        global folder_path
        folder_path = image_folder_path
        print("Image Folder Path:", folder_path)

    def on_video_folder_path_changed(new_video_path):
        global video_path 
        video_path = new_video_path 
        file_name = os.path.basename(video_path)
        print("Video File Path:", video_path)
        print("Video File Name:", file_name)

    viewer.image_folder_path_changed.connect(on_image_folder_path_changed)
    viewer.video_folder_path_changed.connect(on_video_folder_path_changed)

    sys.exit(app.exec_())