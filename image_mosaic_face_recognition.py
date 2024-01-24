"""
Team8_IamImage_'image_mosaic_face_recognition.py'

To run the provided program, you need to install the required Python libraries. 
You can use the following command to install the necessary packages using pip:

pip install PyQt5 face_recognition opencv-python pillow
"""

import sys
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QFileDialog, QPushButton, QComboBox, QSpinBox
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor
import face_recognition
import cv2
from PIL import Image

class ImagePopup(QMainWindow):
    # Initialize the UI elements
    def __init__(self):
        super().__init__()
        self.selected_face_index = None  # Variable to store the index of the selected face
        self.face_locations = []  # List to store the locations of detected faces
        self.face_coordinates = []  # List to store the coordinates of recognized faces
        self.selected_faces = set()  # non mosaiced image
        self.mosaiced_image = None 
        self.scale_x = 0.0 # Ratio between the original image and the image displayed in the QLabel
        self.scale_y = 0.0
        self.init_ui()

    # Initialize the UI elements
    def init_ui(self):
        self.setWindowTitle('Image Mosaic')
        self.setGeometry(100, 100, 900, 750)  # Set the window size

        self.label = QLabel(self)
        self.label.setGeometry(20, 20, 760, 560)  #  Set QLabel size

        # Button 
        self.select_image_button = QPushButton('Select Image', self)
        self.select_image_button.setGeometry(20, 600, 150, 30)
        self.select_image_button.clicked.connect(self.select_image)

        self.apply_face_recognition_button = QPushButton('Face Recognition', self)
        self.apply_face_recognition_button.setGeometry(200, 600, 170, 30)
        self.apply_face_recognition_button.clicked.connect(self.apply_face_recognition)

        self.clear_image_button = QPushButton('Clear Image', self)
        self.clear_image_button.setGeometry(400, 600, 150, 30)
        self.clear_image_button.clicked.connect(self.clear_image)

        self.mosaic_button = QPushButton('Mosaic', self)
        self.mosaic_button.setGeometry(570, 600, 150, 30)
        self.mosaic_button.clicked.connect(self.apply_mosaic)
        self.mosaic_button.hide()

        self.pixelation_selector = QSpinBox(self)
        self.pixelation_selector.setGeometry(720, 600, 50, 30)
        self.pixelation_selector.setMinimum(5)
        self.pixelation_selector.setMaximum(50)
        self.pixelation_selector.setValue(15)
        self.pixelation_selector.hide()
    
    # Handle mouse press event
    def mousePressEvent(self, event):
        """
        Handles mouse press events to select or deselect the area of the face you want to mosaic.
        When a mouse click occurs, it calculates the scaled coordinates based on the original image size and the QLabel size.
        """
        x_click, y_click = event.x(), event.y()
        x_click *= self.scale_x
        y_click *= self.scale_y

        for idx, (top, right, bottom, left) in enumerate(self.face_locations):
            if left <= x_click <= right and top <= y_click <= bottom:
                if idx in self.selected_faces:
                    self.selected_faces.remove(idx)
                else:
                    self.selected_faces.add(idx)

                self.redraw_faces()
                break
    
    # Redraw faces on the image
    def redraw_faces(self):
        """
        Redraws faces on the image, highlighting selected faces in green and unselected faces in red.
        If a face is selected, it will be outlined in green; otherwise, it will be outlined in red.
        """
        image = face_recognition.load_image_file(self.current_image_path)
        q_image = QImage(image.data, image.shape[1], image.shape[0], image.shape[1] * 3, QImage.Format_RGB888)

        painter = QPainter(q_image)
        font = painter.font()
        font.setFamily("Liberation Sans")
        font.setPixelSize(25)
        painter.setFont(font)

        for idx, (top, right, bottom, left) in enumerate(self.face_locations):
            pen = QPen()
            pen.setWidth(3)

            if idx in self.selected_faces:
                pen.setColor(QColor(0, 255, 0))  # Green for selected faces
            else:
                pen.setColor(QColor(255, 0, 0))  # Red for unselected faces

            painter.setPen(pen)
            painter.drawRect(left, top, right - left, bottom - top)

        painter.end()
        pixmap_with_rectangles = QPixmap.fromImage(q_image)
        self.label.setPixmap(pixmap_with_rectangles.scaled(self.label.width(), self.label.height()))
    
    # Handle image selection  
    def select_image(self):
        """
        Opens a file dialog to select an image file and displays it in the QLabel.
        When the "Select Image" button is clicked, it opens a file dialog to choose an image file.
        """
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp *.gif)", options=options)

        if file_name:
            self.clear_image()
            self.current_image_path = file_name
            pixmap = QPixmap(file_name)
            original_width = pixmap.width()
            original_height = pixmap.height()
            self.scale_x = original_width / self.label.width()
            self.scale_y = original_height / self.label.height()
            self.label.setPixmap(pixmap.scaled(self.label.width(), self.label.height()))
    
    # Apply face recognition to the selected image
    def apply_face_recognition(self):
        """
        Applies face recognition to the selected image.
        Loads the selected image and performs face recognition using the face_recognition library.
        Draws rectangles around detected faces and displays the image with rectangles in the QLabel.
        Displays the 'Mosaic' button and pixelation selector for further processing.
        """
        if not self.label.pixmap():
            return
        
        self.face_coordinates = []  

        # Load the image
        image = face_recognition.load_image_file(self.current_image_path)

        # Face recognition
        face_locations = face_recognition.face_locations(image)
        self.face_locations = face_recognition.face_locations(image) 

        sorted_face_locations = sorted(face_locations, key=lambda x: x[3])
        
        # Convert image to QImage
        height, width, channel = image.shape
        q_image = QImage(image.data, width, height, channel * width, QImage.Format_RGB888)

        # Create QPainter to draw rectangles and numbers
        painter = QPainter(q_image)
        pen = QPen()
        pen.setWidth(3)  # Increase the width for better visibility
        pen.setColor(QColor(255, 0, 0))  # Red color for rectangles
        painter.setPen(pen)
        
        # Draw rectangle around the face
        for face_location in sorted_face_locations:
            top, right, bottom, left = face_location
            painter.drawRect(left, top, right - left, bottom - top)
            self.face_coordinates.append((left, top, right, bottom))     

        painter.end()

        # Convert QPixmap and display in the label
        pixmap_with_rectangles = QPixmap.fromImage(q_image)
        self.label.setPixmap(pixmap_with_rectangles.scaled(self.label.width(), self.label.height()))

        if self.face_locations:
            self.mosaic_button.setText('Mosaic')
            self.mosaic_button.show()
            self.pixelation_selector.show()
    
    # Apply mosaic effect to the image
    def apply_mosaic(self):
        """
        Applies the mosaic effect to the image.
        If a mosaiced image exists, the function allows the user to save it.
        If not, it loads the selected image, applies the mosaic effect to the non-selected faces,
        and displays the mosaiced image in the QLabel. The user can then choose to save the mosaiced image.
        """
        if self.mosaiced_image is not None:  # Save mosaiced image
            options = QFileDialog.Options()
            file_name, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "JPG Files (*.jpg);;PNG Files (*.png);;All Files (*)", options=options)
            if file_name:
                height, width, _ = self.mosaiced_image.shape
                for j in range(height):
                    for i in range(width):
                        b, g, r = self.mosaiced_image[j][i]
                        self.mosaiced_image[j][i] = (r, g, b)
                image = Image.fromarray(self.mosaiced_image)
                image.save(file_name)
            return

        image = cv2.imread(self.current_image_path)
        if image is None:
            print(f"Failed to load image from {self.current_image_path}")
            return
        
        for idx, face_location in enumerate(self.face_locations):
            if idx not in self.selected_faces:
                top, right, bottom, left = face_location
                self.apply_mosaic_to_face(image, top, right, bottom, left, self.pixelation_selector.value())

        self.mosaiced_image = image
        self.display_image(image)
        self.mosaic_button.setText('Save Mosaiced Image')
    
    # Apply mosaic effect to a specific face region
    def apply_mosaic_to_face(self, img, top, right, bottom, left, pixelation):
        """
        Applies the mosaic effect to a specific face region.
        The function extracts the specified face region, resizes it using linear interpolation to create the mosaic effect,
        and then applies it back to the original image.
        """
        face_img = img[top:bottom, left:right]
        face_img = cv2.resize(face_img, (pixelation, pixelation), interpolation=cv2.INTER_LINEAR)
        face_img = cv2.resize(face_img, (right - left, bottom - top), interpolation=cv2.INTER_NEAREST)
        img[top:bottom, left:right] = face_img
    
    # Display the image in the label
    def display_image(self, img):
        height, width, channel = img.shape
        bytesPerLine = channel * width
        q_image = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_image)
        self.label.setPixmap(pixmap.scaled(self.label.width(), self.label.height()))
    
    # Clear the label and reset variables
    def clear_image(self):
        self.label.clear()

        self.mosaic_button.hide()
        self.pixelation_selector.hide()

        self.selected_face_index = None
        self.face_locations = []
        self.face_coordinates = []
        self.selected_faces = set()
        self.mosaiced_image = None


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImagePopup()
    window.show()
    sys.exit(app.exec_())