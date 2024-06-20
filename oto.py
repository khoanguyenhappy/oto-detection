import cv2
import torch
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk

# Load the YOLO v5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp18/weights/best.pt',
                       force_reload=True)

# Create the Tkinter application
root = Tk()
root.title("YOLO v5 Video Detection")


# Function to open a file dialog and select a video file
def open_file():
    video_path = filedialog.askopenfilename()
    if video_path:
        process_video(video_path)


# Function to process and display the video
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error, không thể mở video")
        return

    def show_frame():
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return

        # Perform object detection
        results = model(frame)
        annotated_frame = results.render()[0]

        # Convert frame to ImageTk format
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(annotated_frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        # Update the label with the new frame
        label.imgtk = imgtk
        label.configure(image=imgtk)
        label.after(10, show_frame)

    # Create a label to display the video frames
    label = Label(root)
    label.pack()
    show_frame()


# Create a button to open the file dialog
button = Button(root, text="Open Video File", command=open_file)
button.pack()

# Start the Tkinter event loop
root.mainloop()
