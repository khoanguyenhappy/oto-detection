import torch
import cv2
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk

# Hàm tải mô hình
def load_model(path):
    model = torch.load(path)
    model.eval()
    return model

# Hàm xử lý và dự đoán biển số xe
def predict_image(model, image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Tiền xử lý hình ảnh theo yêu cầu của mô hình
    # (Giả sử mô hình yêu cầu hình ảnh có kích thước 224x224)
    image = cv2.resize(image, (224, 224))
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float()
    with torch.no_grad():
        prediction = model(image)
    return prediction

# Hàm mở file và hiển thị hình ảnh
def open_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img = ImageTk.PhotoImage(img)
        panel.configure(image=img)
        panel.image = img
        prediction = predict_image(model, file_path)
        result_label.config(text=f"Prediction: {prediction}")

# Tạo giao diện tkinter
root = Tk()
root.title("License Plate Recognition")

# Tạo nút mở file
btn = Button(root, text="Open Image", command=open_file)
btn.pack()

# Tạo nhãn để hiển thị kết quả dự đoán
result_label = Label(root, text="Prediction: ")
result_label.pack()

# Tạo panel để hiển thị hình ảnh
panel = Label(root)
panel.pack()

# Tải mô hình
model = load_model("best.pt")

# Chạy ứng dụng tkinter
root.mainloop()
