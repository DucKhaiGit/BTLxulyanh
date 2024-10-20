import cv2
import numpy as np
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk


# Hàm đọc và hiển thị ảnh
def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        # Đọc ảnh bằng OpenCV
        img = cv2.imread(file_path)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Đảm bảo kích thước ảnh là bội số của kích thước khối lớn nhất (256x256)
        gray_img = cv2.resize(gray_img, (256, 256))

        # Hiển thị ảnh gốc
        display_image(img, selected_label)

        # Chạy phân vùng ảnh sau khi nhấn nút
        segmented_img = split_and_merge(gray_img, threshold=10)  # Thay đổi threshold
        boundaries_img = draw_boundaries(gray_img, segmented_img)

        # Hiển thị ảnh phân vùng và ảnh với đường phân chia
        display_image(boundaries_img, boundaries_label)
        display_image(segmented_img, segmented_label)


# Hàm hiển thị ảnh trong giao diện Tkinter
def display_image(img, label):
    img = cv2.resize(img, (250, 250))  # Thiết lập kích thước cố định
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Chuyển từ BGR sang RGB
    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(img)
    label.config(image=img)
    label.image = img


# Hàm vẽ đường phân chia trên ảnh
def draw_boundaries(img, segmented_img):
    contours, _ = cv2.findContours(cv2.Canny(segmented_img, 100, 200), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_with_boundaries = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img_with_boundaries, contours, -1, (0, 255, 0), 1)  # Vẽ đường phân chia mỏng hơn (1 pixel)
    return img_with_boundaries


# Hàm chia và hợp vùng (phân vùng ảnh)
def split_and_merge(img, threshold=10):  # Thay đổi ngưỡng cho hiệu quả phân vùng
    h, w = img.shape
    result = np.zeros_like(img)

    # Đệ quy chia vùng
    def split(x, y, w, h, min_size=1):
        region = img[y:y + h, x:x + w]
        if is_homogeneous(region, threshold) or w <= min_size or h <= min_size:
            # Thay đổi cách tính giá trị vùng
            result[y:y + h, x:x + w] = np.mean(region) + np.random.randint(-20,
                                                                           20)  # Thêm sự ngẫu nhiên vào giá trị vùng
        else:
            hw, hh = w // 2, h // 2
            split(x, y, hw, hh)
            split(x + hw, y, hw, hh)
            split(x, y + hh, hw, hh)
            split(x + hw, y + hh, hw, hh)

    def is_homogeneous(region, threshold):
        mean = np.mean(region)
        stddev = np.std(region)
        return stddev < threshold

    split(0, 0, w, h)

    return result


# Tạo giao diện bằng Tkinter
root = Tk()
root.title("Image Segmentation App")

# Thiết lập kích thước cửa sổ
root.geometry("600x700")

# Tạo label để hiển thị ảnh được chọn
selected_label = Label(root)
selected_label.pack(pady=10)

# Tạo button để chọn và phân vùng ảnh
open_button = Button(root, text="Chọn Ảnh", command=open_image, font=("Arial", 14))
open_button.pack(pady=10)

# Tạo hai label để hiển thị ảnh phân vùng và ảnh với đường phân chia
boundaries_label = Label(root)
boundaries_label.pack(side="left", padx=10, pady=10)

segmented_label = Label(root)
segmented_label.pack(side="right", padx=10, pady=10)

# Chạy ứng dụng
root.mainloop()
