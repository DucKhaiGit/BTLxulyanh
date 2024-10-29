import cv2
import numpy as np
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk


# Hàm đọc và hiển thị ảnh
def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = cv2.imread(file_path)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.resize(gray_img, (256, 256))

        # Hiển thị ảnh gốc
        display_image(img, selected_label)

        # Chia ảnh thành vùng
        split_img = split_image(gray_img)
        display_image(split_img, split_label)

        # Hợp các vùng sau chia
        segmented_img = merge_regions(split_img, threshold=10)
        boundaries_img = draw_boundaries(gray_img, segmented_img)

        # Hiển thị ảnh với đường phân chia và ảnh sau hợp vùng
        display_image(boundaries_img, boundaries_label)
        display_image(segmented_img, segmented_label)


# Hàm hiển thị ảnh trong giao diện Tkinter
def display_image(img, label):
    img = cv2.resize(img, (250, 250))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(img)
    label.config(image=img)
    label.image = img


# Hàm kiểm tra tính đồng nhất của vùng
def is_homogeneous(region, threshold=10):
    mean = np.mean(region)
    stddev = np.std(region)
    return stddev < threshold


# Hàm chia ảnh
def split_image(img):
    h, w = img.shape
    split_result = np.zeros_like(img)

    # Đệ quy chia vùng
    def split(x, y, w, h, min_size=1):
        region = img[y:y + h, x:x + w]
        if is_homogeneous(region, threshold=10) or w <= min_size or h <= min_size:
            split_result[y:y + h, x:x + w] = np.mean(region)
        else:
            hw, hh = w // 2, h // 2
            split(x, y, hw, hh)
            split(x + hw, y, hw, hh)
            split(x, y + hh, hw, hh)
            split(x + hw, y + hh, hw, hh)

    split(0, 0, w, h)
    return split_result


# Hàm hợp vùng
def merge_regions(split_img, threshold=15):
    h, w = split_img.shape
    merged_result = np.copy(split_img)

    # Quét qua từng khối (kích thước khối 16x16) để tìm các vùng tương đồng
    block_size = 16
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            region = split_img[y:y+block_size, x:x+block_size]
            mean_intensity = np.mean(region)

            # Kiểm tra các vùng lân cận và hợp nếu giống nhau
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    ny, nx = y + dy * block_size, x + dx * block_size
                    if 0 <= ny < h and 0 <= nx < w and (dy != 0 or dx != 0):
                        neighbor_region = split_img[ny:ny+block_size, nx:nx+block_size]
                        neighbor_mean_intensity = np.mean(neighbor_region)

                        # Nếu vùng lân cận có độ sáng gần giống, hợp lại
                        if abs(mean_intensity - neighbor_mean_intensity) < threshold:
                            merged_result[ny:ny+block_size, nx:nx+block_size] = mean_intensity

    return merged_result



# Hàm vẽ đường phân chia trên ảnh
def draw_boundaries(img, segmented_img):
    contours, _ = cv2.findContours(cv2.Canny(segmented_img, 100, 200), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_with_boundaries = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img_with_boundaries, contours, -1, (0, 255, 0), 1)
    return img_with_boundaries


# Tạo giao diện bằng Tkinter
root = Tk()
root.title("Image Segmentation App")

# Thiết lập kích thước cửa sổ
root.geometry("800x800")

# Tạo label để hiển thị ảnh được chọn
selected_label = Label(root)
selected_label.pack(pady=10)

# Tạo button để chọn và phân vùng ảnh
open_button = Button(root, text="Chọn Ảnh", command=open_image, font=("Arial", 14))
open_button.pack(pady=10)

# Tạo các label để hiển thị ảnh sau chia, ảnh sau hợp, và ảnh với đường phân chia

boundaries_label = Label(root)
boundaries_label.pack(side="left", padx=10, pady=10)

split_label = Label(root)
split_label.pack(side="left", padx=10, pady=10)



segmented_label = Label(root)
segmented_label.pack(side="right", padx=10, pady=10)

# Chạy ứng dụng
root.mainloop()
