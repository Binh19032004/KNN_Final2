import os
import cv2
import numpy as np
import pickle
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from scipy.signal import find_peaks
from my_knn import MyKNNClassifier
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Tải mô hình KNN
with open("my_knn_model.pkl", "rb") as f:
    knn_model: MyKNNClassifier = pickle.load(f)

# ==== Tiền xử lý ảnh ====
def test_mnist_preprocessing():
    # Bước 1: Tải dữ liệu MNIST
    (X_train, y_train), (_, _) = mnist.load_data()
    
    # Bước 2: Lấy một mẫu ảnh từ tập huấn luyện
    idx = 0  # Chỉ số ảnh để kiểm tra (có thể thay đổi để kiểm tra mẫu khác)
    image = X_train[idx]
    label = y_train[idx]

    # Bước 3: Xử lý ảnh MNIST bằng preprocess_image
    processed_img, _ = preprocess_image(image=image)
    if processed_img is None:
        print("Không tìm thấy chữ số trong ảnh MNIST.")
        return
    
    # Bước 4: Hiển thị ảnh gốc và ảnh sau xử lý để kiểm tra
    plt.figure(figsize=(8, 4))
    
    # Hiển thị ảnh gốc
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title(f'Ảnh gốc MNIST (Label: {label})')
    plt.axis('off')
    
    # Hiển thị ảnh sau xử lý
    plt.subplot(1, 2, 2)
    plt.imshow(processed_img, cmap='gray')
    plt.title('Ảnh sau xử lý')
    plt.axis('off')
    
    plt.show()

def preprocess_image(path=None, image=None, filename=None):
    """
    Xử lý ảnh đầu vào để đưa về định dạng phù hợp với mô hình KNN (28x28, chữ số trắng trên nền đen).
    path: đường dẫn ảnh (cho ảnh thực tế từ người dùng).
    image: ảnh numpy array (cho ảnh MNIST, dạng thang xám 28x28).
    filename: tên file để lưu ảnh xử lý (cho ảnh thực tế).
    """
    # Kiểm tra loại đầu vào
    is_mnist = False
    if image is not None:
        # Nếu có image (ảnh MNIST), kiểm tra kích thước và số kênh
        if image.shape == (28, 28) or (len(image.shape) == 3 and image.shape[:2] == (28, 28) and image.shape[2] == 1):
            is_mnist = True
            gray = image if len(image.shape) == 2 else image.squeeze()
        else:
            raise ValueError("Ảnh đầu vào không đúng định dạng MNIST (phải là 28x28 thang xám).")
    else:
        # Nếu không có image, đọc ảnh từ file (ảnh thực tế)
        if path is None or filename is None:
            raise ValueError("Cần cung cấp path và filename cho ảnh thực tế.")
        img_color = cv2.imread(path)
        if img_color is None:
            raise ValueError(f"Không thể đọc ảnh từ đường dẫn: {path}")
        gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # Chuyển về thang [0, 255] nếu ảnh đã chuẩn hóa
    if gray.max() <= 1.0:
        gray = (gray * 255).astype(np.uint8)
    else:
        gray = gray.astype(np.uint8)

    if is_mnist:
        # Xử lý ảnh MNIST: chỉ kiểm tra và đảo màu nếu cần
        if np.mean(gray) > 127:
            gray = 255 - gray
        final_img = gray  # Đã ở kích thước 28x28, không cần xử lý thêm
    else:
        # Xử lý ảnh thực tế
        # Làm mượt và tăng tương phản
        gray_denoised = cv2.fastNlMeansDenoising(gray, None, h=10)
        gray_norm = cv2.normalize(gray_denoised, None, 0, 255, cv2.NORM_MINMAX)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_contrast = clahe.apply(gray_norm)

        # Nhị phân hóa
        binary = cv2.adaptiveThreshold(gray_contrast, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        combined = cv2.medianBlur(binary, 3)

        # Làm đậm nét nếu cần (dựa trên diện tích contour lớn nhất)
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(largest)
            if contour_area < 500:
                kernel = np.ones((3, 3), np.uint8)
                combined = cv2.dilate(combined, kernel, iterations=1)
                combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

            # Giữ lại contour lớn nhất để loại bỏ nhiễu
            mask = np.zeros_like(combined)
            cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)
            combined = mask
        else:
            return None, None

        # Cắt và resize ảnh về kích thước 28x28
        def crop_digit_with_padding(image, padding=4):
            coords = cv2.findNonZero(image)
            if coords is None:
                return None
            x, y, w, h = cv2.boundingRect(coords)
            x, y = max(x - padding, 0), max(y - padding, 0)
            w, h = w + 2 * padding, h + 2 * padding
            return image[y:y+h, x:x+w]

        digit = crop_digit_with_padding(combined)
        if digit is None:
            return None, None
        resized = cv2.resize(digit, (20, 20), interpolation=cv2.INTER_CUBIC)

        # Căn giữa chữ số trong khung 28x28 dựa theo trọng tâm
        def center_image(image):
            M = cv2.moments(image)
            cx, cy = (10, 10) if M["m00"] == 0 else (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
            canvas = np.zeros((28, 28), dtype=np.uint8)
            x_offset, y_offset = 14 - cx, 14 - cy
            for y in range(20):
                for x in range(20):
                    ny, nx = y + y_offset, x + x_offset
                    if 0 <= ny < 28 and 0 <= nx < 28:
                        canvas[ny, nx] = image[y, x]
            return canvas

        final_img = center_image(resized)

    # Lưu ảnh xử lý vào thư mục static/uploads (chỉ cho ảnh thực tế)
    if not is_mnist:
        processed_filename = f"processed_{filename}"
        processed_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
        cv2.imwrite(processed_path, final_img)
    else:
        processed_path = None

    return final_img, processed_path

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)

            # Tiền xử lý ảnh thực tế
            processed_img, processed_path = preprocess_image(path=image_path, filename=filename)
            if processed_img is None or processed_path is None:
                return render_template('index.html', prediction="Không tìm thấy chữ số trong ảnh", confidence=None)

            img_flat = processed_img.reshape(1, -1) / 255.0

            # Dự đoán
            pred = knn_model.predict(img_flat)[0]
            confidence = None  # Vì không có xác suất

            return render_template('index.html',
                                   prediction=pred,
                                   confidence=confidence,
                                   image_url=image_path,
                                   processed_url=processed_path)

    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    print("🚀 Flask đang chạy tại http://127.0.0.1:5000")
    app.run(debug=True, host='127.0.0.1', port=5000)

# Chỉ chạy kiểm tra nếu không chạy web
if __name__ == '__main__' and os.getenv('TEST_MNIST', '0') == '1':
    test_mnist_preprocessing()
else:
    print("🚀 Flask đang chạy tại http://127.0.0.1:5000")
    app.run(debug=True, host='127.0.0.1', port=5000)


