# Đồ án: Nhận diện và Phân vùng Biển báo Giao thông

Đây là dự án xử lý ảnh số nhằm phát hiện, phân vùng và nhận diện biển báo giao thông từ ảnh tĩnh. Dự án sử dụng các thư viện Python phổ biến như OpenCV và NumPy.

## Cấu trúc Thư mục

-   `/data/raw/`: Chứa các ảnh đầu vào.
-   `/results/`: Chứa kết quả của các bước xử lý ảnh.
-   `/src/`: Chứa toàn bộ mã nguồn xử lý.
    -   `utils.py`: Các hàm tiện ích (tải, lưu, hiển thị ảnh).
    -   `enhancement.py`: Các hàm tăng cường chất lượng ảnh.
    -   `restoration.py`: Các hàm khôi phục, giảm nhiễu ảnh.
    -   `segmentation.py`: Các hàm phân vùng ảnh để tách đối tượng.
    -   `morphology.py`: Các hàm xử lý hình thái để tinh chỉnh kết quả.
    -   `recognition.py`: Các hàm nhận dạng đối tượng (ví dụ: so khớp mẫu).
-   `main.py`: Tệp thực thi chính để chạy toàn bộ quy trình.
-   `requirements.txt`: Các thư viện cần thiết.

## Hướng dẫn Cài đặt và Chạy

1.  **Cài đặt các thư viện cần thiết:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Chuẩn bị dữ liệu:**
    -   Đặt các ảnh biển báo bạn muốn xử lý vào thư mục `data/raw/`.

3.  **Chạy chương trình:**
    -   Mở file `main.py` và thay đổi giá trị của biến `IMAGE_FOLDER_PATH` thành đường dẫn tới thư mục ảnh của bạn.
    -   Chạy tệp `main.py` từ terminal:
    ```bash
    python main.py
    ```

4.  **Xem kết quả:**
    -   Kết quả xử lý cho mỗi ảnh sẽ được lưu trong thư mục `results/` theo từng bước.
