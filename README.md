# Face-Tracking-Updated
Phần mềm Face Tracking được nâng cấp từ bản Beta

Bản Beta: https://github.com/172009/Face-Tracking/tree/release-beta

Developer/Author: WernKg

# Các Bước Thực Hiện Để Chạy Chương Trình

* Yêu cầu bắt buộc:

Python 3.11.7+ [https://www.python.org/downloads/]

* Bước 1:
  Tải Microsoft Visual Studio Build Tools [https://visualstudio.microsoft.com/visual-cpp-build-tools/]

  Khi cài đặt chọn: <Desktop development with C++>
  
  Tùy chọn - chọn thêm “Windows 10 SDK” hoặc “Windows 11 SDK” nếu có

* Bước 2:
Tải tất cả thư viện được gói trong "requirements.txt"

      pip install -r requirements.txt

Lưu Ý: Khi cài nếu phiên bản thư viện bị lỗi thì hãy thử giảm hoặc tăng phiên bản

* Bước 3:
Chạy file script.exe để khởi động chương trình

#Notes: Khi phát sinh lỗi xin hãy báo cho Dev hoặc Author để được xử lí.

# Chi Tiết
Tổng chương trình gồm có các File:

* Script.exe (Main)
* known_face_embeddings.npy (face-data)
* known_face_names.npy (name-data)
* known_face_class.npy (class-data)
* log_nhandien.csv (log excel)
* buffalo_l (File Model)

Các chương trình hổ trợ phụ (tùy thích)

* Chương trình kiểm tra dữ liệu trong các File Data .npy [check-data.py]
* Chương trình khởi tạo lại các File Data .npy [create_file_data.py]

# Phím Tắt

* Phím tắt "a" - Thêm trực tiếp dữ liệu (data) vào file npy
  
  Exam:
  
      Nhập họ và tên: Nguyễn Văn A
  
      Nhập lớp: 12A1
  
      Chọn ảnh (ảnh mặt mộc/thẳng mặt/không chá ánh sáng)

* Phím tắt "p" - Chọn ảnh để bắt đầu nhận diện

* Phím tắt "c" - Chuyển đổi Webcam

* Phím tắt "e" - Thoát chương trình


