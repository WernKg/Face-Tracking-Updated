from insightface.app import FaceAnalysis
import cv2
import os, sys, csv, time
import numpy as np
from ultralytics import YOLO
from datetime import datetime
from tkinter import Tk, filedialog, simpledialog
#from openpyxl import Workbook

# (Hàm lấy dữ liệu)
def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'): 
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

#Khởi tạo model
model_path = resource_path("yolov12n-face.pt")
model = YOLO(model_path)

#       Tạo Log Excel
csv_file = "log_nhandien.csv"
if not os.path.exists(csv_file):
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Thoi gian", "Nguon", "Ten", "Lop", "Đo tin cay"])

# tải dữ liệu nếu có ( AI Debug)
if os.path.exists("known_face_embeddings.npy"):
    known_face_embeddings = list(
        np.load("known_face_embeddings.npy", allow_pickle=True)
    )
else:
    known_face_embeddings = []

if os.path.exists("known_face_names.npy"):
    known_face_names = list(np.load("known_face_names.npy", allow_pickle=True))
else:
    known_face_names = []

if os.path.exists("known_face_class.npy"):
    known_face_class = list(np.load("known_face_class.npy", allow_pickle=True))
else:
    known_face_class = []


# Thêm người
def add_new_person(app):
    Tk().withdraw()  # cửa sổ mới
    new_name = simpledialog.askstring("Thông tin", "Nhập tên: ") # nhập tên
    new_class = simpledialog.askstring("Thông tin", "Nhập lớp học:") #nhập lớp
    if not new_name or not new_class:
        print("Hủy: chưa nhập tên hoặc lớp.")
        return

    # Ép kiểu chuỗi cho chắc (tránh bytes khi load/save)
    new_name = str(new_name).strip()
    new_class = str(new_class).strip()

    image_paths = filedialog.askopenfilenames(
        title="Chọn tệp: ",
        filetypes=[("Tệp hình ảnh", "*.jpg *.jpeg *.png")], 
    ) # Chọn hình có chứa mặt
    if not image_paths:
        print("Không có ảnh nào được chọn.")
        return

    added_count = 0
    for image_path in image_paths:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Không thể load {image_path}")
            continue
        
        if img.shape[0] > 1000 or img.shape[1] > 1000:
            img = cv2.resize(img, (640, 640))

        faces = app.get(img)
        if len(faces) > 0:
            for face in faces:
                embedding = np.array(face.embedding, dtype=np.float32).flatten()
                known_face_embeddings.append(embedding) # thêm khuôn mặt nhúng mới sau khi nhập hình ảnh
                known_face_names.append(str(new_name)) # thêm tên mới sau khi nhập tên
                known_face_class.append(str(new_class)) # thêm tên mới sau khi nhập tên
                added_count += 1 
                print(f"[ADDED] {os.path.basename(image_path)} -> {new_name} - {new_class}")

        else:
            print(f"Không tìm thấy mặt trong hình ảnh {os.path.basename(image_path)}")
    try:
        np.save("known_face_embeddings.npy", np.array(known_face_embeddings)) # csdl hình ảnh (có thể nâng cấp để bảo mật)
        np.save("known_face_names.npy", np.array(known_face_names)) # như trên nhưng là tên
        np.save("known_face_class.npy", np.array(known_face_class)) # như trên nhưng là lớp
        print(f"Tất cả hình ảnh của {new_name} đã được sao lưu.")
    except Exception as e:
        print(f"Lưu file .npy thất bại: {e}")
        return

def init_insightface(): #Hàm khởi tạo InsightFace, bộ não chính nơi AI được dùng để nhận diện khuôn mặt
    try:
        model_directory = resource_path("buffalo_l")
        app = FaceAnalysis(
                name="buffalo_l", #Mô hình được sử dụng cho dự án, đây là mô hình mạnh nhất mà tôi biết
                root=model_directory,
                providers=["DmlExecutionProvider", "CPUExecutionProvider"], #Phần cứng mà InsightFace sẽ sử dụng: GPU hoặc CPU (khuyến nghị dùng GPU để có tốc độ khung hình cao hơn)
            )
        app.prepare(ctx_id=0, det_size=(320, 320)) #  ctx_id = 0 nghĩa là dùng GPU (nếu có), -1 là dùng CPU, 1 là GPU khác nếu bạn có nhiều GPU
        print("InsightFace init success!!") #  In ra khi khởi tạo mô hình thành công
        return app
    except Exception as e:
        print(f"Error while trying init insightface {e}") 
        return None


app = init_insightface()
if app is None:
    print("InsightFace init failed. Exiting...") # Nếu khởi tạo không thành công, chương trình sẽ thoát
    sys.exit(1)
# Dùng để tính FPS (số khung hình mỗi giây)
time_prev = 0

# khởi động máy ảnh
webcam = cv2.VideoCapture(0)


image_path = None # lưu đường dẫn hình ảnh
use_webcam = True # Mặc định máy ảnh
while True:
    if use_webcam:
        ret, frame = webcam.read()
        if not ret:
            break
        source = "Webcam"
        current_time = time.time()
        fps = 1 / (current_time - time_prev) if time_prev > 0 else 0
        time_prev = current_time
        # Hiển thị FPS lên khung hình
        cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
       )
    else:
        if image_path:
            frame = cv2.imread(image_path)
        else:
            continue

#               Nhận diện khuôn mặt            
    results = model(frame)
    best_conf = 0
    best_info = None
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])
            #Vẽ khung
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255,255,255), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
#               Log Vào Excel
#            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = app.get(frame)
            for face in faces:
                embedding = np.array(face.embedding, dtype=np.float32)
                embedding = embedding / np.linalg.norm(embedding)

                best_score = 0
                best_name = "Khong xac dinh"
                best_class = "N/A"

                for i, known_embedding in enumerate(known_face_embeddings):
                    known_embedding = known_embedding / np.linalg.norm(known_embedding)
                    sim = np.dot(embedding, known_embedding)
                    if sim > best_score:
                        best_score = sim
                        best_name = known_face_names[i]
                        best_class = known_face_class[i]

                if best_score > 0.5:
                    recognized_name = best_name
                    recognized_class = best_class
                else:
                    recognized_name = "Khong xac dinh"
                    recognized_class = "N/A"
                with open(csv_file, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        source,
                        recognized_name,
                        recognized_class,
                        round(best_score, 3)
                    ])
#               HIỂN THỊ        
    cv2.imshow("YOLO Detect --- (Currently Under Development)", frame)

#               XỬ LÝ PHÍM                  
    key = cv2.waitKey(1) & 0xFF
    #chọn ảnh
    if key == ord("p"): 
        Tk().withdraw()
        image_path = filedialog.askopenfilename(
            title="Chọn ảnh ",
            filetypes=[("Tệp hình ảnh hoặc video", "*.jpg *.jpeg *.png *.gif *.mp4")]
        )
        if image_path:
            use_webcam = False
            webcam.release()
    #chuyển về webcam
    elif key == ord("c"): 
        if not use_webcam:  
            webcam = cv2.VideoCapture(0)
            use_webcam = True
    # thoát
    elif key == ord("e"):
        break
    #Thêm tên và hình ảnh
    elif key == ord("a"):
        add_new_person(app)
# tắt
if webcam.isOpened():
    webcam.release()
cv2.destroyAllWindows()

print("Đã load các khuôn mặt:", known_face_names)
print("→ Số khuôn mặt phát hiện:", len(faces))





