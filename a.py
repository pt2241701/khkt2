try:
    from PyQt5.QtWidgets import QApplication, QMainWindow
    from PyQt5.QtGui import QImage, QPixmap
    from PyQt5.QtCore import QThread, pyqtSignal, Qt
    from PyQt5.uic import loadUi
    import cv2
    import sys
    from ultralytics import YOLO
    import speech_recognition as sr
    import os
    from moviepy.editor import VideoFileClip, ImageSequenceClip
except:
    import os
    os.system("pip install pyqt5 pyqt5-tools opencv-python ultralytics SpeechRecognition pyaudio")
    from PyQt5.QtWidgets import QApplication, QMainWindow
    from PyQt5.QtGui import QImage, QPixmap
    from PyQt5.QtCore import QThread, pyqtSignal, Qt
    from PyQt5.uic import loadUi
    import cv2
    import sys
    from ultralytics import YOLO
    import speech_recognition as sr

class SpeechToVideoThread(QThread):
    video = pyqtSignal(QImage)
    audioTextChanged = pyqtSignal(str)
    def __init__(self, img_dir, video_output_path):
        super(SpeechToVideoThread, self).__init__()
        self.img_dir = img_dir
        self.video_output_path = video_output_path
        self.audio_text = ""
        self.is_recording = False
    def run(self):
        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 300
        while self.is_recording:
            with sr.Microphone() as source:
                print("Recording...")
                audio = recognizer.listen(source)
            try:
                self.audio_text = recognizer.recognize_google(audio)
                print("Text from audio: ", self.audio_text)
                self.create_video_from_text()
                self.audioTextChanged.emit(self.audio_text)
            except sr.UnknownValueError:
                pass  # Ignore unrecognized audio
            except sr.RequestError as e:
                print(f"Error with the request to Google Speech Recognition service; {e}")
    def start_recording(self):
        self.is_recording = True
        self.start()

    def stop_recording(self):
        self.is_recording = False
        self.wait()

    def create_video_from_text(self):
        img_list = []
        for char in self.audio_text.lower():
            if char != ' ':
                img_path = os.path.join(self.img_dir, f"{char}.jpg").replace('\\', '/')
                if os.path.exists(img_path):
                    img_list.append(img_path)
            elif char == ' ':
                img_path = os.path.join(self.img_dir, 'space.jpg').replace('\\', '/')
                if os.path.exists(img_path):
                    img_list.append(img_path)
            else:
                continue
        print("Image List:", img_list)
        if img_list:
            self.show_video(img_list)
            self.audioTextChanged.emit("Video created!")
            print("Done")

    def show_video(self, img_list):
        frame_list = []
        for img_path in img_list:
            frame = cv2.imread(img_path)
            if frame is not None:
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_list.append(rgb_image)

        if frame_list:
            fps = 0.25
            clip = ImageSequenceClip(frame_list, fps=fps)
            clip.write_videofile(self.video_output_path, codec='libx264', fps=fps)
class Video(QThread):
    vid = pyqtSignal(QImage)

    def run(self):
        self.hilo_corriendo = True
        video_path = r"D:\a\output_video.mp4"
        cap = cv2.VideoCapture(video_path)
        
        fps = cap.get(cv2.CAP_PROP_FPS)  # Get the frame rate
        delay = int(1000 / fps)  # Calculate delay between frames
        
        while self.hilo_corriendo:
            ret, frame = cap.read()
            if ret:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                flipped_image = cv2.flip(image, 1)
                convert_to_QT = QImage(flipped_image.data, flipped_image.shape[1], flipped_image.shape[0], QImage.Format_RGB888)
                pic = convert_to_QT.scaled(890, 440, Qt.KeepAspectRatio)
                self.vid.emit(pic)
                
                self.msleep(delay)  # Introduce delay to match the frame rate
        cap.release()

    def stop(self):
        self.hilo_corriendo = False
        self.quit()


class Ham_Camera(QThread):
    luongPixMap = pyqtSignal(QImage)
    luongString = pyqtSignal(str)
    def __init__(self):
        super(Ham_Camera, self).__init__()
        
        self.checkTrung = ""
        # Khởi tạo biến checkTrung để lưu tên vật thể đã được hiển thị lên label text.
        # Biến này sẽ được sử dụng để kiểm tra xem vật thể hiện tại có trùng với vật thể đã được hiển thị
        self.trangThai = True 
    def run(self):
        string = ""
        main_model = YOLO("yolov8n.pt") # tải model đã được huấn luyện cho dự án
        cap = cv2.VideoCapture(0) #khởi tạo webcam
        while self.trangThai:# chạy liên tục quá trình nhận diện
            ret, frame = cap.read() #đọc ảnh từ webcam
            if ret: #nếu như camera được khởi tạo thành công thì sẽ chạy phần xử lý, nếu không thì sẽ thoát chương trình
                ketQua = main_model.predict(source = frame)# kết quả khi chạy chương trình, kết quả này đươc cập nhật theo tgian thực
                for r in ketQua:
                    for vatThe in r.boxes:
                        class_id = int(vatThe.cls.item())#trích ra giá trị của vật thể trong ketQua
                        score = vatThe.conf.item()#trích ra độ tin cậy của vật thể trong ketQua
                        item_name = main_model.names[class_id]# xác định xem vật thể đó tên gì thông qua class_id
                        if score >= 0.85 and item_name != self.checkTrung: #nếu như độ tin cậy cao và tên vật thể khác checkTrung 
                            if item_name == "space":
                                string = string + " "
                                self.luongString.emit(string)
                                self.checkTrung = item_name
                            else:
                                string = string + item_name #hiển thị tê của vật thể lên label text
                                self.luongString.emit(string)
                                self.checkTrung = item_name
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                p = convert_to_Qt_format.scaled(891, 461, Qt.KeepAspectRatio)
                self.luongPixMap.emit(p)
            else:
                break
        cap.release()
    def stop(self): 
        self.trangThai = False
"""
Ham_Camera được sử dụng để khởi tạo webcam và chạy mô hình dự đoán của dự án, hình ảnh được ghi nhận từ webcam sẽ được
chuyển thành hình ảnh sau đó cập nhật lên label_cam, tốc dộ cập nhật gần như bằng với thời gian thực
"""
class Ham_Chinh(QMainWindow):
    
    # Lớp Ham_Chinh là lớp chính của chương trình, chịu trách nhiệm khởi tạo các thành phần giao diện và kết nối các tín hiệu giữa các lớp.
    def __init__(self):
        # Gọi hàm khởi tạo của lớp QMainWindow
        super(Ham_Chinh, self).__init__()
        # Tải giao diện từ file ui.ui
        loadUi('main.ui', self)
        
        # Khởi tạo luồng camera
        self.Work = Video()
        self.thread_camera = Ham_Camera()
        # Khởi tạo luồng video
        self.img_dir = r'D:\a\img'
        self.video_output_path = r'output_video.mp4'
        self.thread_vid = SpeechToVideoThread(self.img_dir, self.video_output_path)
        # Kết nối tín hiệu luongPixMap của luồng camera với hàm setCamera
        self.thread_camera.luongPixMap.connect(self.setCamera)
        # Kết nối tín hiệu startcam của nút startcam với hàm khoiDongCamera
        self.startcam.clicked.connect(self.khoiDongCamera)
        # Kết nối tín hiệu pausecam của nút pausecam với hàm tamDungCamera
        self.pausecam.clicked.connect(self.tamDungCamera)
        # Kết nối tín hiệu clear của nút clear với hàm xoaToanBo
        self.clear.clicked.connect(self.xoaToanBo)
        # Kết nối tín hiệu delete_2 của nút delete_2 với hàm xoaChu
        self.delete_2.clicked.connect(self.xoaChu)
        # Kết nối tín hiệu luongString của luồng camera với hàm setText của label text
        self.thread_camera.luongString.connect(self.text.setText)
        #voice to text/video
        self.record_button.clicked.connect(self.start_recording)
        self.stop_record_button.clicked.connect(self.stop_recording)
        self.stop_record_button.setEnabled(False)
        self.play_video.clicked.connect(self.start_video)
        self.stop_video.clicked.connect(self.stop_vide)
        self.thread_vid.audioTextChanged.connect(self.text_2.setText)

        
    def start_video(self):
        self.Work.start()
        self.Work.vid.connect(self.Imageupd_slot)

    def Imageupd_slot(self, Image):
        self.img_label.setPixmap(QPixmap.fromImage(Image))

    def stop_vide(self):
        self.Work.stop()

    
    def setCamera(self, image):
        # Cập nhật hình ảnh lên label cam
        self.label.setPixmap(QPixmap.fromImage(image))
    def khoiDongCamera(self):
        # Khởi động luồng camera để bắt đầu nhận diện vật thể
        self.thread_camera.start()
    def tamDungCamera(self):
        # Dừng luồng camera để tạm dừng nhận diện vật thể
        self.thread_camera.stop()
        # Chờ luồng camera hoàn toàn dừng trước khi tiếp tục
        self.thread_camera.wait()
    def xoaToanBo(self):
        # Xóa toàn bộ nội dung trong label text
        self.text.clear()
    def xoaChu(self):
        # Xóa ký tự cuối cùng trong textt
        textt = self.text.text()
        textt = textt.rsplit(' ', 1)[0]
        # Cập nhật textt lên label text
        self.text.setText(textt)
    def start_recording(self):
        self.record_button.setEnabled(False)
        self.stop_record_button.setEnabled(True)
        self.thread_vid.start_recording()
    def stop_recording(self):
        self.record_button.setEnabled(True)
        self.stop_record_button.setEnabled(False)
        self.thread_vid.stop_recording()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Ham_Chinh()
    window.setWindowTitle('tp')
    window.show()
    sys.exit(app.exec_())