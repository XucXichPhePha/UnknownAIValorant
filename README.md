# Nhận diện và vẽ Box trên màn hình

Chương trình dùng model Ultralytics `best.pt` để nhận diện nội dung trên màn
hình chính và vẽ khung cùng tên lớp/độ tin cậy bằng một lớp phủ trong suốt.
`best.onnx` là một định dạng model khác; chương trình hiện dùng `best.pt`.

## Cài đặt trên máy NVIDIA/AMD

1. Cài Python 3.10 trở lên và driver NVIDIA mới; kết nối sạc laptop để GPU
   hoạt động ổn định.
2. Chạy `setup.bat`. Script tạo môi trường `.venv` riêng, cài PyTorch CUDA
   13.0 cùng các thư viện cần thiết, rồi kiểm tra GPU.
3. Chạy `run.bat`.

Ứng dụng dùng RTX NVIDIA (CUDA), FP16, giới hạn khung hình đầu vào tối đa
640px và kích thước suy luận 640px; giới hạn vòng lặp hiện tại là 144 FPS.
Tốc độ thực tế phụ thuộc vào model và tải GPU, được hiển thị trong cửa sổ CMD
mỗi 5 giây. Thu nhỏ khung hình trước khi tiền xử lý tránh xử lý ảnh desktop
Full HD dư thừa, nhưng không thay đổi kích thước suy luận của model. GPU AMD
tích hợp vẫn có thể điều khiển màn hình; việc suy luận model được gửi riêng
tới NVIDIA. Chương trình báo lỗi thay vì âm thầm chuyển sang CPU nếu CUDA
không khả dụng.

Để tăng tốc suy luận thêm, chạy `build_engine.bat`. Lần đầu có thể tải khoảng
2 GB TensorRT, sau đó build `best.engine` FP32 phù hợp với GPU hiện tại.
Ứng dụng tự ưu tiên engine này nếu tồn tại; nếu chưa build, sẽ tiếp tục dùng
`best.pt`. Engine phụ thuộc GPU/driver và không được đưa lên GitHub.
Trên RTX 4050 Laptop của máy này, benchmark warm-up với ảnh màn hình đã thu
nhỏ đo được khoảng 130 FPS TensorRT; tốc độ thực tế thay đổi theo tải GPU.

## Sử dụng

- Chương trình tự nhận diện màn hình chính; khung trắng hiện trên vùng
  nhận diện được.
- Trong cửa sổ CMD, gõ `text off` để tắt chữ tên lớp/độ tin cậy mà vẫn giữ
  khung; gõ `text on` để bật lại, hoặc `text toggle` để đảo trạng thái.
  Có thể dùng `chu tat`, `chu bat`, `chu doi` hoặc gõ `help` để xem lệnh.
- Gõ `show head` để chỉ hiện khung Head, `show body` để chỉ hiện Body,
  hoặc `show both` để hiện lại cả hai. Có thể dùng `chi head`, `chi body`,
  hoặc `hien tat ca` tương ứng.
- Gõ `quit` trong CMD để thoát chương trình.
- Nhấp biểu tượng chấm xanh ở khay hệ thống để tạm ẩn/hiện lớp phủ.
- Nhấp phải biểu tượng đó và chọn **Thoát** để đóng chương trình.
- Để đóng lớp console sau khi chạy, không dùng nút đóng cửa sổ đó; hãy thoát
  từ menu khay hệ thống trước.

Nếu biểu tượng khay hệ thống không hiện, dùng `Ctrl+C` trong cửa sổ console để
dừng chương trình.

Để ưu tiên hiệu năng khi chơi game, cắm sạc và chọn chế độ Performance trong
OMEN Gaming Hub nếu máy có tùy chọn đó (Máy Dell HP). Windows có thể tiếp tục dùng đồ họa AMD
để hiển thị desktop; không cần ép ứng dụng giao diện sang NVIDIA vì PyTorch đã
chỉ định GPU CUDA trực tiếp.

## Các thông số cài đặt màn hình cho Valorant để chạy mượt mà nhất

- Native Resolution: 1920x1080p
- Outline Color: Purple
- Material Quality: Low
- Texture Quality: Medium
- Detail Quality: Low
- Anti-Aliasing: MSAA 2x
- Anisotropic Filtering: 2x
- Improve Clarity: On
- Experimental Sharpening: On
- Bloom: Off
- Distortion: Off
- Cast Shadows: Off

<div align="center">

<p>Developed with ❤️ by <a href="https://github.com/XucXichPhePha"><b>XucXichPhePha</b></a></p>

[![GitHub profile](https://img.shields.io/badge/GitHub-XucXichPhePha-181717?style=flat-for-the-badge&logo=github)](https://github.com/XucXichPhePha)

</div>
