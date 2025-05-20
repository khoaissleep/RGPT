# SpatialRGPT

SpatialRGPT là một hệ thống tạo đồ thị cảnh 3D (3D Scene Graph) từ hình ảnh. Dự án này sử dụng các mô hình AI để trích xuất thông tin 3D từ ảnh 2D và tạo ra các mối quan hệ không gian giữa các đối tượng.

## Tính năng chính

- **Phát hiện và phân đoạn đối tượng**: Sử dụng GroundingDINO và SAM để phát hiện và phân đoạn đối tượng trong ảnh
- **Ước tính độ sâu**: Sử dụng UniK3D để tạo bản đồ độ sâu từ ảnh đơn
- **Tạo điểm đám mây 3D**: Chuyển đổi bản đồ độ sâu thành điểm đám mây 3D
- **Phân tích quan hệ không gian**: Xác định các mối quan hệ không gian giữa các đối tượng (trên/dưới, bên trái/phải, trước/sau, khoảng cách)
- **Tạo câu hỏi-trả lời về không gian**: Sử dụng LLM để tạo các câu hỏi và trả lời về quan hệ không gian

## Cài đặt

Xem file [setup_environment.txt](setup_environment.txt) để biết chi tiết về cách thiết lập môi trường.

## Sử dụng

```bash
# Chạy với ảnh mẫu
python dataset_pipeline/generate_3dsg.py
```

## Cấu trúc dự án

```
SpatialRGPT/
├── configs/              # File cấu hình
├── dataset_pipeline/     # Mã nguồn chính của pipeline
│   ├── generate_3dsg.py  # Tệp chính để tạo đồ thị cảnh 3D
│   └── osdsynth/         # Thư viện chức năng
├── demo_images/          # Hình ảnh mẫu
└── UniK3D/               # Mô hình ước tính độ sâu
```

## Khắc phục sự cố

- Nếu gặp lỗi với PerspectiveFields, hãy sửa đổi cấu hình để sử dụng "geo_calib" thay vì "perspective_fields" trong configs/v2_hf_llm.py
- Nếu gặp lỗi "QuadContourSet", hãy sửa đổi file visualizer.py trong thư mục PerspectiveFields 