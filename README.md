# Phân đoạn và phân loại tổn thương da

Đồ án CS406 xây dựng một quy trình nghiên cứu có thể tái lập để phân đoạn vùng tổn thương trên ảnh soi da, phân loại ảnh theo nhãn chẩn đoán và khảo sát ảnh hưởng của vùng quan tâm (ROI) đến kết quả phân loại.

> Đây là phần mềm phục vụ nghiên cứu và học tập, không phải thiết bị y tế và không đưa ra chẩn đoán lâm sàng.

## Mục tiêu

- Phân đoạn nhị phân vùng tổn thương từ ảnh soi da.
- So sánh U-Net và Attention U-Net trên cùng cách chia dữ liệu và điều kiện huấn luyện.
- Phân loại ảnh thành 7 lớp: MEL, NV, BCC, AKIEC, BKL, DF, VASC.
- So sánh ResNet50 và EfficientNet-B0.
- Khảo sát phân loại trên ảnh gốc so với ROI lấy từ mask dự đoán; nếu đủ dữ liệu giao hợp lệ, dùng ROI từ mask thật làm tham chiếu.
- Phân tích lỗi, thời gian suy luận và giới hạn của phương pháp.

## Dữ liệu

Nguồn dữ liệu chính dự kiến là ISIC 2018:

- Task 1: ảnh và mask phân đoạn tổn thương.
- Task 3: ảnh và nhãn phân loại 7 lớp.

Nhóm cần ghi nguồn dữ liệu và tuân thủ điều kiện sử dụng của ISIC. Không đưa ảnh dữ liệu, thông tin cá nhân hoặc checkpoint lớn vào GitHub. Hãy tải dữ liệu theo hướng dẫn nguồn chính thức và lưu bên ngoài repository.

Trước khi huấn luyện:

1. Tạo manifest có image ID, lesion ID (nếu có), đường dẫn ảnh/mask, nhãn và trạng thái kiểm tra.
2. Kiểm tra ảnh lỗi, ID trùng, nhãn thiếu, mask và kích thước ảnh.
3. Thống kê số lượng theo lớp và phần giao giữa Task 1 với Task 3; chỉ báo cáo số lượng đo được.
4. Chia train/validation/test theo lesion ID (gợi ý 70/15/15), phân tầng theo nhãn nếu khả thi. Ghi seed và danh sách ID vào CSV.
5. Dùng validation để chọn mô hình/tham số; chỉ đánh giá test sau khi đã chốt quy trình.
6. Khi đánh giá ROI, bảo đảm ảnh/lesion test classification không xuất hiện trong dữ liệu train segmentation. Nếu tập giao hợp lệ quá nhỏ, báo cáo đây là thí nghiệm thăm dò.

Lưu manifest và split có thể chia sẻ trong `data/splits/`; không lưu ảnh gốc trong repo.

## Chỉ số và báo cáo

- Phân đoạn: Dice, IoU và tỷ lệ ảnh phân đoạn thất bại.
- Phân loại: balanced accuracy, macro F1, AUC và recall từng lớp; phân tích riêng MEL và các lớp ít mẫu.
- Báo cáo confusion matrix, số mẫu mỗi lớp, ví dụ dự đoán đúng/sai/khó và thời gian suy luận.
- So sánh ROI dự đoán với ảnh gốc trên cùng các trường hợp; không khẳng định hiệu quả ngoài phạm vi tập đánh giá.
- Ghi cấu hình, seed, phiên bản mã và điều kiện chạy cho mỗi thí nghiệm.

## Cấu trúc thư mục dự kiến

Các thư mục dưới đây là cấu trúc mục tiêu; tạo dần theo các issue được giao.

~~~text
.
├── src/
│   ├── data/          # Tiền xử lý, augmentation, Dataset/DataLoader
│   ├── models/        # U-Net, Attention U-Net, ResNet50, EfficientNet-B0
│   ├── training/      # Vòng lặp huấn luyện và lưu cấu hình
│   ├── evaluation/    # Metrics, bảng và biểu đồ
│   └── inference/     # Mask, ROI và phân loại
├── configs/           # Cấu hình dữ liệu và thí nghiệm
├── data/
│   └── splits/        # Manifest và danh sách ID các split; không chứa ảnh
├── notebooks/         # EDA và thử nghiệm ban đầu
├── app/               # Demo ảnh đơn
├── results/           # Bảng/hình chọn lọc, không chứa checkpoint lớn
├── docs/              # Báo cáo, kế hoạch và quyết định
├── README.md
└── .gitignore
~~~

## Bắt đầu và clone repository

Cần cài Git. Clone repository bằng HTTPS:

~~~bash
git clone https://github.com/vphuocthinh2006/Skin-lesion-analysis-and-skin-cancer-analysis.git
cd Skin-lesion-analysis-and-skin-cancer-analysis
git switch main
git pull origin main
~~~

Sau khi clone, hãy đọc issue được giao và trao đổi với nhóm về dữ liệu/môi trường trước khi bắt đầu. Các phiên bản Python, thư viện và lệnh chạy sẽ được bổ sung khi nhóm chốt cấu hình ở issue #1.

## Quy trình làm việc nhóm

### 1. Chọn và nhận issue

Mỗi đầu việc cần có một issue, một người phụ trách, đầu ra và tiêu chí hoàn thành. Bảy issue khởi động:

- #1: khởi tạo cấu trúc repository và quy trình.
- #2: kiểm tra dữ liệu ISIC và chốt split dùng chung.
- #3: U-Net và Attention U-Net.
- #4: ResNet50 và EfficientNet-B0.
- #5: metrics, biểu đồ và phân tích lỗi.
- #6: pipeline segmentation đến ROI đến classification.
- #7: demo, báo cáo và hướng dẫn chạy.

Issue #2 cần chốt split trước khi bắt đầu các thí nghiệm chính ở #3 và #4. Hai nhánh mô hình có thể phát triển song song sau khi giao diện dữ liệu/split thống nhất. Pipeline ở #6 phụ thuộc vào dữ liệu và hai nhánh mô hình.

### 2. Tạo branch riêng cho issue

Các branch khởi động đã được tạo sẵn trên GitHub. Sau khi cập nhật main, checkout branch của issue được giao. Ví dụ issue #2:

~~~bash
git switch main
git pull origin main
git fetch origin
git switch --track origin/feature/data-split
~~~

Nếu branch mới chưa tồn tại trên GitHub, tạo từ main bằng git switch -c <ten-branch>, rồi push lần đầu với git push -u origin <ten-branch>.

Tên branch:

~~~text
chore/repo-bootstrap
feature/data-split
feature/segmentation
feature/classification
feature/evaluation
feature/roi-pipeline
feature/demo-report
~~~

Mỗi người chỉ làm trên branch được tạo cho đầu việc của mình. Không commit trực tiếp lên main. Nếu branch đó đã được checkout ở máy, chuyển lại bằng git switch <ten-branch>.

### 3. Commit và push

Chỉ đưa các thay đổi liên quan đến issue vào commit. Trước khi commit, kiểm tra Git đang dùng đúng danh tính cá nhân của bạn:

~~~bash
git config user.name
git config user.email
~~~

Nếu cần đặt danh tính riêng cho repository này:

~~~bash
git config user.name "Tên của bạn"
git config user.email "email-gắn-với-GitHub-của-bạn"
~~~

Sau khi chỉnh sửa:

~~~bash
git status
git add <file-can-commit>
git commit -m "data: create shared dataset split"
git push
~~~

Không thêm dòng Co-authored-by cho AI/công cụ hỗ trợ. Commit cần phản ánh người thực hiện và tài khoản GitHub của chính thành viên đó.

### 4. Mở Pull Request và review

- Mở PR từ branch task vào `main`.
- Trong mô tả PR, nêu việc đã làm, cách kiểm tra và hạn chế còn lại.
- Liên kết issue bằng cú pháp `Closes #2` khi PR giải quyết xong issue đó.
- Nhờ ít nhất một thành viên khác review; sửa góp ý rồi mới merge.
- Sau khi merge, cập nhật main và xóa branch đã hoàn tất nếu không cần giữ.

~~~bash
git switch main
git pull origin main
~~~

### 5. Theo dõi tiến độ

Nếu nhóm dùng GitHub Projects, quản lý issue theo các cột: Todo → Doing → Review → Done. Họp ngắn định kỳ để xử lý phụ thuộc, thống nhất split/metrics và ghép pipeline sớm.

## Phân công theo mảng

Đề cương chia công việc thành bốn mảng để nhóm tự gán cho Quốc Thịnh, Phước Thịnh, Trường An và Thiên:

- **Dữ liệu:** manifest, EDA, preprocessing, split và DataLoader.
- **Phân đoạn:** U-Net, Attention U-Net, mask và metric phân đoạn.
- **Phân loại:** ResNet50, EfficientNet-B0, xác suất và metric phân loại.
- **Đánh giá/tích hợp:** tổng hợp kết quả, phân tích lỗi, ROI pipeline và demo.

Một thành viên có thể phối hợp nhiều mảng; hãy ghi người phụ trách chính trong issue để tránh trùng việc. Việc chia nhánh theo task giúp PR nhỏ, dễ review và dễ theo dõi hơn chia một branch dài hạn cho mỗi người.

## Tài liệu tham khảo

- Đề cương đồ án CS406 do nhóm cung cấp.
- ISIC 2018: https://challenge.isic-archive.com/landing/2018/
