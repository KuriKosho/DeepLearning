# Khai báo thư viện transform để tiền xử lý ảnh
from torchvision import transforms
# Khai báo thư viện Image để đọc ảnh
from PIL import Image
# Khai báo thư viện torch để chuyển ảnh thành tensor
import torch
# Khai báo thư viện torchvision.models để load mô hình ResNet
from torchvision.models import resnet101, ResNet101_Weights

# Khai báo hàm tiền xử lý ảnh
# Tạo một đối tượng transforms.Compose() để chứa các phép biến đổi ảnh
preprocess = transforms.Compose([
    # Resize ảnh về kích thước 256x256
    transforms.Resize(256),
    # Cắt ảnh ở giữa với kích thước 224x224
    transforms.CenterCrop(224),
    # Chuyển ảnh từ định dạng ảnh sang tensor
    transforms.ToTensor(),
    # Chuẩn hóa ảnh theo mean và std của ảnh ImageNet
    transforms.Normalize(
        # mean: Giá trị trung bình của ảnh ImageNet
        mean=[0.485, 0.456, 0.406],
        # std: Độ lệch chuẩn của ảnh ImageNet
        std=[0.229, 0.224, 0.225]
    )
])

# Đọc ảnh từ đường dẫn
img = Image.open("data/image/babydog.jpg")
# Hiển thị ảnh gốc
img.show("Original Image")
# Tiền xử lý ảnh
img_t = preprocess(img)
# Chuyển ảnh thành tensor
batch_t = torch.unsqueeze(img_t, 0)

# Load mô hình ResNet với weights
resnet = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
#
resnet.eval()

# Dự đoán ảnh
out = resnet(batch_t)

# Đọc file labels.txt
with open("data/txt/imagenet_classes.txt") as f:
    labels = [line.strip() for line in f.readlines()]

# Lấy ra vị trí có giá trị lớn nhất
_, index = torch.max(out, 1)

# Tính xác suất
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

# In ra kết quả dự đoán cao nhất
print(labels[index[0]], percentage[index[0]].item())

# In ra top 5 kết quả dự đoán
_, indices = torch.sort(out, descending=True)
top5 = [(labels[idx], percentage[idx].item()) for idx in indices[0][:5]]
print(top5)
