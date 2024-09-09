import torch
from PIL import Image
from torchvision import transforms
import torch
import torch.nn as nn

# Định nghĩa lớp ResNetBlock
class ResNetBlock(nn.Module): # <1>
    # Hàm khởi tạo
    def __init__(self, dim):
        # Gọi hàm khởi tạo của lớp cha
        super(ResNetBlock, self).__init__()
        # Gán giá trị dim cho biến dim
        self.conv_block = self.build_conv_block(dim)
    # Hàm xây dựng block Convolution
    def build_conv_block(self, dim):
        # Khởi tạo một mảng conv_block rỗng
        conv_block = []
        # Thêm một lớp ReflectionPad2d với padding = 1 vào mảng conv_block
        conv_block += [nn.ReflectionPad2d(1)]
        # Thêm một lớp Conv2d với dim, dim, kernel_size=3, padding=0, bias=True vào mảng conv_block
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
                       nn.InstanceNorm2d(dim),
                       nn.ReLU(True)]
        # Thêm một lớp ReflectionPad2d với padding = 1 vào mảng conv_block
        conv_block += [nn.ReflectionPad2d(1)]
        # Thêm một lớp Conv2d với dim, dim, kernel_size=3, padding=0, bias=True vào mảng conv_block
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
                       nn.InstanceNorm2d(dim)]
        # Trả về mảng conv_block
        return nn.Sequential(*conv_block)
    # Hàm forward
    def forward(self, x):
        # Tính giá trị out
        out = x + self.conv_block(x) # <2>
        return out

# Định nghĩa lớp ResNetGenerator
class ResNetGenerator(nn.Module):
    # Hàm khởi tạo
    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=9): # <3>
        # Gọi hàm khởi tạo của lớp cha
        assert(n_blocks >= 0)
        super(ResNetGenerator, self).__init__()
        # Gán giá trị input_nc, output_nc, ngf cho các biến tương ứng
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=True),
                 nn.InstanceNorm2d(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=True),
                      nn.InstanceNorm2d(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResNetBlock(ngf * mult)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=True),
                      nn.InstanceNorm2d(int(ngf * mult / 2)),
                      nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input): # <3>
        return self.model(input)

netG = ResNetGenerator()

# Nạp trọng số đã được huấn luyện
model_path = 'data/pth/horse2zebra_0.4.0.pth'
model_data = torch.load(model_path, map_location=torch.device('cpu'))  # Nạp trọng số vào CPU
netG.load_state_dict(model_data)

# Đặt mô hình vào chế độ đánh giá
netG.eval()

# Import PIL và torchvision để xử lý hình ảnh
# Đã import ở trên

# Định nghĩa các phép biến đổi đầu vào
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),  # Thêm phép cắt để đảm bảo kích thước đầu vào đúng
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Đảm bảo chuẩn hóa hình ảnh
])

# Mở file ảnh ngựa
img = Image.open("data/image/horse.jpg").convert('RGB')  # Đảm bảo ảnh có 3 kênh màu RGB

# Tiền xử lý và tạo biến đầu vào cho mô hình
img_t = preprocess(img)
batch_t = torch.unsqueeze(img_t, 0)  # Thêm một chiều batch

# Gửi hình ảnh vào mô hình để chuyển đổi
with torch.no_grad():  # Không tính gradient cho quá trình dự đoán
    batch_out = netG(batch_t)

# Chuyển đổi kết quả đầu ra thành hình ảnh và hiển thị
out_t = (batch_out.squeeze().cpu() + 1.0) / 2.0  # Chuyển đổi từ [-1, 1] về [0, 1]
out_img = transforms.ToPILImage()(out_t)

# Lưu hình ảnh kết quả (nếu cần) và hiển thị
out_img.save('output/zebra.jpg')
out_img.show()
