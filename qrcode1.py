import qrcode
from PIL import  Image
#创建QRCode对象
qr = qrcode.QRCode(version = 5,
                   error_correction = qrcode.constants.ERROR_CORRECT_H,
                   box_size = 10,
                   border = 4)

#设置二维码中的数据
data = "https://www.qq.com"
qr.add_data(data)

#填充数据并且生成二维码
qr.make(fit=True)
img = qr.make_image()
img = img.convert('RGBA')

#加载 logo 图像
logo = Image.open("0001.png")

#logo缩小为 img二维码的 1/4
# 计算 logo 的位置
img_w, img_h = img.size
factor = 4
size_w = int(img_w / factor)
size_h = int(img_h / factor)
logo_w, logo_h = logo.size
if (logo_w > size_w) or (logo_h > size_h):
    logo_w = size_w
    logo_h = size_h
logo = logo.resize((logo_w, logo_h), Image.Resampling.LANCZOS).convert('RGBA')
#Image.Resampling.LANCZOS 是一种重采样方法，用于在调整图像大小时进行高质量的插值计算，以获得较好的图像质量。
#放置中心位置
l_w = int((img_w-logo_w)/2)
l_h = int((img_h-logo_h)/2)

#将 logo 嵌入二维码中
img.paste(logo,(l_w,l_h),logo)
img.show()
img.save('qrcode2.png')



















