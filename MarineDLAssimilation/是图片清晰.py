from PIL import Image, ImageDraw, ImageFont

# 加载图片
input_image_path = r"C:\Users\dell\Downloads\CNN3.jpg"
output_image_path = r"C:\Users\dell\Downloads\CNN3_modified.jpg"

# 打开原始图片
img = Image.open(input_image_path)

# 示例操作：在图片上添加文本
draw = ImageDraw.Draw(img)
font = ImageFont.load_default()  # 使用默认字体
text = "Modified Image"
draw.text((10, 10), text, font=font, fill="white")  # 在图片左上角添加白色文本

# 保存修改后的图片
img.save(output_image_path)

print(f"图片已保存至 {output_image_path}")