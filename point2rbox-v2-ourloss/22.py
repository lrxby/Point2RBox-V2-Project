import os
import csv
from PIL import Image
from tqdm import tqdm  # 修复：正确导入tqdm可调用对象
from collections import defaultdict

# ===================== 配置参数 =====================
# 数据集根路径
ROOT_PATH = "/mnt/data/xiekaikai/split_ss_dota/trainval"
# 图片文件夹名
IMG_FOLDER = "images"
# 标注文件夹名
LABEL_FOLDER = "labelTxt"
# DOTA类别列表（按你的需求定义）
CLASSES = (
    'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
    'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
    'harbor', 'swimming-pool', 'helicopter'
)
# 是否统计标注文件中的类别数量（True/False）
COUNT_CLASSES = True
# 统计结果保存路径（None则仅打印，不保存）
OUTPUT_CSV = "dota_image_stats.csv"

# ===================== 核心函数 =====================
def get_image_info(img_path):
    """获取单张图片的信息（尺寸、文件大小）"""
    try:
        # 获取文件大小（字节），转换为MB（保留4位小数）
        file_size = os.path.getsize(img_path) / (1024 * 1024)
        # 打开图片获取尺寸
        with Image.open(img_path) as img:
            width, height = img.size
        return {
            "file_name": os.path.basename(img_path),
            "width": width,
            "height": height,
            "file_size_mb": round(file_size, 4),
            "status": "success"
        }
    except Exception as e:
        # 处理图片损坏/无法读取的情况
        return {
            "file_name": os.path.basename(img_path),
            "width": 0,
            "height": 0,
            "file_size_mb": 0,
            "status": f"error: {str(e)}"
        }

def count_label_classes(txt_path, classes):
    """统计单个标注文件中的各类别数量"""
    class_count = defaultdict(int)
    if not os.path.exists(txt_path):
        return class_count, "file_not_found"
    
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # 按空格分割行内容，取倒数第二个字段作为类别（如small-vehicle）
            parts = line.split()
            if len(parts) >= 10:  # 确保行格式正确（8个坐标 + 类别 + 难度）
                cls = parts[8]
                if cls in classes:
                    class_count[cls] += 1
        return class_count, "success"
    except Exception as e:
        return class_count, f"error: {str(e)}"

def main():
    # 声明要修改的全局变量（关键修复点1）
    global COUNT_CLASSES
    
    # 1. 检查根路径是否存在
    if not os.path.exists(ROOT_PATH):
        print(f"错误：根路径 {ROOT_PATH} 不存在！")
        return
    
    # 2. 拼接图片和标注文件夹路径
    img_dir = os.path.join(ROOT_PATH, IMG_FOLDER)
    label_dir = os.path.join(ROOT_PATH, LABEL_FOLDER)
    
    # 检查文件夹是否存在
    if not os.path.exists(img_dir):
        print(f"错误：图片文件夹 {img_dir} 不存在！")
        return
    if COUNT_CLASSES and not os.path.exists(label_dir):
        print(f"警告：标注文件夹 {label_dir} 不存在，将跳过类别统计！")
        COUNT_CLASSES = False
    
    # 3. 获取所有图片文件（支持常见格式）
    img_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(img_extensions)]
    
    if not img_files:
        print("错误：图片文件夹中未找到任何图片文件！")
        return
    
    # 4. 遍历图片并统计信息
    all_stats = []  # 存储所有图片的统计信息
    total_size = 0.0  # 总文件大小（MB）
    total_width = 0  # 总宽度
    total_height = 0  # 总高度
    total_images = len(img_files)
    class_total = defaultdict(int)  # 所有标注的类别总数
    
    print(f"\n开始统计 {total_images} 张图片的信息...")
    for img_file in tqdm(img_files):  # 现在tqdm是可调用对象，不会报错
        # 获取图片完整路径
        img_path = os.path.join(img_dir, img_file)
        # 获取图片基本信息
        img_info = get_image_info(img_path)
        
        if img_info["status"] == "success":
            total_size += img_info["file_size_mb"]
            total_width += img_info["width"]
            total_height += img_info["height"]
        
        # 统计标注类别（如果开启）
        if COUNT_CLASSES:
            # 标注文件名为图片文件名替换后缀为.txt
            txt_file = os.path.splitext(img_file)[0] + ".txt"
            txt_path = os.path.join(label_dir, txt_file)
            cls_count, cls_status = count_label_classes(txt_path, CLASSES)
            
            # 合并类别统计到图片信息中
            img_info["label_status"] = cls_status
            for cls in CLASSES:
                img_info[f"{cls}_count"] = cls_count.get(cls, 0)
                class_total[cls] += cls_count.get(cls, 0)
        else:
            img_info["label_status"] = "not_counted"
        
        all_stats.append(img_info)
    
    # 5. 计算汇总统计
    avg_size = round(total_size / total_images, 4) if total_images > 0 else 0
    avg_width = round(total_width / total_images) if total_images > 0 else 0
    avg_height = round(total_height / total_images) if total_images > 0 else 0
    
    # 6. 打印统计结果
    print("\n" + "="*50)
    print("DOTA数据集图片统计结果")
    print("="*50)
    print(f"总图片数量：{total_images}")
    print(f"总文件大小：{round(total_size, 2)} MB")
    print(f"平均文件大小：{avg_size} MB")
    print(f"平均图片尺寸：{avg_width} × {avg_height} 像素")
    
    # 打印类别统计（如果开启）
    if COUNT_CLASSES:
        print("\n标注类别数量统计：")
        for cls in CLASSES:
            print(f"  {cls}: {class_total[cls]} 个")
    
    # 7. 保存统计结果到CSV（如果指定路径）
    if OUTPUT_CSV:
        # 构建CSV表头
        headers = ["file_name", "width", "height", "file_size_mb", "status"]
        if COUNT_CLASSES:
            headers += ["label_status"] + [f"{cls}_count" for cls in CLASSES]
        
        # 写入CSV文件
        with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(all_stats)
        print(f"\n详细统计结果已保存到：{os.path.abspath(OUTPUT_CSV)}")

# ===================== 执行脚本 =====================
if __name__ == "__main__":
    # 安装依赖（如果未安装）
    try:
        import PIL
    except ImportError:
        print("正在安装Pillow库...")
        os.system("pip install pillow")
    try:
        from tqdm import tqdm  # 修复：检查时也用正确的导入方式
    except ImportError:
        print("正在安装tqdm库...")
        os.system("pip install tqdm")
        from tqdm import tqdm  # 安装后重新导入
    
    main()