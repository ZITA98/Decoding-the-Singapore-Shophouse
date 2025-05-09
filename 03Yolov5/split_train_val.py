import os
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import random
from datetime import datetime
import sys

def check_directory(path, create=True):
    """检查目录是否存在，如果create=True则创建"""
    if not os.path.exists(path):
        if create:
            try:
                os.makedirs(path)
                print(f"创建目录: {path}")
            except Exception as e:
                print(f"创建目录失败 {path}: {str(e)}")
                sys.exit(1)
        else:
            print(f"目录不存在: {path}")
            sys.exit(1)
    return True

class DatasetSplitter:
    def __init__(self):
        self.label_names = {
            0: 'Main Pilaster',
            1: 'Secondary Pilaster',
            2: 'Long Window',
            3: 'Casement Window',
            4: 'Fanlight',
            5: 'Modillion',
            6: 'Festoon',
            7: 'Chinese Plaque',
            8: 'Green Glazed Tiles Canopy',
            9: 'Chinese Decorative Panel',
            10: 'Malay Transom',
            11: 'Fretwork Fascia',
            12: 'Majolica Tiles',
            13: 'Stepping Parapet',
            14: 'Modern Window',
            15: 'Shades',
            16: 'Building'
        }
        
        self.image_to_labels = defaultdict(set)
        self.label_to_images = defaultdict(list)
        self.label_counts = defaultdict(int)
        
        self.rare_labels = set()
        self.medium_labels = set()
        self.frequent_labels = set()
        
        self.train_images = set()
        self.val_images = set()
        self.test_images = set()
        
        random.seed(42)
        np.random.seed(42)

    def load_data(self, label_dir):
        """加载数据"""
        print("\n开始加载数据...")
        if not os.path.exists(label_dir):
            print(f"标签目录不存在: {label_dir}")
            sys.exit(1)
            
        label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
        if not label_files:
            print(f"未找到标签文件在: {label_dir}")
            sys.exit(1)
            
        print(f"找到 {len(label_files)} 个标签文件")
        
        # 读取标签文件
        for file in tqdm(label_files, desc="读取标签文件"):
            image_name = file[:-4]
            file_path = os.path.join(label_dir, file)
            
            try:
                with open(file_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            label_id = int(line.split()[0])
                            if label_id in self.label_names:
                                self.label_counts[label_id] += 1
                                self.label_to_images[label_id].append(image_name)
                                self.image_to_labels[image_name].add(label_id)
            except Exception as e:
                print(f"读取文件出错 {file}: {str(e)}")
                continue
                
        print(f"成功加载 {len(self.image_to_labels)} 张图片的标签信息")
        
        # 输出每个标签的样本数量
        print("\n各标签样本数量:")
        for label_id, count in sorted(self.label_counts.items()):
            print(f"{self.label_names[label_id]:<30}: {count:>5}")

    def classify_labels(self, rare_threshold=100, frequent_threshold=500):
        """将标签分类为稀有、中等和高频"""
        print("\n开始对标签进行分类...")
        
        for label, count in self.label_counts.items():
            if count < rare_threshold:
                self.rare_labels.add(label)
            elif count > frequent_threshold:
                self.frequent_labels.add(label)
            else:
                self.medium_labels.add(label)
        
        print(f"稀有标签 (数量 < {rare_threshold}): {len(self.rare_labels)} 个")
        for label in sorted(self.rare_labels):
            print(f"  - {self.label_names[label]}: {self.label_counts[label]}")
            
        print(f"\n中等频率标签 ({rare_threshold} <= 数量 <= {frequent_threshold}): {len(self.medium_labels)} 个")
        for label in sorted(self.medium_labels):
            print(f"  - {self.label_names[label]}: {self.label_counts[label]}")
            
        print(f"\n高频标签 (数量 > {frequent_threshold}): {len(self.frequent_labels)} 个")
        for label in sorted(self.frequent_labels):
            print(f"  - {self.label_names[label]}: {self.label_counts[label]}")

    def split_dataset(self):
        """划分数据集"""
        print("\n开始划分数据集...")
        
        # 1. 处理稀有标签
        rare_images = set()
        for label in self.rare_labels:
            rare_images.update(self.label_to_images[label])
        
        print(f"包含稀有标签的图片数量: {len(rare_images)}")
        rare_images = list(rare_images)
        random.shuffle(rare_images)
        
        # 70:15:15 split for rare labels
        rare_train_size = int(len(rare_images) * 0.7)
        rare_val_size = int(len(rare_images) * 0.15)
        
        self.train_images.update(rare_images[:rare_train_size])
        self.val_images.update(rare_images[rare_train_size:rare_train_size + rare_val_size])
        self.test_images.update(rare_images[rare_train_size + rare_val_size:])
        
        # 2. 处理中等频率标签
        medium_images = set()
        for label in self.medium_labels:
            medium_images.update(self.label_to_images[label])
        medium_images = medium_images - set(rare_images)
        
        print(f"包含中等频率标签的图片数量(排除已分配): {len(medium_images)}")
        medium_images = list(medium_images)
        random.shuffle(medium_images)
        
        # 80:10:10 split for medium frequency labels
        medium_train_size = int(len(medium_images) * 0.8)
        medium_val_size = int(len(medium_images) * 0.1)
        
        self.train_images.update(medium_images[:medium_train_size])
        self.val_images.update(medium_images[medium_train_size:medium_train_size + medium_val_size])
        self.test_images.update(medium_images[medium_train_size + medium_val_size:])
        
        # 3. 处理剩余图片
        remaining_images = set(self.image_to_labels.keys()) - self.train_images - self.val_images - self.test_images
        print(f"剩余图片数量: {len(remaining_images)}")
        remaining_images = list(remaining_images)
        random.shuffle(remaining_images)
        
        # 80:10:10 split for remaining images
        remaining_train_size = int(len(remaining_images) * 0.8)
        remaining_val_size = int(len(remaining_images) * 0.1)
        
        self.train_images.update(remaining_images[:remaining_train_size])
        self.val_images.update(remaining_images[remaining_train_size:remaining_train_size + remaining_val_size])
        self.test_images.update(remaining_images[remaining_train_size + remaining_val_size:])
        
        print("\n划分结果:")
        print(f"训练集: {len(self.train_images)} 张图片")
        print(f"验证集: {len(self.val_images)} 张图片")
        print(f"测试集: {len(self.test_images)} 张图片")

    def save_results(self, image_dir, output_dir, save_dir):
        """保存划分结果和统计信息"""
        print("\n保存划分结果...")
        
        # 检查并创建目录
        check_directory(output_dir)
        check_directory(save_dir)
        
        # 1. 保存数据集划分结果
        splits = {
            'train.txt': self.train_images,
            'val.txt': self.val_images,
            'test.txt': self.test_images
        }

        for filename, images in splits.items():
            output_path = os.path.join(output_dir, filename)
            try:
                with open(output_path, 'w') as f:
                    for img in sorted(images):
                        f.write(f"{os.path.join(image_dir, img)}.jpg\n")
                print(f"已保存: {output_path}")
            except Exception as e:
                print(f"保存文件失败 {output_path}: {str(e)}")
                continue

        # 2. 保存统计信息
        dataset_dirname = os.path.basename(os.path.dirname(output_dir))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stats_filename = f'split_stats_{dataset_dirname}_{timestamp}.txt'
        stats_path = os.path.join(save_dir, stats_filename)
        
        try:
            with open(stats_path, 'w', encoding='utf-8') as f:
                # 计算每个集合中的标签统计
                stats = {'train': defaultdict(int), 'val': defaultdict(int), 'test': defaultdict(int)}
                
                for img in self.train_images:
                    for label in self.image_to_labels[img]:
                        stats['train'][label] += 1
                        
                for img in self.val_images:
                    for label in self.image_to_labels[img]:
                        stats['val'][label] += 1
                        
                for img in self.test_images:
                    for label in self.image_to_labels[img]:
                        stats['test'][label] += 1

                # 写入统计信息
                f.write("数据集划分统计报告\n")
                f.write("=" * 50 + "\n\n")
                
                # 标签分类信息
                for label_type, labels in [
                    ("稀有标签", self.rare_labels),
                    ("中等频率标签", self.medium_labels),
                    ("高频标签", self.frequent_labels)
                ]:
                    f.write(f"\n{label_type}:\n")
                    for label in sorted(labels):
                        name = self.label_names[label]
                        count = self.label_counts[label]
                        train_count = stats['train'][label]
                        val_count = stats['val'][label]
                        test_count = stats['test'][label]
                        
                        f.write(f"{name:<30}: 总数={count:>5}, "
                               f"训练集={train_count:>5} ({train_count/count*100:>5.1f}%), "
                               f"验证集={val_count:>5} ({val_count/count*100:>5.1f}%), "
                               f"测试集={test_count:>5} ({test_count/count*100:>5.1f}%)\n")
                
                # 整体统计
                total_images = len(self.image_to_labels)
                f.write(f"\n整体统计:\n")
                f.write(f"总图片数: {total_images}\n")
                f.write(f"训练集: {len(self.train_images)} ({len(self.train_images)/total_images*100:.1f}%)\n")
                f.write(f"验证集: {len(self.val_images)} ({len(self.val_images)/total_images*100:.1f}%)\n")
                f.write(f"测试集: {len(self.test_images)} ({len(self.test_images)/total_images*100:.1f}%)\n")
                
            print(f"已保存统计报告: {stats_path}")
            
        except Exception as e:
            print(f"保存统计信息失败: {str(e)}")

def main():
    # 设置路径
    label_dir = "/root/autodl-tmp/yolov5/yolov5/city_data_component_zihui/data_R2_01/labels"
    image_dir = "/root/autodl-tmp/yolov5/yolov5/city_data_component_zihui/data_R2_01/images"
    output_dir = "/root/autodl-tmp/yolov5/yolov5/city_data_component_zihui/data_R2_01/dataset"
    save_dir = "/root/autodl-tmp/yolov5/yolov5/city_data_component_zihui/data_statistics"
    
    print("数据集划分程序开始运行...")
    print(f"标签目录: {label_dir}")
    print(f"图片目录: {image_dir}")
    print(f"输出目录: {output_dir}")
    print(f"统计目录: {save_dir}")
    
    try:
        splitter = DatasetSplitter()
        
        # 加载数据
        splitter.load_data(label_dir)
        
        # 分类标签
        splitter.classify_labels(rare_threshold=100, frequent_threshold=500)
        
        # 划分数据集
        splitter.split_dataset()
        
        # 保存结果
        splitter.save_results(image_dir, output_dir, save_dir)
        
        print("\n程序成功完成!")
        
    except Exception as e:
        print(f"\n程序执行出错: {str(e)}")
        raise e

if __name__ == "__main__":
    main()