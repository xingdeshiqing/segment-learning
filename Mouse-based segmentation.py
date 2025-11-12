import os
import cv2
import numpy as np
import matplotlib
import torch
import matplotlib.pyplot as plt
from segment_anything import SamPredictor, sam_model_registry
from pathlib import Path
import json
from datetime import datetime

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class InteractiveSegmentTool:
    def __init__(self, model_type="vit_h", checkpoint_path="sam_vit_h_4b8939.pth", 
                 image_path=None, output_folder="results"):
        """
        初始化交互式分割工具
        """
        # 设置设备
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {self.device}")
        
        # 加载模型
        print("正在加载SAM模型...")
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(device=self.device)
        self.predictor = SamPredictor(self.sam)
        
        # 文件路径设置
        self.image_path = Path(image_path) if image_path else None
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True)
        
        # 状态变量
        self.current_image = None
        self.current_image_path = None
        self.click_points = []
        self.click_labels = []  # 1: 正样本点, 0: 负样本点
        self.current_mask = None
        
        print("模型加载完成！")

    def load_image(self, image_path=None):
        """加载单张图片"""
        if image_path is None:
            image_path = self.image_path
            
        if image_path is None or not image_path.exists():
            print(f"错误: 图片路径不存在: {image_path}")
            return None
            
        self.current_image_path = Path(image_path)
        self.current_image = cv2.imread(str(image_path))
        if self.current_image is None:
            print(f"错误: 无法读取图片: {image_path}")
            return None
            
        self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
        
        # 重置点击点
        self.click_points = []
        self.click_labels = []
        self.current_mask = None
        
        # 设置预测器图像
        self.predictor.set_image(self.current_image)
        
        print(f"已加载图片: {self.current_image_path.name}")
        print(f"图片尺寸: {self.current_image.shape}")
        
        return self.current_image

    def on_click(self, event):
        """鼠标点击事件处理"""
        if event.inaxes and event.button == 1:  # 左键点击
            x, y = int(event.xdata), int(event.ydata)
            self.click_points.append([x, y])
            self.click_labels.append(1)
            print(f"添加正样本点: ({x}, {y})")
            self.update_segmentation()

        elif event.inaxes and event.button == 3:  # 右键点击
            x, y = int(event.xdata), int(event.ydata)
            self.click_points.append([x, y])
            self.click_labels.append(0)
            print(f"添加负样本点: ({x}, {y})")
            self.update_segmentation()

    def on_key(self, event):
        """键盘事件处理"""
        if event.key == 'r':  # 按R键重置
            self.click_points = []
            self.click_labels = []
            self.current_mask = None
            self.update_display()
            print("已重置所有点")
            
        elif event.key == 's':  # 按S键保存
            if self.current_mask is not None:
                self.save_results()
            else:
                print("没有可保存的分割结果")
                
        elif event.key == 'n':  # 按N键加载新图片
            self.load_new_image()

    def load_new_image(self):
        """手动输入新图片路径"""
        new_path = input("请输入新图片路径: ").strip().strip('"').strip("'")
        new_path = Path(new_path)
        
        if new_path.exists():
            self.image_path = new_path
            self.load_image(new_path)
            self.update_display()
        else:
            print(f"文件不存在: {new_path}")

    def update_segmentation(self):
        """更新分割结果"""
        if len(self.click_points) == 0:
            return
            
        input_points = np.array(self.click_points)
        input_labels = np.array(self.click_labels)
        
        # 使用模型预测
        masks, scores, logits = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=False,
        )
        
        self.current_mask = masks[0]
        self.update_display()

    def update_display(self):
        """更新显示"""
        plt.clf()
        
        # 显示原图
        plt.subplot(1, 2, 1)
        plt.imshow(self.current_image)
        plt.title("原图 - 左键:添加目标, 右键:排除区域")
        plt.axis('off')
        
        # 显示分割结果
        plt.subplot(1, 2, 2)
        plt.imshow(self.current_image)
        
        if self.current_mask is not None:
            self.show_mask(self.current_mask, plt.gca())
        
        # 显示点击点
        for point, label in zip(self.click_points, self.click_labels):
            color = 'green' if label == 1 else 'red'
            marker = '+' if label == 1 else 'x'
            plt.scatter(point[0], point[1], c=color, marker=marker, s=100, linewidths=2)
        
        plt.title("分割结果 - R:重置 S:保存 N:新图片")
        plt.axis('off')
        
        plt.tight_layout()
        plt.draw()

    def show_mask(self, mask, ax, random_color=False):
        """显示掩码"""
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    def save_results(self):
        """保存分割结果 - 物体为白色，背景为黑色，保持原图分辨率"""
        if self.current_mask is None:
            print("没有分割结果可保存")
            return
            
        # 使用原图片的文件名
        base_name = self.current_image_path.stem
        file_extension = self.current_image_path.suffix
        
        # 创建输出文件名（与原图同名）
        output_filename = f"{base_name}_segmented{file_extension}"
        output_path = self.output_folder / output_filename
        
        # 创建黑色背景的灰度图像（单通道）
        height, width = self.current_image.shape[:2]
        segmented_image = np.zeros((height, width), dtype=np.uint8)  # 黑色背景
        
        # 将分割区域设置为白色 (255)
        segmented_image[self.current_mask] = 255
        
        # 保存图像，保持原图分辨率
        cv2.imwrite(str(output_path), segmented_image)
        
        print(f"分割结果已保存: {output_path}")
        print(f"分辨率: {width} x {height}")

    def run(self):
        """运行交互式工具"""
        if self.image_path is None:
            image_path = input("请输入图片路径: ").strip().strip('"').strip("'")
            self.image_path = Path(image_path)
        
        if not self.image_path.exists():
            print(f"错误: 图片路径不存在: {self.image_path}")
            return
        
        # 加载图片
        if self.load_image() is None:
            return
        
        # 设置图形界面
        plt.rcParams['keymap.save'].remove('s')
        fig = plt.figure(figsize=(15, 8))
        
        # 绑定事件
        fig.canvas.mpl_connect('button_press_event', self.on_click)
        fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        # 初始显示
        self.update_display()
        
        # 显示操作说明
        print("\n=== 操作说明 ===")
        print("左键点击: 添加要分割的区域")
        print("右键点击: 添加要排除的区域") 
        print("R键: 重置所有点")
        print("S键: 保存当前分割结果")
        print("N键: 加载新图片")
        print("================")
        
        plt.show()

# 使用示例
if __name__ == "__main__":
    tool = InteractiveSegmentTool(
        model_type="vit_h",
        checkpoint_path="sam_vit_h_4b8939.pth",
        image_path="D:/segment/images/input/1.jpg",
        output_folder="D:/segment/images/output"
    )
    
    tool.run()
    