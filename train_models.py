import os
import re
import time
import base64
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image

# 🌟 引入可视化绘图库（如果提示找不到，请在终端运行: pip install matplotlib）
import matplotlib.pyplot as plt

# ==========================================
# 🔑 核心配置区域
# ==========================================
UNLABELED_DIR = os.path.join(os.path.dirname(__file__), '00')
DATASET_DIR = os.path.join(os.path.dirname(__file__), 'test_images')

# ⚠️ 必须保证这个 Key 有余额且有效，否则洗出来全是“其他”，模型无法训练！
QWEN_API_KEY = "你的apikey"
QWEN_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
QWEN_MODEL_NAME = "qwen-vl-plus"

# 🎯 严格定义的 8 大核心标准膳食分类
ALLOWED_CLASSES = ["油炸食品", "汤类", "面类", "肉类", "鸡蛋", "乳制品", "米饭", "蔬菜与水果"]


# ==========================================
# 🧠 模块一：千问 Qwen-VL 八分类定向打标引擎
# ==========================================
def encode_image_to_base64(image_path):
    try:
        img = Image.open(image_path)
        img.thumbnail((512, 512))
        from io import BytesIO
        buffered = BytesIO()
        img.convert("RGB").save(buffered, format="JPEG", quality=85)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"⚠️ 图片 {os.path.basename(image_path)} 像素压缩失败: {e}")
        return None


def clean_folder_name(name):
    if not name or not isinstance(name, str):
        return None
    name = name.strip().replace(" ", "").replace("\n", "").replace("\r", "")
    name = re.sub(r'[\\/:*?"<>|.]', '', name)
    return name if len(name) > 0 else None


def ask_qwen_vl_to_label_strictly(image_path):
    base64_image = encode_image_to_base64(image_path)
    if not base64_image:
        return "其他"

    headers = {
        "Authorization": f"Bearer {QWEN_API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = (
        "仔细观察这张食物图片。请从以下指定的 8 个分类中，选择一个最符合该食物属性的分类名称：\n"
        "【油炸食品、汤类、面类、肉类、鸡蛋、乳制品、米饭、蔬菜与水果】\n\n"
        "【极其严格的格式要求】：\n"
        "1. 你只能且必须回答上述 8 个名词中的其中一个，禁止自己发明新词。\n"
        "2. 严禁回答任何标点符号、解释、一句话或形容词！例如，如果是炸鸡，只需回答：油炸食品"
    )

    payload = {
        "model": QWEN_MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        "max_tokens": 15,
        "temperature": 0.1
    }

    session = requests.Session()
    session.trust_env = False
    session_proxies = {"http": None, "https": None}

    # 🌟 核心升级：增加 3 次自动重试机制
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # 将 timeout 延长至 30 秒，防止大图传输超时
            response = session.post(QWEN_API_URL, json=payload, headers=headers, proxies=session_proxies, timeout=30)

            if response.status_code == 200:
                raw_label = response.json()['choices'][0]['message']['content']
                cleaned_label = clean_folder_name(raw_label)

                if cleaned_label in ALLOWED_CLASSES:
                    return cleaned_label
                else:
                    print(f"⚠️ 大模型返回了非预设标签 [{cleaned_label}]，已降级归入“其他”")
                    return "其他"
            elif response.status_code == 401:
                print(f" ⚠️ [401错误] 鉴权失败！请检查 API_KEY 是否有效。")
                return "其他"

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            print(f"🔄 图片 {os.path.basename(image_path)} 请求超时/网络抖动 (第 {attempt + 1}/{max_retries} 次重试)...")
            if attempt < max_retries - 1:
                time.sleep(2)  # 等待 2 秒后再次尝试
            else:
                print(f"❌ 连续 {max_retries} 次网络请求超时，该图片强制归入“其他”")
        except Exception as e:
            print(f"⚠️ 遇到未知异常: {e}")
            break

    return "其他"


def auto_label_and_classify_dataset():
    if not os.path.exists(UNLABELED_DIR):
        return

    os.makedirs(DATASET_DIR, exist_ok=True)
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    unlabeled_files = [f for f in os.listdir(UNLABELED_DIR) if f.lower().endswith(valid_extensions)]

    if not unlabeled_files:
        return

    print(f"🚀 [定向无监督清洗中] 正在将 {len(unlabeled_files)} 张散图按照 8 大标准分类分流...")
    for idx, img_name in enumerate(unlabeled_files):
        src_img_path = os.path.join(UNLABELED_DIR, img_name)
        predicted_label = ask_qwen_vl_to_label_strictly(src_img_path)

        target_folder = os.path.join(DATASET_DIR, predicted_label)
        os.makedirs(target_folder, exist_ok=True)

        base, ext = os.path.splitext(img_name)
        target_img_path = os.path.join(target_folder, img_name)
        counter = 1
        while os.path.exists(target_img_path):
            target_img_path = os.path.join(target_folder, f"{base}_{counter}{ext}")
            counter += 1

        os.rename(src_img_path, target_img_path)
    print("✅ 散图全定向分流归仓完毕！")


# ==========================================
# 📊 辅助模块：专业级指标绘图引擎
# ==========================================
def draw_training_metrics(history_loss, history_acc):
    try:
        # 设置支持中文的全局字体（防止乱码）
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False

        epochs = range(1, len(history_loss) + 1)

        # 创建画布
        plt.figure(figsize=(12, 5))

        # 1. 绘制 Loss 曲线
        plt.subplot(1, 2, 1)
        plt.plot(epochs, history_loss, 'r-o', label='训练损失值 (Loss)', linewidth=2)
        plt.title('模型收敛趋势 (Loss Curve)', fontsize=13, fontweight='bold', pad=10)
        plt.xlabel('训练周期 (Epoch)', fontsize=11)
        plt.ylabel('损失度 (Loss)', fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=10)

        # 2. 绘制 Accuracy 曲线
        plt.subplot(1, 2, 2)
        plt.plot(epochs, history_acc, 'b-^', label='本地精准度 (Accuracy)', linewidth=2)
        plt.title('拟合精确度进化 (Accuracy Curve)', fontsize=13, fontweight='bold', pad=10)
        plt.xlabel('训练周期 (Epoch)', fontsize=11)
        plt.ylabel('正确率 (Accuracy)', fontsize=11)
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))  # 百分比显示
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=10)

        plt.tight_layout()
        save_path = os.path.join(os.path.dirname(__file__), 'training_metrics.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"\n📈 [深度可视化] 已自动生成专业级微调评估图表: {save_path}")
    except Exception as e:
        print(f"⚠️ 指标图表渲染失败 (不影响权重保存): {e}")


# ==========================================
# 🏋️‍♂️ 模块二：精细化深度网络级微调训练流程
# ==========================================
def train_my_diet_model():
    # 1. 先触发定向数据分流清洗
    auto_label_and_classify_dataset()

    if not os.path.exists(DATASET_DIR) or len(os.listdir(DATASET_DIR)) == 0:
        print(f"❌ 错误：未在目标路径 {DATASET_DIR} 下检测到任何有效分类，训练中止。")
        return

    available_classes = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    if len(available_classes) <= 1:
        print(f"\n❌ [拒绝训练] 当前仅包含 {available_classes} 一个分类，数据不足以进行分类器训练。")
        return

    print("\n" + "=" * 50)
    print("🏋️‍♂️ 标准 8 分类数据集检测通过！ConvNeXt 深度微调引擎启动...")
    print("=" * 50)

    data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(DATASET_DIR, transform=data_transforms)
    class_names = dataset.classes
    total_images = len(dataset)
    print(f"✅ 成功映射分类拓扑树！当前已激活分类共 {len(class_names)} 种：{class_names}")
    print(f"📊 样本总体积: {total_images} 张图片。")

    with open('classes.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(class_names))

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)

    print("\n🚀 正在拉取现代高级预训练权重网络模型 (ConvNeXt_Base)...")
    model = models.convnext_base(pretrained=True)

    num_ftrs = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(num_ftrs, len(class_names))

    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"🔥 加速算力节点就绪: 【{device}】。")

    # 用于保存绘图数据的数据魔方
    plot_losses = []
    plot_accs = []

    # ==========================================
    # 📈 阶段一：冻结骨干，快速热身分类层（10个Epoch）
    # ==========================================
    print("\n[⚡ 优化阶段一激活]：锁死主干参数，快速热身新分类头...")
    for name, param in model.named_parameters():
        if "classifier.2" not in name:
            param.requires_grad = False

    optimizer_stage1 = optim.AdamW(model.classifier[2].parameters(), lr=0.003, weight_decay=1e-4)

    stage1_epochs = 10
    for epoch in range(stage1_epochs):
        model.train()
        running_loss = 0.0
        corrects = 0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer_stage1.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_stage1.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / total_images
        epoch_acc = corrects.double() / total_images

        # 记录热身阶段指标
        plot_losses.append(epoch_loss)
        plot_accs.append(epoch_acc.item())

        print(f"📊 [Warmup - Epoch {epoch + 1:02d}/{stage1_epochs}] Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2%}")

    # ==========================================
    # 📈 阶段二：全线解锁，深度全局微调（10个Epoch）
    # ==========================================
    print("\n[🔥 优化阶段二激活]：全面解冻整个模型，进入收拢微调...")
    for param in model.parameters():
        param.requires_grad = True

    optimizer_stage2 = optim.AdamW(model.parameters(), lr=0.00005, weight_decay=1e-4)

    stage2_epochs = 10
    scheduler = CosineAnnealingLR(optimizer_stage2, T_max=stage2_epochs, eta_min=1e-7)

    for epoch in range(stage2_epochs):
        model.train()
        running_loss = 0.0
        corrects = 0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer_stage2.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_stage2.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data)

        scheduler.step()
        current_lr = optimizer_stage2.param_groups[0]['lr']

        epoch_loss = running_loss / total_images
        epoch_acc = corrects.double() / total_images

        # 记录微调阶段指标
        plot_losses.append(epoch_loss)
        plot_accs.append(epoch_acc.item())

        print(
            f"🚀 [FineTune - Epoch {epoch + 1:02d}/{stage2_epochs}] Loss: {epoch_loss:.4f} | 本地精准度(Acc): {epoch_acc:.2%} | 实时LR: {current_lr:.7f}")

    # 3. 固化最优权重
    MODEL_PATH = 'diet_model.pth'
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\n🏆 [完美收官] 精简 8 分类 ConvNeXt 模型已固化写盘: {MODEL_PATH}")

    # 4. 触发专业级绘图
    draw_training_metrics(plot_losses, plot_accs)


if __name__ == '__main__':
    train_my_diet_model()