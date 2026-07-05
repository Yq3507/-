智能健身膳食系统
🌟 系统简介
智能健身膳食系统是一款融合本地 ConvNeXt 深度学习模型与通义千问多模态大模型的智能膳食推荐工具。支持「图片食材识别」「文本需求检索」「个性化食谱推荐」三大核心功能，可根据食材识别结果或用户健身目标（如减脂 / 增肌），结合本地膳食专家数据库，生成精准、精简的健身膳食方案。
📋 核心功能
表格
功能模块	能力说明
图片食材识别	上传食材图片，本地 ConvNeXt 模型初筛 + 通义千问 VL 模型二次验证，精准识别食材
文本需求检索	支持输入食材名称 / 健身目标（如 “减脂”“痛风禁忌”），检索本地膳食数据库
双阶段模型训练	自动清洗无标签数据、分阶段微调 ConvNeXt 模型，适配自定义食材分类
个性化食谱推荐	结合本地膳食数据 + 千问文本大模型，输出精简的食谱和营养建议
可视化交互界面	简洁美观的 Web 界面，支持图片预览、结果可视化展示
🛠️ 环境依赖
基础环境
Python 3.8+
PyTorch 1.12+（建议 GPU 版本，支持 CUDA 更佳）
Flask 2.0+
其他依赖：pandas Pillow requests torchvision
安装依赖
bash
运行
pip install torch torchvision flask pandas pillow requests
📁 项目结构
plaintext
├── 00/                    # 未标注食材图片文件夹（待自动分类）
├── test_images/           # 自动分类后的食材数据集（训练用）
├── fitness_diet.csv       # 本地膳食专家数据库（食谱/营养信息）
├── classes.txt            # 食材分类标签文件（自动生成）
├── diet_model.pth         # 训练后的ConvNeXt模型权重（自动生成）
├── train.py               # 主程序（模型加载+Web服务+核心逻辑）
├── train_models.py        # 数据清洗+模型训练脚本
├── uploads/               # 图片上传临时文件夹（自动生成）
⚙️ 配置说明
关键配置项（需手动修改）
通义千问 API Key
在 train.py 和 train_models.py 中替换以下配置为你的有效 API Key（需确保有余额）：
python
运行
QWEN_API_KEY = "你的通义千问API Key"  # 替换为有效Key
QWEN_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
本地文件路径
可根据实际需求调整以下路径（默认无需修改）：
python
运行
# 膳食数据库路径
CSV_FILE_PATH = os.path.join(os.path.dirname(__file__), 'fitness_diet.csv')
# 数据集路径
DATASET_DIR = os.path.join(os.path.dirname(__file__), 'test_images')
# 未标注图片路径
UNLABELED_DIR = os.path.join(os.path.dirname(__file__), '00')
🚀 快速启动
步骤 1：数据准备
将无标签的食材图片放入 00/ 文件夹；
准备 fitness_diet.csv 本地膳食数据库（格式示例）：
表格
食材名称	适用体质与目标	推荐食谱	营养说明
鸡胸肉	减脂 / 增肌	香煎鸡胸肉 + 西兰花	高蛋白低脂肪
牛肉	增肌	清炒牛肉 + 糙米饭	优质蛋白 + 铁
步骤 2：模型训练
bash
运行
# 运行数据清洗+模型训练脚本
python train_models.py
脚本会自动将 00/ 中的图片通过通义千问 VL 模型打标，分类到 test_images/；
分两阶段训练 ConvNeXt 模型：冻结主干训练分类层→解冻全网络微调；
训练完成后生成 classes.txt（分类标签）和 diet_model.pth（模型权重）。
步骤 3：启动 Web 服务
bash
运行
# 运行主程序
python train.py
程序会自动检测是否有训练好的模型，无则触发自动训练；
启动 Flask 服务，默认地址：http://0.0.0.0:5000
步骤 4：使用系统
打开浏览器访问 http://localhost:5000；
可选操作：
图片识别：上传食材图片，点击 “生成定制膳食食谱”；
文本检索：输入食材名称 / 健身目标（如 “减脂”“痛风禁忌”），点击生成方案；
查看系统输出的 “本地专家库推荐食谱” 和 “极简营养精算小建议”。
🧠 核心流程说明
1. 模型训练流程
plaintext
无标签图片(00/) → 通义千问VL打标 → 分类到test_images/ → 检测分类数（≥2才训练）
→ 阶段1：冻结ConvNeXt主干，训练分类层 → 阶段2：解冻全网络，低速微调 → 保存模型权重
2. 推理流程
plaintext
用户输入（图片/文本）→ 意图识别 → 
├─ 图片输入：本地ConvNeXt初筛（置信度≥85%直接返回，否则调用千问VL二次验证）
├─ 文本输入：根据关键词检索本地CSV数据库
└─ 调用千问文本大模型 → 结合本地数据生成精简食谱和营养建议 → 前端展示
⚠️ 注意事项
API Key 有效性：必须替换为有效且有余额的通义千问 API Key，否则图片打标 / 验证功能失效；
数据集要求：test_images/ 中需至少包含 2 个分类文件夹，否则模型训练会被拒绝；
GPU 加速：建议使用 GPU 训练（CUDA），CPU 训练速度极慢；
CSV 编码：fitness_diet.csv 支持 UTF-8/GBK 编码，程序会自动适配；
端口占用：默认端口 5000，若被占用可修改 train.py 中 app.run 的 port 参数。
📌 常见问题
Q1：训练时提示 “仅包含一个分类”？
A：检查 test_images/ 文件夹，确保至少有 2 个不同的食材分类文件夹；若 00/ 图片打标后全为 “其他”，需检查 API Key 是否有效。
Q2：图片识别结果不准确？
A：1. 确保模型训练时数据集足够；2. 本地模型置信度低于 85% 时，系统会自动调用千问 VL 二次验证，需确保 API Key 可用。
Q3：Web 界面无法访问？
A：检查是否启动成功、端口是否被占用、防火墙是否放行 5000 端口；若部署在服务器，需将 app.run 的 host 设为 0.0.0.0（默认已设置）。
🎯 优化建议
扩充数据集：增加 00/ 中食材图片数量，提升模型泛化能力；
调整置信度阈值：修改 train.py 中 CONFIDENCE_THRESHOLD（默认 0.85），平衡精度和召回率；
轻量化模型：若显存不足，将 convnext_base 替换为 convnext_tiny；
扩充 CSV 数据库：丰富 fitness_diet.csv 中的食谱和营养信息，提升推荐质量。
