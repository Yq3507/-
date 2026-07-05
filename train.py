import os
import re
import torch
import torch.nn as nn
from torchvision import transforms, models
from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image

app = Flask(__name__)

# ==========================================
# 🛠️ 核心配置区域
# ==========================================
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

TXT_FILE_PATH = os.path.join(os.path.dirname(__file__), 'fitness_diet.txt')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'diet_model.pth')
CLASSES_PATH = os.path.join(os.path.dirname(__file__), 'classes.txt')
DATASET_DIR = os.path.join(os.path.dirname(__file__), 'test_images')

# 全局模型与类别状态
AI_MODEL = None
CLASS_NAMES = []


# ==========================================
# 🧠 辅助工具与模型自适应加载
# ==========================================
def clean_folder_name(name):
    if not name or not isinstance(name, str): return None
    name = re.sub(r'[\\/:*?"<>|.]', '', name.strip().replace(" ", ""))
    return name if len(name) > 0 else None


def load_trained_model():
    global AI_MODEL, CLASS_NAMES
    try:
        if not os.path.exists(CLASSES_PATH) or not os.path.exists(MODEL_PATH):
            print("⚠️ 未发现本地微调权重或类别映射文件，请先运行 train_models.py 进行训练。")
            return

        with open(CLASSES_PATH, 'r', encoding='utf-8') as f:
            CLASS_NAMES = [line.strip() for line in f.readlines() if line.strip()]

        AI_MODEL = models.convnext_base()
        num_ftrs = AI_MODEL.classifier[2].in_features
        AI_MODEL.classifier[2] = nn.Linear(num_ftrs, len(CLASS_NAMES))

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        AI_MODEL.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        AI_MODEL = AI_MODEL.to(device).eval()
        print(f"🔮 [系统激活] 成功载入本地 ConvNeXt 模型！当前分类数：{len(CLASS_NAMES)}")
    except Exception as e:
        print(f"❌ 载入本地 AI 模型失败: {e}")


def auto_train_model_if_needed():
    if os.path.exists(MODEL_PATH) and os.path.exists(CLASSES_PATH):
        load_trained_model()
    else:
        print("⚠️ [检测提示] 未发现本地微调权重，请运行配套的单独训练脚本以生成 diet_model.pth。")


# ==========================================
# 🎨 纯本地图像单分类预测
# ==========================================
pred_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def predict_image_by_pytorch(image_path):
    if AI_MODEL is None or not CLASS_NAMES: return "其他"
    try:
        device = next(AI_MODEL.parameters()).device
        img = Image.open(image_path).convert("RGB")
        img_tensor = pred_transforms(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = AI_MODEL(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]

        max_idx = torch.argmax(probabilities).item()
        confidence = probabilities[max_idx].item()
        local_predicted_result = CLASS_NAMES[max_idx]
        print(f"🧠 [本地初筛] ConvNeXt 结果: 【{local_predicted_result}】，置信度: {confidence:.2%}")
        return local_predicted_result
    except Exception as e:
        print(f"神经网络推理异常: {e}")
        return "其他"


# ==========================================
# 📊 文本数据集检索引擎
# ==========================================
def query_csv_dataset(class_filter=None, keyword=None, column_to_search=None):
    if not os.path.exists(TXT_FILE_PATH): return []
    parsed_recipes = []

    try:
        with open(TXT_FILE_PATH, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        with open(TXT_FILE_PATH, 'r', encoding='gbk') as f:
            lines = f.readlines()

    record_pattern = re.compile(
        r'^\d+\.\s*(.*?)[，,](.*?)[，,](.*?)[，,]\s*(\d+)\s*[，,]\s*(\d+)\s*[，,]\s*(\d+)\s*[，,]\s*(\d+)\s*[，,]\s*(.*?)[，,](.*?)[，,](.*)$')

    for line in lines:
        match = record_pattern.match(line.strip())
        if match:
            g = [item.strip() for item in match.groups()]
            parsed_recipes.append({
                "本地大类标签": g[0], "菜谱名称": g[1], "主要食材与配料": g[2],
                "热量(kcal)": int(g[3]), "蛋白质(g)": int(g[4]), "碳水(g)": int(g[5]), "脂肪(g)": int(g[6]),
                "适用体质与目标(增肌/减脂)": g[7], "推荐餐段": g[8], "专家营养素小建议与卖点": g[9]
            })

    if not parsed_recipes: return []

    filtered_by_class = parsed_recipes
    if class_filter and class_filter != "Other":
        filtered_by_class = [r for r in parsed_recipes if
                             class_filter in r["本地大类标签"] or class_filter in r["菜谱名称"]]
        if not filtered_by_class: filtered_by_class = parsed_recipes

    if not keyword: return filtered_by_class[:3]

    clean_kw = str(keyword).split("(")[0].strip()
    matched_results = []
    keywords = [k.strip() for k in clean_kw.split("、") if k.strip()]

    for r in filtered_by_class:
        for kw in keywords:
            val = r.get(column_to_search, "") if column_to_search else "".join(str(v) for v in r.values())
            if kw.lower() in str(val).lower():
                matched_results.append(r)
                break

    return matched_results[:3] if matched_results else filtered_by_class[:3]


# ==========================================
# 🤖 业务调度与后端路由 (保持数据通道稳定)
# ==========================================
def core_rag_workflow(text_input=None, image_path=None):
    intent, detected_class, search_col, text_keyword = "chat", None, None, text_input
    welcome_msg = "正在智能检索本地知识系统..."

    if image_path:
        detected_class = predict_image_by_pytorch(image_path)
        welcome_msg = f"📸 本地神经网络识别膳食大类为：【{detected_class}】"
        intent = "image_guided_rag"

    if text_input:
        if any(k in text_input for k in ["不能吃", "禁忌", "痛风"]):
            intent = "knowledge_graph_cypher"
            welcome_msg = "⚠️ 检测到特殊体质限制，正在融合交叉检索健康禁忌..."
        elif any(k in text_input for k in ["减脂", "刷脂", "减肥"]):
            search_col, text_keyword = "适用体质与目标(增肌/减脂)", "减脂"
            welcome_msg = welcome_msg + " 🏃‍♂️ 联动为您匹配该分类下的【减脂】专属食谱..." if image_path else "🏃‍♂️ 检索本地知识表中【减脂】期专属食谱..."
        elif any(k in text_input for k in ["增肌", "强壮", "长肉"]):
            search_col, text_keyword = "适用体质与目标(增肌/减脂)", "增肌"
            welcome_msg = welcome_msg + " 💪 联动为您匹配该分类下的【增肌】专属食谱..." if image_path else "💪 正在为您检索本地知识表中【增肌】期专属食谱..."
        else:
            if not image_path:
                intent = "csv_global_match"
                welcome_msg = f"🔍 正在全表中检索与【{text_input}】相关的营养素指标..."

    if intent == "knowledge_graph_cypher":
        db_results = [{"提示": "知识图谱关系流命中", "体质": "痛风", "禁忌": "高嘌呤红肉及海鲜"}]
        display_class = "痛风限制"
    else:
        db_results = query_csv_dataset(class_filter=detected_class, keyword=text_keyword, column_to_search=search_col)
        display_class = detected_class if detected_class else text_input

    total_calories = sum(r.get('热量(kcal)', 0) for r in db_results if '热量(kcal)' in r)
    avg_cal = int(total_calories / len(db_results)) if db_results and "提示" not in db_results[0] else 0

    goal_hint = "保持饮食均衡，控制总热量摄入。"
    if intent == "knowledge_graph_cypher":
        goal_hint = "严防痛风突发！日常请多饮水促进尿酸排泄，禁止摄入高嘌呤红肉及浓海鲜汤。"
    elif text_input:
        if any(k in text_input for k in ["减脂", "减肥"]):
            goal_hint = f"本组推荐平均热量为 {avg_cal} kcal。当前处于减脂刷脂期，重点在于制造热量缺口并维持高蛋白防止肌肉流失。"
        elif any(k in text_input for k in ["增肌", "长肉"]):
            goal_hint = f"本组推荐平均热量为 {avg_cal} kcal。当前处于增肌期，重点在于补充复合碳水促合成，并提供充足的蛋白质红肉。"
    elif avg_cal > 0:
        goal_hint = f"本组推荐食谱平均热量为 {avg_cal} kcal。根据您的身体指标与健身训练诉求，请按需调配。"

    return {
        "intent_detected": intent,
        "extracted_entity": display_class,
        "llm_analysis": welcome_msg,
        "recipes_list": db_results,
        "advice": goal_hint
    }


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    # 核心更改：使用独立的模板文件进行渲染
    return render_template('index.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/api/diet/analysis', methods=['POST'])
def diet_analysis():
    text_query = request.form.get('query')
    image_file = request.files.get('image')
    image_path, web_image_url = None, None

    if image_file and image_file.filename != '' and allowed_file(image_file.filename):
        raw_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
        image_file.save(raw_path)
        image_path = raw_path
        web_image_url = f"/uploads/{image_file.filename}"

    workflow_result = core_rag_workflow(text_input=text_query, image_path=image_path)
    return jsonify({
        "status": "success",
        "intent_detected": workflow_result["intent_detected"],
        "extracted_entity": workflow_result["extracted_entity"],
        "llm_analysis": workflow_result["llm_analysis"],
        "recipes_list": workflow_result["recipes_list"],
        "advice": workflow_result["advice"],
        "image_url": web_image_url
    })


if __name__ == '__main__':
    auto_train_model_if_needed()
    app.run(host='0.0.0.0', port=5000, debug=True)