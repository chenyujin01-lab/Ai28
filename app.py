import requests, time, numpy as np, json, os, threading
from datetime import datetime
from flask import Flask, render_template_string

app = Flask(__name__)

# --------------------------
# 基础配置与分类定义
# --------------------------
DATA_FILE = "ai_brain.json"

def get_category(s):
    if s in [1, 3, 5, 7, 9, 11, 13]: return "小单"
    if s in [0, 2, 4, 6, 8, 10, 12]: return "小双"
    if s in [14, 16, 18, 20, 22, 24, 26]: return "大双"
    if s in [15, 17, 19, 21, 23, 25, 27]: return "大单"
    return "未知"

# --------------------------
# 持久化存储管理
# --------------------------
class BrainStorage:
    @staticmethod
    def load():
        if os.path.exists(DATA_FILE):
            try:
                with open(DATA_FILE, 'r') as f:
                    return json.load(f)
            except: pass
        return {
            "total": 0, "sum_h": 0, "cat_h": 0,
            "weights": {"lcg": 1.0, "lagrange": 1.0, "vmd": 1.0},
            "last_qihao": "",
            "last_sum": 0,
            "predictions": [],
            "trend": [],
            "last_update": ""
        }

    @staticmethod
    def save(data):
        with open(DATA_FILE, 'w') as f:
            json.dump(data, f)

# --------------------------
# 核心反解算法引擎
# --------------------------
class RandomCrackEngine:
    def lcg_logic(self, sums):
        """修复版LCG：增加逆元存在性检查"""
        if len(sums) < 3: return 13
        x1, x2, x3 = sums[-3], sums[-2], sums[-1]
        
        base = (x2 - x1) % 29
        if base != 0:
            try:
                inv = pow(base, -1, 29)
                a = (x3 - x2) * inv % 29
                c = (x3 - a * x2) % 29
                return (a * x3 + c) % 28
            except: 
                return (27 - x3) % 28
        return (x3 + 11) % 28

    def vmd_adaptive(self, sums):
        """残差分析：监测偏离均值的回归趋势"""
        if len(sums) < 10: return 14
        trend = np.mean(sums[-10:])
        res = sums[-1] - trend
        return int(round(trend - 0.7 * res)) % 28

    def lagrange_logic(self, sums):
        """拉格朗日插值：寻找多项式曲线干扰（纯Python实现）"""
        if len(sums) < 5: return 13
        y = sums[-4:]
        n = len(y)
        # 简化版插值预测
        x_new = n
        result = 0
        for i in range(n):
            term = y[i]
            for j in range(n):
                if i != j:
                    term = term * (x_new - j) / (i - j)
            result += term
        return int(abs(result)) % 28

# --------------------------
# AI决策中心（简化版，移除TensorFlow）
# --------------------------
class PersistentAI:
    def __init__(self):
        self.brain = BrainStorage.load()
        self.weights = self.brain.get("weights", {"lcg": 1.0, "lagrange": 1.0, "vmd": 1.0})
        self.cracker = RandomCrackEngine()

    def predict(self, sums):
        # 1. 子引擎预测
        p_lcg = self.cracker.lcg_logic(sums)
        p_lag = self.cracker.lagrange_logic(sums)
        p_vmd = self.cracker.vmd_adaptive(sums)

        all_p = {"lcg": int(p_lcg), "lagrange": int(p_lag), "vmd": int(p_vmd)}
        
        # 2. 加权投票
        scores = np.zeros(28)
        for k, v in all_p.items():
            scores[v % 28] += self.weights.get(k, 1.0)
        
        # 混沌震荡修正
        if len(sums) >= 2 and abs(sums[-1] - sums[-2]) > 9:
            scores[27 - sums[-1]] += 0.5

        rec_sums = scores.argsort()[-2:][::-1]
        return [int(x) for x in rec_sums], all_p

    def update_and_save(self, all_p, actual, rec_sums, rec_cats):
        act_cat = get_category(actual)
        self.brain["total"] += 1
        
        is_s_hit = actual in rec_sums
        is_c_hit = act_cat in rec_cats
        
        if is_s_hit: self.brain["sum_h"] += 1
        if is_c_hit: self.brain["cat_h"] += 1

        # 动态权重演化
        for m, p in all_p.items():
            if p == actual: 
                self.weights[m] = min(self.weights[m] + 0.3, 5.0)
            else: 
                self.weights[m] = max(self.weights[m] * 0.9, 0.5)
        
        self.brain["weights"] = self.weights
        BrainStorage.save(self.brain)
        return is_s_hit, is_c_hit

# --------------------------
# 全局AI实例
# --------------------------
ai = PersistentAI()

# --------------------------
# 后台更新线程
# --------------------------
def background_updater():
    """后台线程：持续监控并更新预测"""
    while True:
        try:
            r = requests.get("https://www.gaga28.com/gengduo.php?page=1&type=1", timeout=10).json()
            history = r["data"]
            sums = [int(d["sum"]) for d in history][::-1]
            latest = history[0]
            l_qihao, l_sum = int(latest["qihao"]), int(latest["sum"])
            
            # 如果是新一期，先验证上期预测
            if ai.brain.get("last_qihao") and ai.brain["last_qihao"] != str(l_qihao):
                # 检查上期预测是否命中
                if ai.brain.get("predictions"):
                    old_rec_sums = ai.brain["predictions"]
                    old_rec_cats = [get_category(old_rec_sums[0]), get_category(old_rec_sums[1])]
                    old_all_p = ai.brain.get("all_predictions", {})
                    ai.update_and_save(old_all_p, l_sum, old_rec_sums, old_rec_cats)
            
            # 生成新预测
            rec_sums, all_p = ai.predict(sums)
            rec_cats = [get_category(rec_sums[0]), get_category(rec_sums[1])]
            
            # 强制组合多样性
            if rec_cats[0] == rec_cats[1]:
                potential_cats = [get_category(i) for i in all_p.values()]
                for c in potential_cats:
                    if c != rec_cats[0]:
                        rec_cats[1] = c
                        break
            
            # 更新状态
            ai.brain["last_qihao"] = str(l_qihao)
            ai.brain["last_sum"] = l_sum
            ai.brain["predictions"] = rec_sums
            ai.brain["rec_cats"] = rec_cats
            ai.brain["all_predictions"] = all_p
            ai.brain["trend"] = sums[-30:]
            ai.brain["last_update"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            BrainStorage.save(ai.brain)
            
        except Exception as e:
            print(f"更新失败: {e}")
        
        time.sleep(30)  # 每30秒更新一次

# 启动后台线程
thread = threading.Thread(target=background_updater, daemon=True)
thread.start()

# --------------------------
# Web界面
# --------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>极速28 AI预测</title>
    <style>
        body {
            font-family: 'Courier New', monospace;
            background: #0a0a0a;
            color: #00ff00;
            padding: 20px;
            margin: 0;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: #1a1a1a;
            border: 2px solid #00ff00;
            padding: 20px;
            border-radius: 10px;
        }
        h1 {
            text-align: center;
            color: #ff0066;
            text-shadow: 0 0 10px #ff0066;
        }
        .section {
            margin: 20px 0;
            padding: 15px;
            border: 1px solid #333;
            background: #0d0d0d;
        }
        .label {
            color: #00ffff;
            font-weight: bold;
        }
        .value {
            color: #ffff00;
            font-size: 1.2em;
        }
        .trend {
            word-wrap: break-word;
            color: #ff9900;
            line-height: 1.8;
        }
        .stats {
            color: #ff00ff;
        }
        .prediction {
            font-size: 1.5em;
            color: #00ff00;
            text-align: center;
            padding: 10px;
            background: #003300;
            border-radius: 5px;
        }
        .time {
            text-align: center;
            color: #666;
            font-size: 0.9em;
            margin-top: 20px;
        }
        .hit {
            color: #00ff00;
        }
        .miss {
            color: #ff0000;
        }
    </style>
    <script>
        // 每10秒自动刷新页面
        setTimeout(function(){
            location.reload();
        }, 10000);
    </script>
</head>
<body>
    <div class="container">
        <h1>🤖 极速28 AI矩阵预测系统</h1>
        
        <div class="section">
            <div class="label">📊 当前期号:</div>
            <div class="value">{{ qihao }}</div>
        </div>
        
        <div class="section">
            <div class="label">🎯 上期开奖:</div>
            <div class="value">和值 {{ last_sum }} ({{ last_cat }})</div>
        </div>
        
        <div class="section prediction">
            <div>🔮 下期预测</div>
            <div style="margin-top: 10px;">
                和值: {{ pred_sums[0] }}, {{ pred_sums[1] }}
            </div>
            <div style="margin-top: 5px;">
                组合: {{ pred_cats[0] }} + {{ pred_cats[1] }}
            </div>
        </div>
        
        <div class="section">
            <div class="label">📈 近30期走势:</div>
            <div class="trend">{{ trend }}</div>
        </div>
        
        <div class="section stats">
            <div class="label">📊 历史统计:</div>
            <div>总预测次数: {{ total }}</div>
            <div>和值命中率: <span class="{{ 'hit' if sum_rate > 50 else 'miss' }}">{{ sum_rate }}%</span></div>
            <div>组合命中率: <span class="{{ 'hit' if cat_rate > 50 else 'miss' }}">{{ cat_rate }}%</span></div>
        </div>
        
        <div class="section">
            <div class="label">⚙️ 模型权重:</div>
            <div>
                {% for k, v in weights.items() %}
                    {{ k.upper() }}: {{ "%.2f"|format(v) }}  
                {% endfor %}
            </div>
        </div>
        
        <div class="time">
            最后更新: {{ last_update }}<br>
            页面将在10秒后自动刷新...
        </div>
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    brain = ai.brain
    total = brain.get("total", 0) or 1
    sum_rate = round((brain.get("sum_h", 0) / total) * 100, 1)
    cat_rate = round((brain.get("cat_h", 0) / total) * 100, 1)
    
    trend_str = " ".join([str(x) for x in brain.get("trend", [])])
    
    return render_template_string(
        HTML_TEMPLATE,
        qihao=brain.get("last_qihao", "等待数据..."),
        last_sum=brain.get("last_sum", "-"),
        last_cat=get_category(brain.get("last_sum", 0)),
        pred_sums=brain.get("predictions", ["-", "-"]),
        pred_cats=brain.get("rec_cats", ["-", "-"]),
        trend=trend_str or "正在加载...",
        total=total,
        sum_rate=sum_rate,
        cat_rate=cat_rate,
        weights=brain.get("weights", {}),
        last_update=brain.get("last_update", "未知")
    )

if __name__ == '__main__':
    # 本地测试用
    app.run(host='0.0.0.0', port=5000, debug=False)
