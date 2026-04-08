import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 原始数据
data = [
    [0, 0.401980198, 0.344554455, 0.382178218, 0.20990099, 0.211881188, 0.231683168],
    [0.324384787, 0, 0.337807606, 0.431767338, 0.246085011, 0.219239374, 0.277404922],
    [0.331313131, 0.402020202, 0, 0.343434343, 0.202020202, 0.202020202, 0.183838384],
    [0.645454545, 0.711363636, 0.630681818, 0, 0.148863636, 0.2625, 0.355681818],
    [0.601796407, 0.663672655, 0.605788423, 0.25249501, 0, 0.249500998, 0.328343313],
    [0.564551422, 0.618161926, 0.567833698, 0.289934354, 0.177242888, 0, 0.320568928],
    [0.535329341, 0.613173653, 0.516167665, 0.320958084, 0.194011976, 0.256287425, 0]
]

# 模型名称
models = [
    "claude3.5-sonnet", "gpt-4o", "gemini1.5-flash",
    "qwen2.5-vl-7b-instruct", "idefics2-8b",
    "llama3.2-11b-vision-instruct", "smolvlm2-2b-instruct"
]

# 创建DataFrame
df = pd.DataFrame(data, index=models, columns=models)

# 设置字体
plt.rcParams.update({
    "font.size": 18,
    "font.family": "Times New Roman"
})

# 创建图像和热图
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(df, annot=True, fmt=".2%", cmap="Blues", vmin=0, vmax=1, cbar=True,
            linewidths=0.5, linecolor='gray', square=True, ax=ax)

# 设置坐标轴字体大小
ax.set_xticklabels(ax.get_xticklabels(), fontsize=18, rotation=45, ha='right')
ax.set_yticklabels(ax.get_yticklabels(), fontsize=18, rotation=0)

# 保存为PDF，去除白边
plt.tight_layout(pad=0)
plt.savefig("heatmap.pdf", format="pdf", bbox_inches='tight', pad_inches=0)

