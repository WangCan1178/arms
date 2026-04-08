
import matplotlib.pyplot as plt
# 设置全局字体
plt.rcParams.update({'font.size': 18, 'font.family': 'Times New Roman'})
# 数据
data = {
"Data Size": [2, 4, 6, 8, 10],
    "Accuracy": [0.474082702, 0.509680639, 0.545278576, 0.580876514, 0.652072388],
    "F1": [0.125847047, 0.219616774, 0.313386501, 0.407156228, 0.594695682],
    "EM": [0.4321, 0.445572687, 0.45522467, 0.485575365, 0.555066079]
}
#
# 创建折线图
plt.figure(figsize=(8, 6))

plt.plot(data["Data Size"], data["Accuracy"], 'o-', label="Accuracy", color='b', markersize=6)
# plt.plot(data["Data Size"], data["Precision"], 'o-', label="Precision", color='g', markersize=6)
# plt.plot(data["Data Size"], data["Recall"], 'o-', label="Recall", color='r', markersize=6)
plt.plot(data["Data Size"], data["F1"], 'o-', label="F1-score", color='r', markersize=6)
plt.plot(data["Data Size"], data["EM"], 'o-', label="EM", color='g', markersize=6)

# for label, values in data.items():
#     if label != "Data Size":
#         max_value = max(values)
#         max_index = values.index(max_value)
#         plt.text(data["Data Size"][max_index], max_value-0.002, f'{max_value:.4f}', fontsize=18, ha='center', va='top')

for label, values in data.items():
    if label != "Data Size":
        for i, value in enumerate(values):
            plt.text(data["Data Size"][i], value - 0.002, f'{value:.4f}', fontsize=18, ha='center', va='top')

# 设置字体
plt.rcParams.update({'font.size': 18, 'font.family': 'Times New Roman'})
# 设置legend和坐标轴字体大小
plt.legend(fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
# 设置标题和标签
# plt.title('Performance Metrics vs Data Size', fontsize=18)
plt.xlabel('Epochs', fontsize=18)
# plt.ylabel('Metric Values', fontsize=18)

# 设置纵坐标范围
plt.ylim(0.1, 0.68)


# 保存为PDF
plt.tight_layout(pad=0)
plt.savefig('ood_performance_epochs.pdf', format='pdf', bbox_inches='tight')

# 显示图形
plt.show()
