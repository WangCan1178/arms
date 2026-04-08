import matplotlib.pyplot as plt
# 设置全局字体
plt.rcParams.update({'font.size': 18, 'font.family': 'Times New Roman'})
# 数据
data = {
    # "Data Size": [20, 40, 60, 80, 90, 100],
"Data Size": [20, 40, 60, 80, 90, 95, 100],
    "Accuracy": [0.652072388, 0.66209096, 0.670965058, 0.678488033, 0.677452579, 0.676452579,0.675124792],
    # "Precision": [0.663505311, 0.682371295, 0.621186911, 0.643649473, 0.66112138, 0.640321686],
    # "Recall": [0.538817006, 0.606489185, 0.68986757, 0.691803279, 0.660510926, 0.688226528],
    "F1": [0.594695682, 0.642196447, 0.653728294, 0.66685821, 0.66341043, 0.66112138, 0.660816012, ],
    "EM": [0.561674009, 0.563876652, 0.565528634, 0.56938326, 0.570484581, 0.571354,0.572136564],
}
#
# 创建折线图
plt.figure(figsize=(8, 6))

plt.plot(data["Data Size"], data["Accuracy"], 'o-', label="Accuracy", color='b', markersize=6)
# plt.plot(data["Data Size"], data["Precision"], 'o-', label="Precision", color='g', markersize=6)
# plt.plot(data["Data Size"], data["Recall"], 'o-', label="Recall", color='r', markersize=6)
plt.plot(data["Data Size"], data["F1"], 'o-', label="F1-score", color='r', markersize=6)
plt.plot(data["Data Size"], data["EM"], 'o-', label="EM", color='g', markersize=6)

for label, values in data.items():
    if label != "Data Size":
        max_value = max(values)
        max_index = values.index(max_value)
        plt.text(data["Data Size"][max_index], max_value-0.002, f'{max_value:.4f}', fontsize=18, ha='center', va='top')

# 设置字体
plt.rcParams.update({'font.size': 18, 'font.family': 'Times New Roman'})
# 设置legend和坐标轴字体大小
plt.legend(fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
# 设置标题和标签
# plt.title('Performance Metrics vs Data Size', fontsize=18)
plt.xlabel('Data Amount (%)', fontsize=18)
# plt.ylabel('Metric Values', fontsize=18)

# 设置纵坐标范围
plt.ylim(0.55, 0.68)


# 保存为PDF
plt.tight_layout(pad=0)
plt.savefig('samples-performance.pdf', format='pdf', bbox_inches='tight')

# 显示图形
plt.show()
