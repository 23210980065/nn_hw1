import matplotlib.pyplot as plt
import seaborn as sns
from model import load_model

# 参数热力图
def plot_heatmap(weights, layer_name):
    plt.figure(figsize=(10, 8))
    sns.heatmap(weights, annot=False, cmap='coolwarm', center=0)
    plt.title(f"Heatmap for {layer_name}")
    plt.xlabel("Input Neurons")
    plt.ylabel("Output Neurons")
    plt.show()

#  权重直方图
def plot_histogram(weights, layer_name):
    plt.figure(figsize=(6, 4))
    plt.hist(weights.flatten(), bins=50, alpha=0.75)
    plt.title(f"Histogram of weights in {layer_name}")
    plt.xlabel("Weight value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

# 偏置项可视化
def plot_biases(biases, layer_name):
    plt.figure(figsize=(6, 4))
    plt.bar(range(len(biases)), biases)
    plt.title(f"Biases in {layer_name}")
    plt.xlabel("Neuron")
    plt.ylabel("Bias value")
    plt.show()

# 绘制并存储
best_model_relu = load_model("best_model_relu")
best_model_sigmoid = load_model("best_model_sigmoid")

plot_heatmap(best_model_relu["W1"],"W1", "output/relu/heatmap/W1.png")
plot_heatmap(best_model_sigmoid["W1"],"W1", "output/sigmoid/heatmap/W1.png")

plot_heatmap(best_model_relu["W2"],"W2", "output/relu/heatmap/W2.png")
plot_heatmap(best_model_sigmoid["W2"],"W2", "output/sigmoid/heatmap/W2.png")

plot_heatmap(best_model_relu["W3"],"W3", "output/relu/heatmap/W3.png")
plot_heatmap(best_model_sigmoid["W3"],"W3", "output/sigmoid/heatmap/W3.png")

plot_histogram(best_model_relu["W1"],"W1", "output/relu/histogram/W1.png")
plot_histogram(best_model_sigmoid["W1"],"W1", "output/sigmoid/histogram/W1.png")

plot_histogram(best_model_relu["W2"],"W2", "output/relu/histogram/W2.png")
plot_histogram(best_model_sigmoid["W2"],"W2", "output/sigmoid/histogram/W2.png")

plot_histogram(best_model_relu["W3"],"W3", "output/relu/histogram/W3.png")
plot_histogram(best_model_sigmoid["W3"],"W3", "output/sigmoid/histogram/W3.png")

plot_biases(best_model_relu["b1"],"b1", "output/relu/biases/W1.png")
plot_biases(best_model_sigmoid["b1"],"b1", "output/sigmoid/biases/W1.png")

plot_biases(best_model_relu["b2"],"b2", "output/relu/biases/W2.png")
plot_biases(best_model_sigmoid["b2"],"b2", "output/sigmoid/biases/W1.png")

plot_biases(best_model_relu["b3"],"b3", "output/relu/biases/W3.png")
plot_biases(best_model_sigmoid["b3"],"b3", "output/sigmoid/biases/W1.png")