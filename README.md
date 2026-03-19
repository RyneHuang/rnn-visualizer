# RNN 循环神经网络可视化演示

> 🧠 交互式Web应用，深入理解循环神经网络的结构与原理

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![JavaScript](https://img.shields.io/badge/JavaScript-ES6+-yellow.svg)](https://www.javascript.com/)
[![HTML5](https://img.shields.io/badge/HTML5-E34F26?logo=html5&logoColor=white)](https://developer.mozilla.org/en-US/docs/Web/HTML)

## 📖 项目简介

这是一个面向教学和学习的RNN可视化工具，通过交互式动画和实时演示，帮助理解循环神经网络的核心概念：

- **RNN结构可视化** - 直观展示循环连接和时间步展开
- **训练过程演示** - 观察前向传播和反向传播(BPTT)
- **实时推理** - 输入序列，看RNN如何逐步处理
- **参数调节** - 动态调整隐藏层大小、学习率等参数

## ✨ 功能特性

### 🎨 可视化功能

- **网络结构图** - Canvas绘制的动态神经网络
- **时间步展开** - 展示RNN在多个时间步的状态变化
- **数据流动画** - 可视化前向传播的数据流动过程
- **神经元激活** - 高亮显示当前激活的神经元和连接

### 🎓 教学功能

- **数学公式展示** - 动态显示RNN核心公式
  ```
  h_t = tanh(W_xh · x_t + W_hh · h_{t-1} + b_h)
  y_t = softmax(W_hy · h_t + b_y)
  ```
- **原理解释** - 根据操作更新详细解释文本
- **梯度可视化** - 展示BPTT反向传播过程

### 🔧 交互功能

- **字符级语言模型** - 实现完整的CharRNN
- **实时训练** - 观察Loss下降曲线
- **文本生成** - 训练后生成新文本
- **预测功能** - 预测序列的下一个字符

## 🚀 快速开始

### 方式1: 直接打开（推荐）

1. 克隆仓库
```bash
git clone https://github.com/RyneHuang/rnn-visualizer.git
cd rnn-visualizer
```

2. 启动本地服务器
```bash
# 使用Python
python3 -m http.server 8080

# 或使用Node.js
npx http-server -p 8080
```

3. 打开浏览器访问
```
http://localhost:8080
```

### 方式2: 直接打开HTML文件

直接在浏览器中打开 `index.html` 文件（部分功能可能受限）

## 📚 使用指南

### 基本操作

1. **输入序列**
   - 在输入框中输入字符序列（如 "hello"）
   - 每个字符作为一个时间步的输入

2. **前向传播**
   - 点击 "▶️ 前向传播" 按钮
   - 观察RNN逐时间步处理输入
   - 查看隐藏状态变化和输出预测

3. **训练模型**
   - 点击 "🔄 训练模型" 按钮
   - 观察Loss下降曲线
   - 等待训练完成（默认100轮）

4. **预测下一个**
   - 输入部分序列
   - 点击 "🎯 预测下一个"
   - 查看预测结果和概率分布

5. **调整参数**
   - 隐藏层大小: 4-32
   - 学习率: 0.001-0.1
   - 训练轮数: 10-500

### 示例用例

#### 学习RNN结构
1. 观察时间步展开视图
2. 点击前向传播，看数据流动
3. 理解隐藏状态如何传递信息

#### 理解训练过程
1. 输入简单文本（如 "hello"）
2. 点击训练，观察Loss下降
3. 多次训练，看Loss收敛

#### 测试文本生成
1. 训练模型后
2. 输入种子字符
3. 查看生成的文本

## 🏗️ 技术架构

### 文件结构

```
rnn-visualizer/
├── index.html          # 主页面
├── style.css           # 样式（响应式设计）
├── rnn.js              # RNN核心算法
│   ├── class RNN       # 基础RNN类
│   └── class CharRNN   # 字符级RNN
├── visualize.js        # 可视化模块
│   ├── class RNNVisualizer      # 网络可视化
│   ├── class UnrolledVisualizer # 展开视图
│   └── class FormulaVisualizer  # 公式展示
├── app.js              # 应用主逻辑
└── test.js             # 自动化测试套件
```

### 核心算法

#### RNN前向传播
```javascript
h_t = tanh(W_xh · x_t + W_hh · h_{t-1} + b_h)
y_t = softmax(W_hy · h_t + b_y)
```

#### BPTT反向传播
```javascript
∂L/∂h_t = ∂L/∂y_t · W_hy^T + ∂L/∂h_{t+1} · W_hh^T
```

#### 特性
- **Xavier初始化** - 合理的权重初始化
- **梯度裁剪** - 防止梯度爆炸
- **Softmax归一化** - 输出概率分布

## 🧪 测试

### 运行测试

```bash
node test.js
```

### 测试覆盖

- ✅ RNN初始化
- ✅ Xavier初始化
- ✅ 前向传播（单步和序列）
- ✅ 交叉熵损失计算
- ✅ BPTT反向传播
- ✅ 梯度裁剪
- ✅ CharRNN功能
- ✅ 性能测试

### 测试结果
```
总测试数: 14
✅ 通过: 14
❌ 失败: 0
通过率: 100%
```

## 🎯 应用场景

### 教育教学
- 深度学习课程演示
- RNN原理讲解
- 神经网络可视化教学

### 自学探索
- 理解RNN工作原理
- 观察训练过程
- 实验不同参数

### 研究原型
- 快速验证想法
- 算法对比实验
- 可视化分析

## 📊 性能指标

- **前向传播**: 100步 < 1秒
- **训练速度**: 10轮 < 2秒
- **内存占用**: 纯前端，无服务器压力
- **浏览器兼容**: Chrome, Firefox, Safari, Edge

## 🛠️ 技术栈

- **前端**: HTML5 + CSS3 + JavaScript (ES6+)
- **可视化**: Canvas API
- **算法**: 纯JavaScript实现，无外部依赖
- **测试**: Node.js测试框架

## 🤝 贡献指南

欢迎贡献代码、报告Bug或提出新功能建议！

1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建Pull Request

## 📝 开发计划

- [ ] 添加LSTM可视化
- [ ] 支持GRU结构
- [ ] 添加注意力机制可视化
- [ ] 支持自定义网络结构
- [ ] 添加更多示例数据集
- [ ] 支持模型保存/加载

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 👨‍💻 作者

**Ryne Huang**
- GitHub: [@RyneHuang](https://github.com/RyneHuang)

## 🙏 致谢

- 感谢所有为深度学习教育做出贡献的开源项目
- 灵感来源于Andrej Karpathy的[Char-RNN](https://github.com/karpathy/char-rnn)

## 📮 联系方式

如有问题或建议，欢迎：
- 提交 [Issue](https://github.com/RyneHuang/rnn-visualizer/issues)
- 发送邮件至项目维护者

---

**如果这个项目对你有帮助，请给一个 ⭐️ Star！**

 Made with ❤️ for education
