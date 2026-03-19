/**
 * RNN 可视化应用主逻辑
 * 连接UI、RNN模型和可视化模块
 */

// 全局变量
let rnn = null;
let visualizer = null;
let charRNN = null;
let isTraining = false;
let trainingData = null;

// 文本数据集 - 用于字符级语言模型演示
const SAMPLE_TEXTS = [
    'hello world',
    'the quick brown fox jumps over the lazy dog',
    'artificial intelligence is transforming education',
    'machine learning enables computers to learn from data',
    'neural networks are inspired by biological brains',
    'deep learning uses multiple layers to learn features',
    'recurrent neural networks handle sequential data',
    'natural language processing is a key AI application',
    'computer vision enables machines to see and understand',
    'reinforcement learning teaches agents through rewards'
];

/**
 * 初始化应用
 */
function init() {
    // 初始化可视化器
    visualizer = new RNNVisualizer('rnn-canvas');
    visualizer.initializeStructure(4, 6, 4); // 4输入，6隐藏，4输出
    
    // 设置事件监听器
    setupEventListeners();
    
    // 初始化参数显示
    updateParameterDisplays();
    
    // 初始化字符RNN
    initializeCharRNN();
    
    // 显示欢迎信息
    showMessage('RNN可视化演示已就绪！输入字符序列或选择示例文本开始。', 'info');
}

/**
 * 设置事件监听器
 */
function setupEventListeners() {
    // 参数滑块
    document.getElementById('hidden-size').addEventListener('input', function() {
        document.getElementById('hidden-size-value').textContent = this.value;
    });
    
    document.getElementById('learning-rate').addEventListener('input', function() {
        document.getElementById('learning-rate-value').textContent = this.value;
        if (charRNN) {
            charRNN.setLearningRate(parseFloat(this.value));
        }
    });
    
    document.getElementById('epochs').addEventListener('input', function() {
        document.getElementById('epochs-value').textContent = this.value;
    });
    
    // 按钮事件
    document.getElementById('btn-forward').addEventListener('click', runForwardPropagation);
    document.getElementById('btn-train').addEventListener('click', startTraining);
    document.getElementById('btn-predict').addEventListener('click', runPrediction);
    document.getElementById('btn-reset').addEventListener('click', resetModel);
    
    // 输入框事件
    document.getElementById('input-sequence').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            runForwardPropagation();
        }
    });
}

/**
 * 更新参数显示
 */
function updateParameterDisplays() {
    const hiddenSize = document.getElementById('hidden-size').value;
    const learningRate = document.getElementById('learning-rate').value;
    const epochs = document.getElementById('epochs').value;
    
    document.getElementById('hidden-size-value').textContent = hiddenSize;
    document.getElementById('learning-rate-value').textContent = learningRate;
    document.getElementById('epochs-value').textContent = epochs;
}

/**
 * 初始化字符级RNN
 */
function initializeCharRNN() {
    // 使用所有示例文本构建字符集
    const allChars = new Set();
    SAMPLE_TEXTS.forEach(text => {
        for (const char of text) {
            allChars.add(char);
        }
    });
    
    // 添加空格和常用标点
    allChars.add(' ');
    allChars.add('.');
    allChars.add(',');
    
    const charArray = Array.from(allChars).sort();
    const hiddenSize = parseInt(document.getElementById('hidden-size').value);
    
    charRNN = new CharRNN(charArray, hiddenSize);
    charRNN.setLearningRate(parseFloat(document.getElementById('learning-rate').value));
    
    console.log('字符RNN初始化完成，词汇量:', charArray.length);
}

/**
 * 前向传播
 */
async function runForwardPropagation() {
    if (!charRNN) {
        showMessage('请先初始化模型', 'error');
        return;
    }
    
    const inputSequence = document.getElementById('input-sequence').value.toLowerCase().trim();
    if (!inputSequence) {
        showMessage('请输入字符序列', 'warning');
        return;
    }
    
    // 准备输入向量
    const inputVectors = [];
    for (const char of inputSequence) {
        const vector = charRNN.charToVector(char);
        inputVectors.push(vector);
    }
    
    // 运行前向传播
    const rnn = charRNN.getRNN();
    await visualizer.animateForward(inputVectors, rnn);
    
    // 显示结果
    displayForwardResults(inputSequence, rnn);
}

/**
 * 显示前向传播结果
 */
function displayForwardResults(input, rnn) {
    // 输入显示
    const inputDisplay = document.getElementById('input-display');
    inputDisplay.innerHTML = `<span class="highlight">${input}</span>`;
    
    // 隐藏状态变化
    const hiddenDisplay = document.getElementById('hidden-display');
    let hiddenInfo = '';
    for (let t = 0; t < input.length; t++) {
        const h = rnn.getHiddenState(t + 1);
        const avgH = h.reduce((a, b) => a + Math.abs(b), 0) / h.length;
        hiddenInfo += `t=${t}: 均值=${avgH.toFixed(3)}<br>`;
    }
    hiddenDisplay.innerHTML = hiddenInfo;
    
    // 输出预测
    const outputDisplay = document.getElementById('output-display');
    let outputInfo = '';
    for (let t = 0; t < input.length; t++) {
        const y = rnn.getOutput(t);
        const predictedIdx = rnn.predict(y);
        const predictedChar = charRNN.indexToChar_(predictedIdx);
        const prob = y[predictedIdx];
        outputInfo += `t=${t}: "${predictedChar}" (${(prob * 100).toFixed(1)}%)<br>`;
    }
    outputDisplay.innerHTML = outputInfo;
    
    // 更新解释
    updateExplanation('forward', input);
    
    showMessage('前向传播完成！', 'success');
}

/**
 * 开始训练
 */
function startTraining() {
    if (isTraining) {
        showMessage('训练正在进行中...', 'warning');
        return;
    }
    
    if (!charRNN) {
        showMessage('请先初始化模型', 'error');
        return;
    }
    
    // 重新初始化（应用新的隐藏层大小）
    initializeCharRNN();
    
    // 准备训练数据
    const text = document.getElementById('input-sequence').value.toLowerCase().trim();
    if (!text) {
        // 随机选择一个示例文本
        trainingData = SAMPLE_TEXTS[Math.floor(Math.random() * SAMPLE_TEXTS.length)];
    } else {
        trainingData = text;
    }
    
    const epochs = parseInt(document.getElementById('epochs').value);
    
    isTraining = true;
    updateTrainingUI(true);
    
    showMessage(`开始训练 "${trainingData}" (${epochs} epochs)...`, 'info');
    
    // 开始训练
    charRNN.train(trainingData, epochs, onTrainingProgress);
}

/**
 * 训练进度回调
 */
function onTrainingProgress(epoch, totalEpochs, loss, done) {
    const progress = ((epoch / totalEpochs) * 100).toFixed(1);
    
    document.getElementById('training-progress').style.width = `${progress}%`;
    document.getElementById('current-epoch').textContent = `${epoch}/${totalEpochs}`;
    document.getElementById('current-loss').textContent = loss.toFixed(4);
    
    if (done) {
        isTraining = false;
        updateTrainingUI(false);
        showMessage(`训练完成！最终Loss: ${loss.toFixed(4)}`, 'success');
        updateExplanation('trained', trainingData);
    }
}

/**
 * 运行预测
 */
function runPrediction() {
    if (!charRNN) {
        showMessage('请先初始化模型', 'error');
        return;
    }
    
    const sequence = document.getElementById('input-sequence').value.toLowerCase().trim();
    if (!sequence || sequence.length < 2) {
        showMessage('请输入至少2个字符的序列', 'warning');
        return;
    }
    
    // 预测下一个字符
    const predictions = charRNN.predictNext(sequence);
    
    // 显示结果
    const outputDisplay = document.getElementById('output-display');
    let outputInfo = `<strong>给定序列:</strong> "${sequence}"<br><br>`;
    outputInfo += '<strong>预测下一个字符:</strong><br>';
    
    predictions.forEach((pred, i) => {
        const bar = '█'.repeat(Math.floor(pred.probability * 20));
        outputInfo += `${i + 1}. "${pred.char}" ${(pred.probability * 100).toFixed(1)}% ${bar}<br>`;
    });
    
    outputDisplay.innerHTML = outputInfo;
    
    // 更新解释
    updateExplanation('prediction', sequence);
    
    showMessage('预测完成！', 'success');
}

/**
 * 重置模型
 */
function resetModel() {
    if (isTraining) {
        showMessage('训练中，无法重置', 'warning');
        return;
    }
    
    // 重新初始化
    initializeCharRNN();
    
    // 清空显示
    document.getElementById('input-display').textContent = '-';
    document.getElementById('hidden-display').textContent = '-';
    document.getElementById('output-display').textContent = '-';
    document.getElementById('training-progress').style.width = '0%';
    document.getElementById('current-epoch').textContent = '0';
    document.getElementById('current-loss').textContent = '-';
    
    // 重置可视化
    visualizer.resetHighlight();
    visualizer.draw();
    
    // 重置时间步显示
    document.querySelectorAll('.timestep').forEach(ts => {
        ts.classList.remove('active');
        ts.querySelectorAll('.neuron').forEach(n => n.classList.remove('active'));
    });
    
    updateExplanation('reset', '');
    
    showMessage('模型已重置', 'info');
}

/**
 * 更新训练UI状态
 */
function updateTrainingUI(training) {
    const buttons = document.querySelectorAll('.button-group button');
    buttons.forEach(btn => {
        if (btn.id !== 'btn-reset') {
            btn.disabled = training;
        }
    });
    
    document.getElementById('hidden-size').disabled = training;
    document.getElementById('learning-rate').disabled = training;
    document.getElementById('epochs').disabled = training;
    document.getElementById('input-sequence').disabled = training;
}

/**
 * 更新解释文本
 */
function updateExplanation(type, data) {
    const explanation = document.getElementById('explanation-text');
    
    const explanations = {
        forward: `
            <strong>前向传播演示：</strong><br><br>
            输入序列: "${data}"<br><br>
            RNN逐个时间步处理每个字符：<br>
            • 每个时间步接收当前输入字符和上一时刻的隐藏状态<br>
            • 计算新的隐藏状态，记忆序列信息<br>
            • 产生对下一个字符的预测<br><br>
            隐藏状态h<sub>t</sub>携带了之前所有字符的信息，使RNN能够理解上下文。
        `,
        
        trained: `
            <strong>训练完成！</strong><br><br>
            训练文本: "${data}"<br><br>
            通过${parseInt(document.getElementById('epochs').value)}轮训练，RNN学会了：<br>
            • 字符之间的统计规律<br>
            • 序列的语法结构<br>
            • 预测下一个字符的能力<br><br>
            可以使用"预测下一个"功能测试模型的泛化能力。
        `,
        
        prediction: `
            <strong>预测演示：</strong><br><br>
            给定序列: "${data}"<br><br>
            基于已学到的规律，RNN预测下一个最可能的字符。<br>
            预测基于：<br>
            • 当前输入的上下文<br>
            • 隐藏状态中保存的历史信息<br>
            • 训练过程中学到的权重<br><br>
            概率越高，表示RNN越确信该字符是合理的下一个字符。
        `,
        
        reset: `
            <strong>模型已重置</strong><br><br>
            所有参数已重新随机初始化，RNN回到初始状态。<br><br>
            现在可以：<br>
            • 输入新的序列进行前向传播<br>
            • 使用新的文本训练模型<br>
            • 调整参数（隐藏层大小、学习率等）
        `
    };
    
    explanation.innerHTML = explanations[type] || explanations['reset'];
}

/**
 * 显示消息
 */
function showMessage(message, type = 'info') {
    // 创建消息元素
    const msgElement = document.createElement('div');
    msgElement.className = `message message-${type}`;
    msgElement.textContent = message;
    
    // 样式
    msgElement.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 15px 20px;
        background: ${type === 'error' ? '#f56565' : 
                     type === 'success' ? '#48bb78' : 
                     type === 'warning' ? '#ed8936' : '#667eea'};
        color: white;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        z-index: 1000;
        animation: slideIn 0.3s ease-out;
    `;
    
    document.body.appendChild(msgElement);
    
    // 3秒后自动消失
    setTimeout(() => {
        msgElement.style.animation = 'slideOut 0.3s ease-in';
        setTimeout(() => msgElement.remove(), 300);
    }, 3000);
}

// 添加CSS动画
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(400px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(400px);
            opacity: 0;
        }
    }
    
    .message {
        font-size: 14px;
        font-weight: 500;
        max-width: 300px;
    }
`;
document.head.appendChild(style);

/**
 * 生成示例文本
 */
function generateSampleText() {
    return SAMPLE_TEXTS[Math.floor(Math.random() * SAMPLE_TEXTS.length)];
}

// 页面加载完成后初始化
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}

// 导出函数供调试使用
window.RNNApp = {
    init,
    runForwardPropagation,
    startTraining,
    runPrediction,
    resetModel,
    generateSampleText,
    getCharRNN: () => charRNN,
    getVisualizer: () => visualizer
};
