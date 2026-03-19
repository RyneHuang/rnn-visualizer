/**
 * RNN 可视化应用主逻辑（修复版）
 * 连接UI、RNN模型、数据集和可视化模块
 */

// 全局变量
let rnn = null;
let visualizer = null;
let charRNN = null;
let isTraining = false;
let datasetManager = null;
let trainingData = null;

/**
 * 初始化应用
 */
function init() {
    // 初始化数据集管理器
    datasetManager = new DatasetManager();
    
    // 初始化可视化器
    visualizer = new RNNVisualizer('rnn-canvas');
    visualizer.initializeStructure(4, 6, 4);
    
    // 设置事件监听器
    setupEventListeners();
    
    // 初始化参数显示
    updateParameterDisplays();
    
    // 初始化字符RNN
    initializeCharRNN();
    
    // 加载默认数据集
    loadDataset('tang-poetry');
    
    // 显示欢迎信息
    showMessage('RNN可视化演示已就绪！选择数据集，设置参数，开始训练。', 'info');
}

/**
 * 设置事件监听器
 */
function setupEventListeners() {
    // 数据集选择
    document.getElementById('dataset-selector').addEventListener('change', function() {
        const datasetId = this.value;
        loadDataset(datasetId);
        
        // 显示/隐藏自定义输入框
        const customGroup = document.getElementById('custom-input-group');
        customGroup.style.display = datasetId === 'custom' ? 'block' : 'none';
    });
    
    // 预览数据按钮
    document.getElementById('btn-preview-data').addEventListener('click', toggleDataPreview);
    
    // 自定义输入
    document.getElementById('input-sequence').addEventListener('input', function() {
        if (datasetManager.currentDataset === 'custom') {
            datasetManager.setCustomData(this.value);
            updateDataStats(this.value);
        }
    });
    
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
    document.getElementById('btn-train').addEventListener('click', startTraining);
    document.getElementById('btn-forward').addEventListener('click', runForwardPropagation);
    document.getElementById('btn-predict').addEventListener('click', runPrediction);
    document.getElementById('btn-generate').addEventListener('click', runGeneration);
    document.getElementById('btn-reset').addEventListener('click', resetModel);
}

/**
 * 加载数据集
 */
function loadDataset(datasetId) {
    datasetManager.setDataset(datasetId);
    const data = datasetManager.getTrainingData();
    trainingData = data;
    
    // 更新数据统计
    updateDataStats(data);
    
    // 如果数据预览已打开，更新内容
    const preview = document.getElementById('data-preview');
    if (preview.style.display !== 'none') {
        document.getElementById('data-content').textContent = data;
    }
}

/**
 * 切换数据预览
 */
function toggleDataPreview() {
    const preview = document.getElementById('data-preview');
    const button = document.getElementById('btn-preview-data');
    
    if (preview.style.display === 'none') {
        preview.style.display = 'block';
        document.getElementById('data-content').textContent = trainingData || datasetManager.getTrainingData();
        button.textContent = '👁️ 隐藏数据';
    } else {
        preview.style.display = 'none';
        button.textContent = '👁️ 预览数据';
    }
}

/**
 * 更新数据统计
 */
function updateDataStats(text) {
    const stats = datasetManager.getDataStats(text);
    document.getElementById('char-count').textContent = stats.totalChars;
    document.getElementById('unique-chars').textContent = stats.uniqueChars;
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
    const text = trainingData || datasetManager.getTrainingData();
    
    // 构建字符集
    const charSet = new Set([...text]);
    const charArray = Array.from(charSet).sort();
    
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
    
    // 获取用户输入或使用训练数据
    const inputElement = document.getElementById('test-input');
    let demoText = '';
    
    if (inputElement && inputElement.value.trim()) {
        demoText = inputElement.value.trim();
    } else {
        const text = trainingData || datasetManager.getTrainingData();
        if (!text || text.length < 2) {
            showMessage('请输入测试文本或选择训练数据', 'warning');
            return;
        }
        demoText = text.substring(0, 10);
    }
    
    // 准备输入向量
    const inputVectors = [];
    const charArray = [];
    for (const char of demoText) {
        const vector = charRNN.charToVector(char);
        inputVectors.push(vector);
        charArray.push(char);
    }
    
    // 运行前向传播
    const rnn = charRNN.getRNN();
    await visualizer.animateForward(inputVectors, rnn, charArray);
    
    // 显示结果
    displayForwardResults(demoText, rnn);
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
    for (let t = 0; t < Math.min(input.length, 5); t++) {
        const h = rnn.getHiddenState(t + 1);
        const avgH = h.reduce((a, b) => a + Math.abs(b), 0) / h.length;
        hiddenInfo += `t=${t}: 均值=${avgH.toFixed(3)}<br>`;
    }
    hiddenDisplay.innerHTML = hiddenInfo;
    
    // 输出预测
    const outputDisplay = document.getElementById('output-display');
    let outputInfo = '';
    for (let t = 0; t < Math.min(input.length, 5); t++) {
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
    
    const text = trainingData || datasetManager.getTrainingData();
    if (!text || text.length < 10) {
        showMessage('训练数据太短，至少需要10个字符', 'warning');
        return;
    }
    
    // 重新初始化（应用新的隐藏层大小）
    initializeCharRNN();
    
    const epochs = parseInt(document.getElementById('epochs').value);
    
    isTraining = true;
    updateTrainingUI(true);
    
    const datasetName = datasetManager.getCurrentDataset().name;
    showMessage(`开始训练 "${datasetName}" (${epochs} epochs, ${text.length}字符)...`, 'info');
    
    // 开始训练
    charRNN.train(text, epochs, onTrainingProgress);
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
        updateExplanation('trained', trainingData || datasetManager.getTrainingData());
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
    
    const text = trainingData || datasetManager.getTrainingData();
    if (!text || text.length < 2) {
        showMessage('没有可用的文本数据', 'warning');
        return;
    }
    
    // 随机选择一个片段作为输入
    const startIdx = Math.floor(Math.random() * (text.length - 10));
    const sequence = text.substring(startIdx, startIdx + 10);
    
    // 预测下一个字符
    const predictions = charRNN.predictNext(sequence);
    
    // 显示结果
    const outputDisplay = document.getElementById('output-display');
    let outputInfo = `<strong>输入序列:</strong> "${sequence}"<br><br>`;
    outputInfo += '<strong>预测下一个字符:</strong><br>';
    
    predictions.forEach((pred, i) => {
        const bar = '█'.repeat(Math.floor(pred.probability * 20));
        const char = pred.char || '?';
        outputInfo += `${i + 1}. "${char}" ${(pred.probability * 100).toFixed(1)}% ${bar}<br>`;
    });
    
    outputDisplay.innerHTML = outputInfo;
    
    // 更新解释
    updateExplanation('prediction', sequence);
    
    showMessage('预测完成！', 'success');
}

/**
 * 生成文本
 */
function runGeneration() {
    if (!charRNN) {
        showMessage('请先初始化模型', 'error');
        return;
    }
    
    const text = trainingData || datasetManager.getTrainingData();
    if (!text || text.length < 1) {
        showMessage('没有可用的文本数据', 'warning');
        return;
    }
    
    // 随机选择种子字符
    const seedIdx = Math.floor(Math.random() * text.length);
    const seedChar = text[seedIdx];
    
    // 生成文本
    const generated = charRNN.generate(seedChar, 50);
    
    // 显示结果
    const outputDisplay = document.getElementById('output-display');
    outputDisplay.innerHTML = `
        <strong>种子字符:</strong> "${seedChar}"<br><br>
        <strong>生成文本:</strong><br>
        <div style="font-family: 'Courier New', monospace; line-height: 1.8; padding: 10px; background: rgba(0,0,0,0.2); border-radius: 6px;">
            ${generated}
        </div>
    `;
    
    // 更新解释
    updateExplanation('generation', generated);
    
    showMessage(`生成了 ${generated.length} 个字符`, 'success');
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
    document.querySelectorAll('.timestep-box').forEach(ts => {
        ts.classList.remove('active');
        ts.classList.remove('processed');
        ts.querySelectorAll('.neuron').forEach(n => n.classList.remove('active'));
        ts.querySelectorAll('.value-display').forEach(vd => vd.textContent = '');
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
    document.getElementById('dataset-selector').disabled = training;
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
            <strong>循环连接（RNN核心特点）：</strong><br>
            隐藏状态h<sub>t</sub>通过循环连接传递到下一时间步，<br>
            使网络能够"记住"之前的信息，这是RNN处理序列数据的关键。
        `,
        
        trained: `
            <strong>训练完成！</strong><br><br>
            训练数据: ${data.length}个字符<br><br>
            通过训练，RNN学会了：<br>
            • 字符之间的统计规律<br>
            • 序列的语法结构<br>
            • 预测下一个字符的能力<br><br>
            <strong>可以尝试：</strong><br>
            • "预测文本" - 输入一段文本，预测下一个字符<br>
            • "生成文本" - 从随机种子生成新文本
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
        
        generation: `
            <strong>文本生成演示：</strong><br><br>
            生成结果: "${data}"<br><br>
            RNN通过以下方式生成文本：<br>
            1. 从种子字符开始<br>
            2. 预测下一个字符的概率分布<br>
            3. 从分布中采样一个字符<br>
            4. 将采样的字符作为下一个输入<br>
            5. 重复步骤2-4<br><br>
            循环连接使得每个生成的字符都依赖于之前所有的上下文。
        `,
        
        reset: `
            <strong>模型已重置</strong><br><br>
            所有参数已重新随机初始化，RNN回到初始状态。<br><br>
            现在可以：<br>
            • 选择新的训练数据集<br>
            • 调整参数（隐藏层大小、学习率等）<br>
            • 重新训练模型
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
    runGeneration,
    resetModel,
    getCharRNN: () => charRNN,
    getVisualizer: () => visualizer,
    getDatasetManager: () => datasetManager
};
