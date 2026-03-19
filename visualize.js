/**
 * RNN 可视化模块（改进版）
 * 处理RNN结构、数据流动画等可视化功能
 */

class RNNVisualizer {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.neurons = [];
        this.connections = [];
        this.currentTimestep = -1;
        this.animationSpeed = 800; // ms，增加动画速度
    }
    
    /**
     * 初始化RNN结构可视化
     */
    initializeStructure(inputSize, hiddenSize, outputSize) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        
        this.neurons = this.createNeurons();
        this.connections = this.createConnections();
        
        this.draw();
    }
    
    /**
     * 创建神经元节点
     */
    createNeurons() {
        const neurons = [];
        const canvasWidth = this.canvas.width;
        const canvasHeight = this.canvas.height;
        
        const layerSpacing = canvasWidth / 4;
        const inputX = layerSpacing;
        const hiddenX = layerSpacing * 2;
        const outputX = layerSpacing * 3;
        
        // 输入层
        const inputSpacing = canvasHeight / (this.inputSize + 1);
        for (let i = 0; i < this.inputSize; i++) {
            neurons.push({
                id: `input-${i}`,
                type: 'input',
                x: inputX,
                y: inputSpacing * (i + 1),
                radius: 20,
                value: 0,
                active: false
            });
        }
        
        // 隐藏层（只显示一部分，避免太拥挤）
        const hiddenSpacing = canvasHeight / (Math.min(this.hiddenSize, 8) + 1);
        const visibleHiddenSize = Math.min(this.hiddenSize, 8);
        for (let i = 0; i < visibleHiddenSize; i++) {
            neurons.push({
                id: `hidden-${i}`,
                type: 'hidden',
                x: hiddenX,
                y: hiddenSpacing * (i + 1),
                radius: 20,
                value: 0,
                active: false,
                realIndex: i // 记录实际索引
            });
        }
        
        // 输出层
        const outputSpacing = canvasHeight / (this.outputSize + 1);
        for (let i = 0; i < this.outputSize; i++) {
            neurons.push({
                id: `output-${i}`,
                type: 'output',
                x: outputX,
                y: outputSpacing * (i + 1),
                radius: 20,
                value: 0,
                active: false
            });
        }
        
        return neurons;
    }
    
    /**
     * 创建连接线
     */
    createConnections() {
        const connections = [];
        const inputNeurons = this.neurons.filter(n => n.type === 'input');
        const hiddenNeurons = this.neurons.filter(n => n.type === 'hidden');
        const outputNeurons = this.neurons.filter(n => n.type === 'output');
        
        // 输入层到隐藏层的连接
        inputNeurons.forEach(input => {
            hiddenNeurons.forEach(hidden => {
                connections.push({
                    from: input.id,
                    to: hidden.id,
                    weight: Math.random() - 0.5,
                    active: false,
                    type: 'Wxh'
                });
            });
        });
        
        // 隐藏层到隐藏层的循环连接（用虚线表示）
        hiddenNeurons.forEach((hidden1, i) => {
            hiddenNeurons.forEach((hidden2, j) => {
                if (i !== j) {
                    connections.push({
                        from: hidden1.id,
                        to: hidden2.id,
                        weight: Math.random() - 0.5,
                        active: false,
                        type: 'Whh',
                        isRecurrent: true
                    });
                }
            });
        });
        
        // 隐藏层到输出层的连接
        hiddenNeurons.forEach(hidden => {
            outputNeurons.forEach(output => {
                connections.push({
                    from: hidden.id,
                    to: output.id,
                    weight: Math.random() - 0.5,
                    active: false,
                    type: 'Why'
                });
            });
        });
        
        return connections;
    }
    
    /**
     * 绘制整个网络
     */
    draw() {
        const ctx = this.ctx;
        ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // 绘制背景网格
        this.drawGrid();
        
        // 绘制循环连接（最重要！）
        this.drawRecurrentConnections();
        
        // 绘制普通连接线
        this.connections.forEach(conn => this.drawConnection(conn));
        
        // 绘制神经元
        this.neurons.forEach(neuron => this.drawNeuron(neuron));
        
        // 绘制标签
        this.drawLabels();
        
        // 绘制循环箭头指示
        this.drawRecurrentIndicator();
    }
    
    /**
     * 绘制循环连接 - RNN的核心特点
     */
    drawRecurrentConnections() {
        const ctx = this.ctx;
        const hiddenNeurons = this.neurons.filter(n => n.type === 'hidden');
        
        if (hiddenNeurons.length === 0) return;
        
        // 绘制自循环箭头
        ctx.save();
        ctx.strokeStyle = '#ed64a6';
        ctx.lineWidth = 3;
        ctx.setLineDash([5, 5]);
        
        hiddenNeurons.forEach(neuron => {
            ctx.beginPath();
            // 绘制一个弧形箭头，表示循环连接
            const radius = neuron.radius + 15;
            ctx.arc(neuron.x, neuron.y, radius, -Math.PI * 0.7, -Math.PI * 0.3);
            ctx.stroke();
            
            // 绘制箭头
            const arrowX = neuron.x + radius * Math.cos(-Math.PI * 0.3);
            const arrowY = neuron.y + radius * Math.sin(-Math.PI * 0.3);
            
            ctx.beginPath();
            ctx.moveTo(arrowX, arrowY);
            ctx.lineTo(arrowX - 8, arrowY - 5);
            ctx.lineTo(arrowX - 5, arrowY + 8);
            ctx.closePath();
            ctx.fillStyle = '#ed64a6';
            ctx.fill();
        });
        
        ctx.restore();
    }
    
    /**
     * 绘制循环指示器
     */
    drawRecurrentIndicator() {
        const ctx = this.ctx;
        const hiddenNeurons = this.neurons.filter(n => n.type === 'hidden');
        
        if (hiddenNeurons.length === 0) return;
        
        // 在隐藏层上方添加文字说明
        const avgX = hiddenNeurons.reduce((sum, n) => sum + n.x, 0) / hiddenNeurons.length;
        const minY = Math.min(...hiddenNeurons.map(n => n.y));
        
        ctx.save();
        ctx.fillStyle = '#ed64a6';
        ctx.font = 'bold 14px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('⟲ 循环连接 h_{t-1} → h_t', avgX, minY - 40);
        ctx.font = '12px Arial';
        ctx.fillStyle = '#a0aec0';
        ctx.fillText('(RNN核心：记忆传递)', avgX, minY - 20);
        ctx.restore();
    }
    
    /**
     * 绘制背景网格
     */
    drawGrid() {
        const ctx = this.ctx;
        ctx.strokeStyle = 'rgba(100, 100, 100, 0.1)';
        ctx.lineWidth = 1;
        
        const gridSize = 40;
        for (let x = 0; x < this.canvas.width; x += gridSize) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, this.canvas.height);
            ctx.stroke();
        }
        
        for (let y = 0; y < this.canvas.height; y += gridSize) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(this.canvas.width, y);
            ctx.stroke();
        }
    }
    
    /**
     * 绘制连接线
     */
    drawConnection(conn) {
        const ctx = this.ctx;
        const from = this.neurons.find(n => n.id === conn.from);
        const to = this.neurons.find(n => n.id === conn.to);
        
        if (!from || !to) return;
        
        ctx.beginPath();
        
        if (conn.isRecurrent) {
            // 循环连接用曲线
            ctx.setLineDash([5, 5]);
            ctx.strokeStyle = conn.active ? '#ed64a6' : 'rgba(160, 160, 160, 0.3)';
            ctx.lineWidth = conn.active ? 3 : 1;
            
            // 绘制弧线
            const midX = (from.x + to.x) / 2;
            const midY = (from.y + to.y) / 2;
            const offset = 20;
            
            ctx.moveTo(from.x, from.y);
            ctx.quadraticCurveTo(midX + offset, midY - offset, to.x, to.y);
        } else {
            // 普通连接用直线
            ctx.setLineDash([]);
            ctx.strokeStyle = conn.active ? '#667eea' : 'rgba(160, 160, 160, 0.3)';
            ctx.lineWidth = conn.active ? 3 : 1;
            
            ctx.moveTo(from.x, from.y);
            ctx.lineTo(to.x, to.y);
        }
        
        ctx.stroke();
        ctx.setLineDash([]);
        
        // 绘制权重标签
        if (conn.active && !conn.isRecurrent) {
            const midX = (from.x + to.x) / 2;
            const midY = (from.y + to.y) / 2;
            ctx.fillStyle = '#fff';
            ctx.font = '10px Arial';
            ctx.textAlign = 'center';
            ctx.fillText(conn.weight.toFixed(2), midX, midY);
        }
    }
    
    /**
     * 绘制神经元
     */
    drawNeuron(neuron) {
        const ctx = this.ctx;
        
        // 绘制光晕
        if (neuron.active) {
            const gradient = ctx.createRadialGradient(
                neuron.x, neuron.y, neuron.radius,
                neuron.x, neuron.y, neuron.radius * 1.5
            );
            gradient.addColorStop(0, 'rgba(102, 126, 234, 0.6)');
            gradient.addColorStop(1, 'rgba(102, 126, 234, 0)');
            
            ctx.beginPath();
            ctx.arc(neuron.x, neuron.y, neuron.radius * 1.5, 0, Math.PI * 2);
            ctx.fillStyle = gradient;
            ctx.fill();
        }
        
        // 绘制神经元主体
        ctx.beginPath();
        ctx.arc(neuron.x, neuron.y, neuron.radius, 0, Math.PI * 2);
        
        // 根据类型设置颜色
        if (neuron.type === 'input') {
            ctx.fillStyle = '#4299e1';
        } else if (neuron.type === 'hidden') {
            ctx.fillStyle = '#ed64a6';
        } else {
            ctx.fillStyle = '#48bb78';
        }
        
        if (neuron.active) {
            ctx.fillStyle = neuron.type === 'input' ? '#667eea' :
                          neuron.type === 'hidden' ? '#f687b3' : '#68d391';
        }
        
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.stroke();
        
        // 绘制值
        ctx.fillStyle = '#fff';
        ctx.font = 'bold 10px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        
        // 显示值（如果有）
        const displayValue = neuron.value !== undefined && neuron.value !== 0 
            ? neuron.value.toFixed(2) 
            : '';
        ctx.fillText(displayValue, neuron.x, neuron.y);
    }
    
    /**
     * 绘制标签
     */
    drawLabels() {
        const ctx = this.ctx;
        const layers = {
            input: this.neurons.filter(n => n.type === 'input'),
            hidden: this.neurons.filter(n => n.type === 'hidden'),
            output: this.neurons.filter(n => n.type === 'output')
        };
        
        const positions = {
            input: this.canvas.width / 4,
            hidden: this.canvas.width / 2,
            output: this.canvas.width * 3 / 4
        };
        
        ctx.fillStyle = '#fff';
        ctx.font = 'bold 14px Arial';
        ctx.textAlign = 'center';
        
        for (const [layer, pos] of Object.entries(positions)) {
            const neurons = layers[layer];
            if (neurons.length > 0) {
                const y = neurons[0].y - 30;
                const label = layer === 'input' ? '输入层' :
                              layer === 'hidden' ? '隐藏层' : '输出层';
                ctx.fillText(label, pos, y);
            }
        }
    }
    
    /**
     * 更新神经元的值
     */
    updateNeuronValues(inputValues, hiddenValues, outputValues) {
        // 更新输入层
        this.neurons.filter(n => n.type === 'input').forEach((neuron, i) => {
            neuron.value = inputValues[i] || 0;
        });
        
        // 更新隐藏层
        this.neurons.filter(n => n.type === 'hidden').forEach((neuron, i) => {
            neuron.value = hiddenValues[i] || 0;
        });
        
        // 更新输出层
        this.neurons.filter(n => n.type === 'output').forEach((neuron, i) => {
            neuron.value = outputValues[i] || 0;
        });
        
        this.draw();
    }
    
    /**
     * 激活特定层的神经元
     */
    activateLayer(layerType) {
        this.neurons.forEach(neuron => {
            if (neuron.type === layerType) {
                neuron.active = true;
            } else {
                neuron.active = false;
            }
        });
        
        // 激活相关连接
        this.connections.forEach(conn => {
            const from = this.neurons.find(n => n.id === conn.from);
            const to = this.neurons.find(n => n.id === conn.to);
            conn.active = (from.type === layerType || to.type === layerType);
        });
        
        this.draw();
    }
    
    /**
     * 高亮显示所有节点
     */
    highlightAll() {
        this.neurons.forEach(neuron => neuron.active = true);
        this.connections.forEach(conn => conn.active = true);
        this.draw();
    }
    
    /**
     * 重置高亮
     */
    resetHighlight() {
        this.neurons.forEach(neuron => {
            neuron.active = false;
            neuron.value = 0;
        });
        this.connections.forEach(conn => conn.active = false);
        this.draw();
    }
    
    /**
     * 动画展示前向传播过程 - 改进版：实时更新数值
     */
    async animateForward(inputSequence, rnn, charArray) {
        let h = new Array(this.hiddenSize).fill(0);
        
        for (let t = 0; t < inputSequence.length; t++) {
            // 更新时间步标签
            this.updateTimestepDisplay(t);
            
            // 激活输入层
            this.activateLayer('input');
            
            // 获取输入字符
            const inputChar = charArray ? charArray[t] : `x${t}`;
            this.updateInputDisplay(inputChar, t);
            
            await this.sleep(this.animationSpeed);
            
            // 前向传播
            const result = rnn.forwardStep(inputSequence[t], h);
            h = result.h;
            
            // 更新Canvas中的值
            this.updateNeuronValues(inputSequence[t], h, result.y);
            
            // 更新时间步展开视图中的数值
            this.updateTimestepValues(t, inputChar, h, result.y, charArray);
            
            // 激活隐藏层
            this.activateLayer('hidden');
            await this.sleep(this.animationSpeed);
            
            // 激活输出层
            this.activateLayer('output');
            await this.sleep(this.animationSpeed);
            
            // 重置
            this.resetHighlight();
        }
    }
    
    /**
     * 更新输入显示
     */
    updateInputDisplay(char, t) {
        const display = document.getElementById('current-step');
        if (display) {
            display.innerHTML = `处理时间步 t=${t} | 输入: <strong>"${char}"</strong>`;
            display.style.color = '#667eea';
        }
    }
    
    /**
     * 更新时间步展开视图中的数值显示 - 新增方法
     */
    updateTimestepValues(t, inputChar, hiddenState, output, charArray) {
        const timestepBox = document.querySelector(`#timestep-${t}`);
        if (!timestepBox) return;
        
        // 获取神经元元素
        const neurons = timestepBox.querySelectorAll('.neuron');
        if (neurons.length < 3) return;
        
        // 更新输入值显示
        const inputNeuron = neurons[0];
        const inputDisplay = inputNeuron.querySelector('.value-display');
        if (inputDisplay) {
            inputDisplay.textContent = `"${inputChar}"`;
            inputDisplay.style.color = '#4299e1';
        }
        
        // 更新隐藏状态值显示（显示平均值）
        const hiddenNeuron = neurons[1];
        const hiddenDisplay = hiddenNeuron.querySelector('.value-display');
        if (hiddenDisplay) {
            const avgH = hiddenState.reduce((a, b) => a + Math.abs(b), 0) / hiddenState.length;
            hiddenDisplay.textContent = avgH.toFixed(3);
            hiddenDisplay.style.color = '#ed64a6';
        }
        
        // 更新输出值显示（显示预测字符）
        const outputNeuron = neurons[2];
        const outputDisplay = outputNeuron.querySelector('.value-display');
        if (outputDisplay) {
            // 找到最大概率的输出
            let maxIdx = 0;
            let maxProb = output[0];
            for (let i = 1; i < output.length; i++) {
                if (output[i] > maxProb) {
                    maxProb = output[i];
                    maxIdx = i;
                }
            }
            
            if (charArray && charArray[maxIdx]) {
                outputDisplay.textContent = `"${charArray[maxIdx]}" (${(maxProb * 100).toFixed(1)}%)`;
            } else {
                outputDisplay.textContent = `${(maxProb * 100).toFixed(1)}%`;
            }
            outputDisplay.style.color = '#48bb78';
        }
        
        // 高亮当前时间步
        timestepBox.classList.add('active');
        neurons.forEach(n => n.classList.add('active'));
    }
    
    /**
     * 更新时间步显示
     */
    updateTimestepDisplay(t) {
        // 高亮对应的时间步
        const timesteps = document.querySelectorAll('.timestep-box');
        timesteps.forEach((ts, i) => {
            if (i === t) {
                ts.classList.add('active');
                ts.querySelectorAll('.neuron').forEach(n => n.classList.add('active'));
            } else if (i < t) {
                // 之前的时间步保持已处理状态
                ts.classList.remove('active');
                ts.classList.add('processed');
            } else {
                ts.classList.remove('active');
                ts.classList.remove('processed');
                ts.querySelectorAll('.neuron').forEach(n => n.classList.remove('active'));
            }
        });
    }
    
    /**
     * 延迟函数
     */
    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// 导出
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { RNNVisualizer };
}
