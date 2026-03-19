/**
 * RNN 可视化模块
 * 处理RNN结构、数据流动画等可视化功能
 */

class RNNVisualizer {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.neurons = [];
        this.connections = [];
        this.currentTimestep = -1;
        this.animationSpeed = 500; // ms
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
        
        // 绘制连接线
        this.connections.forEach(conn => this.drawConnection(conn));
        
        // 绘制神经元
        this.neurons.forEach(neuron => this.drawNeuron(neuron));
        
        // 绘制标签
        this.drawLabels();
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
        ctx.fillText(neuron.value.toFixed(1), neuron.x, neuron.y);
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
        this.neurons.forEach(neuron => neuron.active = false);
        this.connections.forEach(conn => conn.active = false);
        this.draw();
    }
    
    /**
     * 动画展示前向传播过程
     */
    async animateForward(inputSequence, rnn) {
        let h = new Array(this.hiddenSize).fill(0);
        
        for (let t = 0; t < inputSequence.length; t++) {
            // 更新时间步标签
            this.updateTimestepDisplay(t);
            
            // 激活输入层
            this.activateLayer('input');
            await this.sleep(this.animationSpeed);
            
            // 前向传播
            const result = rnn.forwardStep(inputSequence[t], h);
            h = result.h;
            
            // 更新值
            this.updateNeuronValues(inputSequence[t], h, result.y);
            
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
     * 更新时间步显示
     */
    updateTimestepDisplay(t) {
        const display = document.getElementById('current-step');
        if (display) {
            display.textContent = `处理时间步 t=${t}`;
            display.style.color = '#667eea';
        }
        
        // 高亮对应的时间步
        const timesteps = document.querySelectorAll('.timestep');
        timesteps.forEach((ts, i) => {
            if (i === t) {
                ts.classList.add('active');
                ts.querySelectorAll('.neuron').forEach(n => n.classList.add('active'));
            } else {
                ts.classList.remove('active');
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
    
    /**
     * 动画展示数据流
     */
    async animateDataFlow(fromType, toType) {
        const fromNeurons = this.neurons.filter(n => n.type === fromType);
        const toNeurons = this.neurons.filter(n => n.type === toType);
        
        // 激活连接
        this.connections.forEach(conn => {
            const from = this.neurons.find(n => n.id === conn.from);
            const to = this.neurons.find(n => n.id === conn.to);
            if (from.type === fromType && to.type === toType) {
                conn.active = true;
            }
        });
        
        this.draw();
        
        // 激活目标层
        await this.sleep(this.animationSpeed / 2);
        this.activateLayer(toType);
        
        await this.sleep(this.animationSpeed);
        this.resetHighlight();
    }
}

/**
 * 展开式可视化 - 显示多个时间步
 */
class UnrolledVisualizer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.timesteps = [];
    }
    
    /**
     * 创建时间步节点
     */
    createTimesteps(numSteps) {
        this.container.innerHTML = '';
        this.timesteps = [];
        
        for (let t = 0; t < numSteps; t++) {
            const step = document.createElement('div');
            step.className = 'timestep';
            step.id = `timestep-${t}`;
            
            step.innerHTML = `
                <div class="timestep-label">t=${t}</div>
                <div class="neuron input-neuron">
                    <span class="label">x${this.toSubscript(t)}</span>
                </div>
                <div class="neuron hidden-neuron">
                    <span class="label">h${this.toSubscript(t)}</span>
                </div>
                <div class="neuron output-neuron">
                    <span class="label">y${this.toSubscript(t)}</span>
                </div>
            `;
            
            this.container.appendChild(step);
            this.timesteps.push(step);
        }
        
        // 添加连接线（使用CSS或SVG）
        this.addConnections();
    }
    
    /**
     * 添加连接线
     */
    addConnections() {
        // 这里可以添加SVG连接线来展示时间步之间的连接
        // 简化起见，使用CSS的border或::after伪元素
        this.timesteps.forEach((step, i) => {
            if (i > 0) {
                step.style.borderLeft = '2px dashed rgba(255, 255, 255, 0.2)';
            }
        });
    }
    
    /**
     * 转换为下标
     */
    toSubscript(num) {
        const subscripts = ['₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉'];
        return num.toString().split('').map(d => subscripts[parseInt(d)]).join('');
    }
    
    /**
     * 激活特定时间步
     */
    activateTimestep(t) {
        this.timesteps.forEach((step, i) => {
            if (i === t) {
                step.classList.add('active');
                step.querySelectorAll('.neuron').forEach(n => n.classList.add('active'));
            } else {
                step.classList.remove('active');
                step.querySelectorAll('.neuron').forEach(n => n.classList.remove('active'));
            }
        });
    }
    
    /**
     * 更新神经元值
     */
    updateValues(input, hidden, output, t) {
        const step = this.timesteps[t];
        if (step) {
            const neurons = step.querySelectorAll('.neuron');
            neurons[0].innerHTML = `x${this.toSubscript(t)}: ${input.toFixed(2)}`;
            neurons[1].innerHTML = `h${this.toSubscript(t)}: ${hidden.toFixed(2)}`;
            neurons[2].innerHTML = `y${this.toSubscript(t)}: ${output.toFixed(2)}`;
        }
    }
    
    /**
     * 重置所有时间步
     */
    reset() {
        this.timesteps.forEach(step => {
            step.classList.remove('active');
            step.querySelectorAll('.neuron').forEach(n => n.classList.remove('active'));
        });
    }
}

/**
 * 公式可视化 - 动态显示公式和计算过程
 */
class FormulaVisualizer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
    }
    
    /**
     * 显示前向传播公式
     */
    showForwardFormula(t) {
        const formula = `
            <div class="formula-step">
                <strong>时间步 t = ${t}:</strong><br>
                h<sub>${t}</sub> = tanh(W<sub>xh</sub> · x<sub>${t}</sub> + W<sub>hh</sub> · h<sub>${t-1}</sub> + b<sub>h</sub>)<br>
                y<sub>${t}</sub> = softmax(W<sub>hy</sub> · h<sub>${t}</sub> + b<sub>y</sub>)
            </div>
        `;
        this.container.innerHTML = formula;
    }
    
    /**
     * 显示反向传播公式
     */
    showBackwardFormula(t) {
        const formula = `
            <div class="formula-step">
                <strong>反向传播 t = ${t}:</strong><br>
                ∂L/∂h<sub>${t}</sub> = ∂L/∂y<sub>${t}</sub> · W<sub>hy</sub><sup>T</sup> + ∂L/∂h<sub>${t+1}</sub> · W<sub>hh</sub><sup>T</sup><br>
                W<sub>xh</sub> := W<sub>xh</sub> - η · ∂L/∂W<sub>xh</sub><br>
                W<sub>hh</sub> := W<sub>hh</sub> - η · ∂L/∂W<sub>hh</sub><br>
                W<sub>hy</sub> := W<sub>hy</sub> - η · ∂L/∂W<sub>hy</sub>
            </div>
        `;
        this.container.innerHTML = formula;
    }
    
    /**
     * 清空公式
     */
    clear() {
        this.container.innerHTML = '';
    }
}

// 导出
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { RNNVisualizer, UnrolledVisualizer, FormulaVisualizer };
}
