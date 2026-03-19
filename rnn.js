/**
 * RNN (循环神经网络) 核心实现
 * 简化版本，用于教学演示
 */

class RNN {
    constructor(inputSize, hiddenSize, outputSize) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        
        // 初始化权重（使用 Xavier 初始化）
        this.Wxh = this.xavierInit(hiddenSize, inputSize);  // 输入到隐藏层
        this.Whh = this.xavierInit(hiddenSize, hiddenSize); // 隐藏层到隐藏层（循环连接）
        this.Why = this.xavierInit(outputSize, hiddenSize); // 隐藏层到输出
        
        // 偏置
        this.bh = new Array(hiddenSize).fill(0);
        this.by = new Array(outputSize).fill(0);
        
        // 用于存储中间状态（反向传播需要）
        this.hiddenStates = [];
        this.inputs = [];
        this.outputs = [];
        
        // 学习率
        this.learningRate = 0.01;
    }
    
    /**
     * Xavier 初始化
     */
    xavierInit(rows, cols) {
        const scale = Math.sqrt(2.0 / (rows + cols));
        const matrix = [];
        for (let i = 0; i < rows; i++) {
            matrix[i] = [];
            for (let j = 0; j < cols; j++) {
                matrix[i][j] = (Math.random() * 2 - 1) * scale;
            }
        }
        return matrix;
    }
    
    /**
     * 向量矩阵乘法
     */
    matMul(matrix, vector) {
        const result = [];
        for (let i = 0; i < matrix.length; i++) {
            let sum = 0;
            for (let j = 0; j < vector.length; j++) {
                sum += matrix[i][j] * vector[j];
            }
            result.push(sum);
        }
        return result;
    }
    
    /**
     * tanh 激活函数
     */
    tanh(x) {
        return Math.tanh(x);
    }
    
    /**
     * tanh 导数
     */
    tanhDerivative(x) {
        return 1 - x * x;
    }
    
    /**
     * softmax 函数
     */
    softmax(arr) {
        const maxVal = Math.max(...arr);
        const expArr = arr.map(x => Math.exp(x - maxVal));
        const sum = expArr.reduce((a, b) => a + b, 0);
        return expArr.map(x => x / sum);
    }
    
    /**
     * 前向传播 - 单个时间步
     */
    forwardStep(x, hPrev) {
        // 计算隐藏状态
        // h = tanh(Wxh * x + Whh * h_prev + bh)
        const inputPart = this.matMul(this.Wxh, x);
        const hiddenPart = this.matMul(this.Whh, hPrev);
        
        const h = [];
        for (let i = 0; i < this.hiddenSize; i++) {
            h[i] = this.tanh(inputPart[i] + hiddenPart[i] + this.bh[i]);
        }
        
        // 计算输出
        // y = softmax(Why * h + by)
        const outputRaw = this.matMul(this.Why, h);
        for (let i = 0; i < this.outputSize; i++) {
            outputRaw[i] += this.by[i];
        }
        const y = this.softmax(outputRaw);
        
        return { h, y };
    }
    
    /**
     * 前向传播 - 完整序列
     */
    forward(inputs) {
        this.inputs = inputs;
        this.hiddenStates = [];
        this.outputs = [];
        
        // 初始隐藏状态
        let h = new Array(this.hiddenSize).fill(0);
        this.hiddenStates.push([...h]);
        
        // 对每个时间步进行前向传播
        for (let t = 0; t < inputs.length; t++) {
            const result = this.forwardStep(inputs[t], h);
            h = result.h;
            this.hiddenStates.push([...h]);
            this.outputs.push(result.y);
        }
        
        return this.outputs;
    }
    
    /**
     * 交叉熵损失
     */
    crossEntropyLoss(predicted, target) {
        let loss = 0;
        for (let i = 0; i < predicted.length; i++) {
            // 避免log(0)
            const p = Math.max(predicted[i], 1e-10);
            loss -= target[i] * Math.log(p);
        }
        return loss;
    }
    
    /**
     * 计算总损失
     */
    computeLoss(targets) {
        let totalLoss = 0;
        for (let t = 0; t < this.outputs.length; t++) {
            totalLoss += this.crossEntropyLoss(this.outputs[t], targets[t]);
        }
        return totalLoss / this.outputs.length;
    }
    
    /**
     * 反向传播通过时间 (BPTT)
     */
    backward(targets) {
        const seqLen = this.inputs.length;
        
        // 初始化梯度
        const dWxh = this.zerosLike(this.Wxh);
        const dWhh = this.zerosLike(this.Whh);
        const dWhy = this.zerosLike(this.Why);
        const dbh = new Array(this.hiddenSize).fill(0);
        const dby = new Array(this.outputSize).fill(0);
        
        // 初始化 dh_next（下一时刻传回的梯度）
        let dhNext = new Array(this.hiddenSize).fill(0);
        
        // 反向传播通过时间
        for (let t = seqLen - 1; t >= 0; t--) {
            // 输出层梯度
            const dy = [...this.outputs[t]];
            for (let i = 0; i < this.outputSize; i++) {
                dy[i] -= targets[t][i]; // softmax + cross-entropy 的简化梯度
            }
            
            // Why 和 by 的梯度
            for (let i = 0; i < this.outputSize; i++) {
                for (let j = 0; j < this.hiddenSize; j++) {
                    dWhy[i][j] += dy[i] * this.hiddenStates[t + 1][j];
                }
                dby[i] += dy[i];
            }
            
            // 隐藏层梯度
            const h = this.hiddenStates[t + 1];
            const dh = new Array(this.hiddenSize).fill(0);
            
            for (let i = 0; i < this.hiddenSize; i++) {
                // 来自输出层的梯度
                let gradient = 0;
                for (let j = 0; j < this.outputSize; j++) {
                    gradient += this.Why[j][i] * dy[j];
                }
                // 来自下一时间步的梯度
                gradient += dhNext[i];
                
                // tanh 导数
                dh[i] = gradient * this.tanhDerivative(h[i]);
            }
            
            // Wxh, Whh, bh 的梯度
            for (let i = 0; i < this.hiddenSize; i++) {
                // bh
                dbh[i] += dh[i];
                
                // Wxh
                for (let j = 0; j < this.inputSize; j++) {
                    dWxh[i][j] += dh[i] * this.inputs[t][j];
                }
                
                // Whh
                for (let j = 0; j < this.hiddenSize; j++) {
                    dWhh[i][j] += dh[i] * this.hiddenStates[t][j];
                }
            }
            
            // 传递梯度到上一时间步
            dhNext = [];
            for (let i = 0; i < this.hiddenSize; i++) {
                let sum = 0;
                for (let j = 0; j < this.hiddenSize; j++) {
                    sum += this.Whh[j][i] * dh[j];
                }
                dhNext.push(sum);
            }
        }
        
        // 梯度裁剪（防止梯度爆炸）
        this.clipGradient(dWxh);
        this.clipGradient(dWhh);
        this.clipGradient(dWhy);
        
        // 更新权重
        this.updateWeights(dWxh, dWhh, dWhy, dbh, dby);
    }
    
    /**
     * 创建零矩阵
     */
    zerosLike(matrix) {
        return matrix.map(row => row.map(() => 0));
    }
    
    /**
     * 梯度裁剪
     */
    clipGradient(matrix, maxNorm = 5) {
        let norm = 0;
        for (const row of matrix) {
            for (const val of row) {
                norm += val * val;
            }
        }
        norm = Math.sqrt(norm);
        
        if (norm > maxNorm) {
            const scale = maxNorm / norm;
            for (const row of matrix) {
                for (let i = 0; i < row.length; i++) {
                    row[i] *= scale;
                }
            }
        }
    }
    
    /**
     * 更新权重（SGD）
     */
    updateWeights(dWxh, dWhh, dWhy, dbh, dby) {
        const lr = this.learningRate;
        
        // 更新 Wxh
        for (let i = 0; i < this.Wxh.length; i++) {
            for (let j = 0; j < this.Wxh[i].length; j++) {
                this.Wxh[i][j] -= lr * dWxh[i][j];
            }
        }
        
        // 更新 Whh
        for (let i = 0; i < this.Whh.length; i++) {
            for (let j = 0; j < this.Whh[i].length; j++) {
                this.Whh[i][j] -= lr * dWhh[i][j];
            }
        }
        
        // 更新 Why
        for (let i = 0; i < this.Why.length; i++) {
            for (let j = 0; j < this.Why[i].length; j++) {
                this.Why[i][j] -= lr * dWhy[i][j];
            }
        }
        
        // 更新偏置
        for (let i = 0; i < this.hiddenSize; i++) {
            this.bh[i] -= lr * dbh[i];
        }
        for (let i = 0; i < this.outputSize; i++) {
            this.by[i] -= lr * dby[i];
        }
    }
    
    /**
     * 训练单个epoch
     */
    trainStep(inputs, targets) {
        // 前向传播
        this.forward(inputs);
        
        // 计算损失
        const loss = this.computeLoss(targets);
        
        // 反向传播
        this.backward(targets);
        
        return loss;
    }
    
    /**
     * 获取当前隐藏状态
     */
    getHiddenState(t) {
        return this.hiddenStates[t] || [];
    }
    
    /**
     * 获取输出
     */
    getOutput(t) {
        return this.outputs[t] || [];
    }
    
    /**
     * 获取预测的字符索引
     */
    predict(output) {
        let maxIdx = 0;
        let maxVal = output[0];
        for (let i = 1; i < output.length; i++) {
            if (output[i] > maxVal) {
                maxVal = output[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }
}

/**
 * 字符级RNN - 用于文本生成任务
 */
class CharRNN {
    constructor(chars, hiddenSize) {
        this.chars = chars;
        this.charToIndex = {};
        this.indexToChar = {};
        
        // 建立字符到索引的映射
        chars.forEach((char, i) => {
            this.charToIndex[char] = i;
            this.indexToChar[i] = char;
        });
        
        const vocabSize = chars.length;
        this.rnn = new RNN(vocabSize, hiddenSize, vocabSize);
    }
    
    /**
     * 将字符转换为 one-hot 向量
     */
    charToVector(char) {
        const vector = new Array(this.chars.length).fill(0);
        const idx = this.charToIndex[char];
        if (idx !== undefined) {
            vector[idx] = 1;
        }
        return vector;
    }
    
    /**
     * 将索引转换为字符
     */
    indexToChar_(idx) {
        const char = this.indexToChar[idx];
        // 如果找不到字符，返回第一个字符
        return char !== undefined ? char : (this.chars[0] || '');
    }
    
    /**
     * 设置学习率
     */
    setLearningRate(lr) {
        this.rnn.learningRate = lr;
    }
    
    /**
     * 训练
     */
    train(text, epochs, onProgress) {
        const inputs = [];
        const targets = [];
        
        // 准备训练数据
        for (let i = 0; i < text.length - 1; i++) {
            inputs.push(this.charToVector(text[i]));
            targets.push(this.charToVector(text[i + 1]));
        }
        
        let epoch = 0;
        const trainLoop = () => {
            if (epoch >= epochs) {
                if (onProgress) onProgress(epochs, epochs, 0, true);
                return;
            }
            
            const loss = this.rnn.trainStep(inputs, targets);
            epoch++;
            
            if (onProgress) {
                onProgress(epoch, epochs, loss, false);
            }
            
            // 继续训练
            setTimeout(trainLoop, 10);
        };
        
        trainLoop();
    }
    
    /**
     * 生成文本
     */
    generate(seedChar, length) {
        let result = seedChar;
        let h = new Array(this.rnn.hiddenSize).fill(0);
        let currentInput = this.charToVector(seedChar);
        
        for (let i = 0; i < length; i++) {
            const { h: newH, y } = this.rnn.forwardStep(currentInput, h);
            h = newH;
            
            // 从输出分布中采样
            const idx = this.sample(y);
            const char = this.indexToChar_(idx);
            
            // 确保字符有效
            if (char && char !== '') {
                result += char;
                currentInput = this.charToVector(char);
            } else {
                // 如果无效，使用第一个字符
                const defaultChar = this.chars[0] || '';
                result += defaultChar;
                currentInput = this.charToVector(defaultChar);
            }
        }
        
        return result;
    }
    
    /**
     * 预测下一个字符
     */
    predictNext(sequence) {
        let h = new Array(this.rnn.hiddenSize).fill(0);
        
        // 处理整个序列
        for (const char of sequence) {
            const x = this.charToVector(char);
            const result = this.rnn.forwardStep(x, h);
            h = result.h;
        }
        
        // 获取最后一个输出
        const lastChar = sequence[sequence.length - 1];
        const x = this.charToVector(lastChar);
        const { y } = this.rnn.forwardStep(x, h);
        
        // 返回概率最高的几个字符
        const predictions = [];
        for (let i = 0; i < y.length; i++) {
            const char = this.indexToChar_[i];
            predictions.push({
                char: char,
                probability: y[i]
            });
        }
        
        predictions.sort((a, b) => b.probability - a.probability);
        return predictions.slice(0, 5);
    }
    
    /**
     * 从概率分布中采样
     */
    sample(probs) {
        const r = Math.random();
        let cumulative = 0;
        for (let i = 0; i < probs.length; i++) {
            cumulative += probs[i];
            if (r < cumulative) {
                return i;
            }
        }
        return probs.length - 1;
    }
    
    /**
     * 获取RNN实例
     */
    getRNN() {
        return this.rnn;
    }
}

// 导出
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { RNN, CharRNN };
}
