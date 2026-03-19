/**
 * 中文训练数据集
 * 提供多种中文文本数据用于RNN训练
 */

const DATASETS = {
    'tang-poetry': {
        name: '唐诗选集',
        description: '经典唐诗，学习古诗词韵律',
        data: `
床前明月光疑是地上霜举头望明月低头思故乡
春眠不觉晓处处闻啼鸟夜来风雨声花落知多少
白日依山尽黄河入海流欲穷千里目更上一层楼
锄禾日当午汗滴禾下土谁知盘中餐粒粒皆辛苦
离离原上草一岁一枯荣野火烧不尽春风吹又生
空山不见人但闻人语响返景入深林复照青苔上
千山鸟飞绝万径人踪灭孤舟蓑笠翁独钓寒江雪
移舟泊烟渚日暮客愁新野旷天低树江清月近人
月落乌啼霜满天江枫渔火对愁眠姑苏城外寒山寺夜半钟声到客船
朝辞白帝彩云间千里江陵一日还两岸猿声啼不住轻舟已过万重山
日照香炉生紫烟遥看瀑布挂前川飞流直下三千尺疑是银河落九天
两个黄鹂鸣翠柳一行白鹭上青天窗含西岭千秋雪门泊东吴万里船
独在异乡为异客每逢佳节倍思亲遥知兄弟登高处遍插茱萸少一人
葡萄美酒夜光杯欲饮琵琶马上催醉卧沙场君莫笑古来征战几人回
秦时明月汉时关万里长征人未还但使龙城飞将在不教胡马度阴山
渭城朝雨浥轻尘客舍青青柳色新劝君更尽一杯酒西出阳关无故人
寒雨连江夜入吴平明送客楚山孤洛阳亲友如相问一片冰心在玉壶
        `.trim()
    },
    
    'chinese-proverbs': {
        name: '中国成语',
        description: '常用成语，学习四字结构',
        data: `
一心一意三心二意七上八下十全十美
百发百中千方百计万众一心
人山人海自由自在无忧无虑各种各样
五花八门五光十色五颜六色五彩缤纷
风吹雨打风吹草动风吹日晒风和日丽
山清水秀山高水长山穷水尽山崩地裂
天长地久天高地厚天翻地覆天罗地网
风调雨顺风平浪静风雨同舟风雨无阻
春暖花开春意盎然春光明媚春风拂面
秋高气爽秋风习习秋色宜人一叶知秋
冰天雪地白雪皑皑银装素裹粉妆玉砌
车水马龙川流不息人来人往络绎不绝
高楼大厦亭台楼阁富丽堂皇雕梁画栋
百花齐放百家争鸣百折不挠百战百胜
千变万化千姿百态千奇百怪千钧一发
亡羊补牢守株待兔掩耳盗铃自相矛盾
刻舟求剑叶公好龙画蛇添足对牛弹琴
井底之蛙杯弓蛇影狐假虎威鹬蚌相争
画龙点睛望梅止渴精卫填海愚公移山
夸父追日女娲补天后羿射日大禹治水
        `.trim()
    },
    
    'ancient-wisdom': {
        name: '古文名句',
        description: '经典古文名句，学习文言文',
        data: `
学而时习之不亦说乎有朋自远方来不亦乐乎人不知而不愠不亦君子乎
三人行必有我师焉择其善者而从之其不善者而改之
学而不思则罔思而不学则殆
知之为知之不知为不知是知也
敏而好学不耻下问
温故而知新可以为师矣
己所不欲勿施于人
知者不惑仁者不忧勇者不惧
德不孤必有邻
君子坦荡荡小人长戚戚
四海之内皆兄弟也
有朋自远方来不亦乐乎
工欲善其事必先利其器
人无远虑必有近忧
小不忍则乱大谋
过而不改是谓过矣
君子成人之美不成人之恶小人反是
其身正不令而行其身不正虽令不从
岁寒然后知松柏之后凋也
天将降大任于斯人也必先苦其心志劳其筋骨饿其体肤空乏其身行拂乱其所为
生于忧患死于安乐
得道者多助失道者寡助
天时不如地利地利不如人和
老吾老以及人之老幼吾幼以及人之幼
富贵不能淫贫贱不能移威武不能屈
        `.trim()
    },
    
    'modern-chinese': {
        name: '现代中文',
        description: '现代中文常用语句',
        data: `
今天天气真不错阳光明媚微风习习
学习使人进步知识改变命运
书籍是人类进步的阶梯
时间就是金钱效率就是生命
团结就是力量坚持就是胜利
有志者事竟成破釜沉舟百二秦关终属楚
苦心人天不负卧薪尝胆三千越甲可吞吴
失败是成功之母
世上无难事只怕有心人
勤能补拙是良训一分辛苦一分才
天才就是百分之九十九的汗水加百分之一的灵感
机会总是留给有准备的人
行动是成功的阶梯行动越多登得越高
成功的关键在于相信自己有成功的能力
态度决定一切细节决定成败
没有做不到只有想不到
千里之行始于足下
水滴石穿绳锯木断
铁杵磨成针功到自然成
世上无难事只要肯攀登
宝剑锋从磨砺出梅花香自苦寒来
        `.trim()
    },
    
    'custom': {
        name: '自定义文本',
        description: '用户自定义训练文本',
        data: ''
    }
};

/**
 * 数据集管理器
 */
class DatasetManager {
    constructor() {
        this.currentDataset = 'tang-poetry';
        this.listeners = [];
    }
    
    /**
     * 获取数据集列表
     */
    getDatasetList() {
        return Object.keys(DATASETS).map(key => ({
            id: key,
            name: DATASETS[key].name,
            description: DATASETS[key].description
        }));
    }
    
    /**
     * 获取当前数据集
     */
    getCurrentDataset() {
        return DATASETS[this.currentDataset];
    }
    
    /**
     * 设置当前数据集
     */
    setDataset(datasetId) {
        if (DATASETS[datasetId]) {
            this.currentDataset = datasetId;
            this.notifyListeners();
            return true;
        }
        return false;
    }
    
    /**
     * 获取训练数据
     */
    getTrainingData() {
        const dataset = DATASETS[this.currentDataset];
        return dataset ? dataset.data : '';
    }
    
    /**
     * 设置自定义数据
     */
    setCustomData(text) {
        DATASETS['custom'].data = text;
        if (this.currentDataset === 'custom') {
            this.notifyListeners();
        }
    }
    
    /**
     * 获取数据统计
     */
    getDataStats(text) {
        const data = text || this.getTrainingData();
        const chars = [...data];
        const uniqueChars = new Set(chars);
        
        return {
            totalChars: chars.length,
            uniqueChars: uniqueChars.size,
            charSet: Array.from(uniqueChars)
        };
    }
    
    /**
     * 添加监听器
     */
    addListener(callback) {
        this.listeners.push(callback);
    }
    
    /**
     * 通知监听器
     */
    notifyListeners() {
        this.listeners.forEach(callback => callback(this.currentDataset));
    }
}

// 导出
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { DATASETS, DatasetManager };
}
