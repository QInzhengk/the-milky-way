import { defineConfig } from 'vitepress'

// https://vitepress.dev/reference/site-config
export default defineConfig({
    title: "数学建模与人工智能",
    description: "专注于数学建模、人工智能和计算机科学的技术博客",
    lang: 'zh-CN',
    
    // GitHub Pages部署路径
    base: '/the-milky-way/',
    
    // 忽略死链接检查
    ignoreDeadLinks: true,
    
    // 页面元数据
    head: [
      ['link', { rel: 'icon', href: '/favicon.ico' }],
      ['meta', { name: 'keywords', content: '数学建模,人工智能,机器学习,深度学习,算法,编程' }],
      ['meta', { name: 'author', content: 'Qin' }]
    ],

  markdown: {
    theme: {
      light: 'github-light',
      dark: 'github-dark'
    },
    lineNumbers: true,
    container: {
      tipLabel: '提示',
      warningLabel: '警告',
      dangerLabel: '危险',
      infoLabel: '信息',
      detailsLabel: '详细信息'
    }
  },

  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    nav: [
      { text: '🏠 首页', link: '/' },
      { text: '🏆 数学建模', 
        items: [
          { text: '🎯 数学竞赛', link: '/mathematical-modeling/math-competition/' },
          { text: '🎓 大学生数学建模', link: '/mathematical-modeling/undergraduate-modeling/' },
          { text: '🏅 研究生数学建模', link: '/mathematical-modeling/graduate-modeling/' }
        ]
      },
      { text: '🤖 人工智能', 
        items: [
          { text: '🧠 机器学习', link: '/artificial-intelligence/machine-learning/' },
          { text: '🚀 深度学习', link: '/artificial-intelligence/deep-learning/' },
          { text: '🌟 大模型', link: '/artificial-intelligence/large-models/' }
        ]
      },
      { text: '💻 计算机科学', 
        items: [
          { text: '📚 编程技术', link: '/computer-science/programming/' },
          { text: '🔍 算法', link: '/computer-science/algorithms/' },
          { text: '📊 数据结构', link: '/computer-science/data-structures/' }
        ]
      },
      { text: '🔗 相关链接', 
        items: [
          { text: '💻 CSDN', link: 'https://blog.csdn.net/qq_45832050?type=blog' },
          { text: '📱 知乎', link: 'https://www.zhihu.com/signin?next=%2F' },
          { text: '📢 微信公众号', link: 'https://mp.weixin.qq.com/s/pTar--ikPGql7HINNycCRg' }
        ]
      }
    ],

    sidebar: {
      '/mathematical-modeling/': [
        {
          text: '🏆 数学建模竞赛实战',
          items: [
            {
              text: '🎓 大学生建模竞赛',
              items: [
                { text: '🥉 MathorCup 2021 B题：三维团簇能量预测', link: '/mathematical-modeling/undergraduate-modeling/2021年MathorCup高校数学建模挑战赛b题：三维团簇的能量预测（三等）.md' }
              ]
            },
            {
              text: '🏅 研究生建模竞赛',
              items: [
                { text: '🥇 华为杯第十八届：抗乳腺癌药物优化建模（一等奖）', link: '/mathematical-modeling/graduate-modeling/华为杯第十八届中国研究生数学建模竞赛D题：抗乳腺癌候选药物的优化建模(一等奖）.md' },
                { text: '📝 河北省第二、三届研究生建模试题', link: '/mathematical-modeling/graduate-modeling/河北省第二、三届研究生数学建模试题.md' },
                { text: '🥈 河北省第三届：交通检测器数据控制预测（二等奖）', link: '/mathematical-modeling/graduate-modeling/河北省第三届研究生数学建模B题（二等）交通检测器数据质量控制及预测.md' }
              ]
            }
          ]
        }
      ],
      '/artificial-intelligence/': [
        {
          text: '人工智能',
          items: [
            {
              text: '🧠 机器学习算法',
              items: [
                { text: '🎯 超参数调优：网格搜索与贝叶斯优化', link: '/artificial-intelligence/machine-learning/超参数调优：网格搜索，贝叶斯优化（optuna）详解.md' },
                { text: '📊 NGBoost概率预测与分位数回归', link: '/artificial-intelligence/machine-learning/概率预测之NGBoost（Natural Gradient Boosting）回归和分位数（Quantile Regression）回归.md' },
                { text: '📈 图像数据处理技术', link: '/artificial-intelligence/machine-learning/机器学习笔试面试之图像数据不足时的处理方法、检验方法、不均衡样本集的重采样、数据集分布是否一致.md' },
                { text: '🔧 特征工程与模型评估完整指南', link: '/artificial-intelligence/machine-learning/机器学习面试笔试之特征工程、优化方法、降维、模型评估.md' },
                { text: '🔗 贝叶斯网络与马尔科夫模型', link: '/artificial-intelligence/machine-learning/机器学习面试笔试知识点-贝叶斯网络(Bayesian Network) 、马尔科夫(Markov) 和主题模型(T M).md' },
                { text: '🌳 决策树与集成学习全解', link: '/artificial-intelligence/machine-learning/机器学习面试笔试知识点-决策树、随机森林、梯度提升决策树(GBDT)、XGBoost、LightGBM、CatBoost.md' },
                { text: '📐 线性回归、逻辑回归和SVM', link: '/artificial-intelligence/machine-learning/机器学习面试笔试知识点-线性回归、逻辑回归(Logistics Regression)和支持向量机(SVM).md' },
                { text: '🎲 非监督学习：聚类与降维', link: '/artificial-intelligence/machine-learning/机器学习面试笔试知识点之非监督学习-K 均值聚类、高斯混合模型（GMM）、自组织映射神经网络（SOM）.md' },
                { text: '🔍 K近邻与EM算法', link: '/artificial-intelligence/machine-learning/机器学习面试笔试知识点之K近邻算法(KNN)、最大期望算法(EM).md' },
                { text: 'TabNet详解', link: '/artificial-intelligence/machine-learning/Tabnet介绍（Decision Manifolds）和PyTorch TabNet之TabNetRegressor.md' },
                { text: 'XGBoost参数详解', link: '/artificial-intelligence/machine-learning/XGBoost原生接口和Sklearn接口参数详解.md' }
              ]
            },
            {
              text: '深度学习',
              items: [
                { text: 'PyTorch张量基础', link: '/artificial-intelligence/deep-learning/深度学习Pytorch框架Tensor张量.md' },
                { text: 'PyTorch张量属性与运算', link: '/artificial-intelligence/deep-learning/深度学习Pytorch-Tensor的属性、算术运算.md' },
                { text: 'PyTorch张量函数', link: '/artificial-intelligence/deep-learning/深度学习Pytorch-Tensor函数.md' },
                { text: 'PyTorch核心模块详解', link: '/artificial-intelligence/deep-learning/Pytorch详解-Pytorch核心模块.md' },
                { text: 'PyTorch数据模块详解', link: '/artificial-intelligence/deep-learning/Pytorch详解-数据模块.md' },
                { text: 'PyTorch模型模块详解', link: '/artificial-intelligence/deep-learning/Pytorch详解-模型模块(RNN,CNN,FNN,LSTM,GRU,TCN,Transformer).md' },
                { text: 'PyTorch优化模块详解', link: '/artificial-intelligence/deep-learning/PyTorch详解-优化模块.md' },
                { text: 'PyTorch可视化模块详解', link: '/artificial-intelligence/deep-learning/PyTorch详解-可视化模块.md' },
                { text: 'PyTorch模型保存与加载', link: '/artificial-intelligence/deep-learning/Pytorch详解-模型保存与加载、Finetune 模型微调、GPU使用、nvidia-smi详解、TorchEnsemble 模型集成库、torchmetrics 模型评估指标库.md' },
                { text: 'PyTorch torch.nn库', link: '/artificial-intelligence/deep-learning/Pytorch torch.nn库以及nn与nn.functional有什么区别？.md' },
                { text: 'PyTorch与autograd自动求导', link: '/artificial-intelligence/deep-learning/Pytorch与autograd自动求导.md' },
                { text: 'PyTorch与卷积神经网络', link: '/artificial-intelligence/deep-learning/Pytorch与卷积神经网络(OpenCV).md' },
                { text: 'PyTorch可视化工具', link: '/artificial-intelligence/deep-learning/Pytorch可视化Visdom、tensorboardX和Torchvision.md' },
                { text: '从零搭建经典模型(CNN等)', link: '/artificial-intelligence/deep-learning/从零搭建GoogLeNet，ResNet18，ResNet50，vgg、mobilenetv1、mobilenetv2、shufflenetv1、shufflenetv2模型（Pytorch代码示例）.md' },
                { text: '从零搭建Attention模型', link: '/artificial-intelligence/deep-learning/从零搭建CBAM、SENet、STN、transformer、mobile_vit、simple_vit、vit模型（Pytorch代码示例）.md' },
                { text: '人脸识别face_recognition详解', link: '/artificial-intelligence/deep-learning/人脸识别：face_recognition参数详解.md' },
                { text: 'CNN卷积神经网络', link: '/artificial-intelligence/deep-learning/深度学习面试笔试之卷积神经网络(CNN).md' },
                { text: '前向神经网络与反向传播', link: '/artificial-intelligence/deep-learning/深度学习面试笔试之前向神经网络-多层感知器、损失函数、反向传播.md' },
                { text: '深度学习优化方法', link: '/artificial-intelligence/deep-learning/深度学习面试笔试之深度学习的优化方法.md' },
                { text: '迁移学习与强化学习', link: '/artificial-intelligence/deep-learning/深度学习面试笔试之迁移学习(Transfer)、强化学习(Reinforcement) & 多任务.md' },
                { text: 'RNN、LSTM与GRU', link: '/artificial-intelligence/deep-learning/深度学习面试笔试之循环神经网络(RNN)、门控循环单元（GRU）、长短期记忆(LSTM).md' },
                { text: '图深度学习与A*算法', link: '/artificial-intelligence/deep-learning/图深度学习、A_（A-Star）算法、EMD和VMD详解.md' },
                { text: '视觉识别技术', link: '/artificial-intelligence/deep-learning/视觉识别：ffmpeg-python、ultralytics.YOLO、OpenCV-Python、标准RTSP地址格式.md' },
                { text: '语音识别技术', link: '/artificial-intelligence/deep-learning/语音识别：PyAudio、SoundDevice、Vosk、openai-whisper、Argos-Translate、FunASR（Python）.md' }
              ]
            },
            {
              text: '大模型',
              items: [
                { text: 'Ollama本地大模型部署', link: '/artificial-intelligence/large-models/Ollama详解，无网环境导入运行本地下载的大模型，无网环境pycharm插件大模型调用、Ollama Python api、coze-studio.md' },
                { text: '大模型微信公众号接入', link: '/artificial-intelligence/large-models/大模型接入微信公众号、QQ频道_群_个人 自动回复（专业领域），智能体（扣子，腾讯元器_QQ开放平台）.md' },
                { text: 'Transformers库详解', link: '/artificial-intelligence/large-models/Transformer；Hugging Face之transformers库、datasets库详解；Modelscope.md' }
              ]
            }
          ]
        }
      ],
      '/computer-science/': [
        {
          text: '计算机科学',
          items: [
            {
              text: '算法',
              items: []
            },
            {
              text: '数据结构',
              items: []
            },
            {
              text: '编程',
              items: [
                { text: '数据科学基础', link: '/computer-science/programming/数据科学：Numpy、Pandas笔记.md' },
                { text: '数据可视化', link: '/computer-science/programming/数据科学：Matplotlib、Seaborn笔记.md' },
                { text: '科学计算库', link: '/computer-science/programming/数据科学：Scipy、Scikit-Learn笔记.md' },
                { text: 'Python编程基础1', link: '/computer-science/programming/Python笔记1.1（datetime、argparse、sys、overwrite、eval、json、os、zfill、endswith、traceback、深浅拷贝）.md' },
                { text: 'Python编程基础2', link: '/computer-science/programming/Python笔记1.2（open、logging、os、shutil、glob、decode、encode、pickle、tqdm）.md' },
                { text: 'Python高级编程', link: '/computer-science/programming/Python笔记2（函数参数、面向对象、装饰器、高级函数、捕获异常、dir）.md' },
                { text: 'Python多线程多进程', link: '/computer-science/programming/Python 线程，进程，多线程，多进程以及并行执行for循环笔记.md' },
                { text: 'Python模块开发', link: '/computer-science/programming/Python 模块的制作、发布、安装.md' },
                { text: 'Python正则表达式', link: '/computer-science/programming/Python正则表达式（re模块）.md' },
                { text: 'Markdown语法', link: '/computer-science/programming/Markdown语法和表情.md' },
                { text: 'ACM模式输入输出', link: '/computer-science/programming/ACM模式之输入输出（Java_Python例题）.md' },
                { text: '高级程序设计C++', link: '/computer-science/programming/CLASS PROJECT高级程序设计C++.md' },
                { text: 'Python网络爬虫基础', link: '/computer-science/programming/笔记-Python爬虫技术基础及爬取百度新闻.md' },
                { text: 'Python爬虫框架', link: '/computer-science/programming/网络爬虫（Python：Requests、Beautiful Soup笔记）.md' },
                { text: 'Python爬虫高级技术', link: '/computer-science/programming/网络爬虫（Python：Selenium、Scrapy框架；爬虫与反爬虫笔记）.md' },
                { text: 'Python自动化工具', link: '/computer-science/programming/通过GitHub Actions给微信公众测试号和钉钉群定时推送消息（Python）.md' },
                { text: 'Python与Redis', link: '/computer-science/programming/Python redis 使用(笔记).md' },
                { text: 'Python操作数据库', link: '/computer-science/programming/Python操作数据库之pyodbc.md' },
                { text: 'Python连接MySQL', link: '/computer-science/programming/Python连接MySQL及查询实战.md' },
                { text: 'Cassandra数据库', link: '/computer-science/programming/Cassandra笔记.md' },
                { text: 'SQL常用语句', link: '/computer-science/programming/SQL常用语句 笔记.md' },
                { text: 'Python最优化算法', link: '/computer-science/programming/Python最优化算法学习笔记（Gurobi）.md' },
                { text: 'Python地图可视化', link: '/computer-science/programming/Python根据经纬度在地图上显示（folium详解）.md' },
                { text: 'Python GUI编程', link: '/computer-science/programming/Tkinter详解和爱心跳动示例（Python）.md' },
                { text: 'Cython脚本加密', link: '/computer-science/programming/如何使用Cython对python脚本加密成pyd_so.md' },
                { text: 'pip与包管理', link: '/computer-science/programming/Python：pip 安装第三方库速度很慢的解决办法，以及离线安装方法和conda换源，以及指定路径打开jupyter notebook.md' },
                { text: 'requirements.txt管理', link: '/computer-science/programming/项目依赖的python包requirements.txt文件的生成与安装.md' },
                { text: '框架安装与配置', link: '/computer-science/programming/Keras和Tensorflow（CPU）安装、Pytorch（CPU和GPU）安装以及jupyter使用虚拟环境.md' },
                { text: '神经网络实现', link: '/computer-science/programming/Keras搭建神经网络、Pytorch搭建神经网络和Sklearn的MLPRegressor.md' },
                { text: 'LightGBM参数详解', link: '/computer-science/programming/LightGBM原生接口和Sklearn接口参数详解.md' },
                { text: 'CatBoost参数详解', link: '/computer-science/programming/CatBoost原生接口和Sklearn接口参数详解.md' },
                { text: '自定义损失函数', link: '/computer-science/programming/LightGBM、XGBoost和CatBoost自定义损失函数和评估指标.md' },
                { text: '数据可视化进阶', link: '/computer-science/programming/Python数据可视化之Matplotlib与Pyecharts参数详解.md' },
                { text: 'Graphviz可视化', link: '/computer-science/programming/Graphviz安装及使用：决策树可视化.md' },
                { text: 'Git版本控制', link: '/computer-science/programming/Mac_Windows Git配置SSH和Git常用命令（笔记）.md' },
                { text: 'Linux基础命令', link: '/computer-science/programming/Linux常用基本命令.md' },
                { text: 'Linux高级工具', link: '/computer-science/programming/Linux jq 、vim以及Linux集群安装miniconda并配置虚拟环境（笔记）.md' },
                { text: 'Windows定时任务', link: '/computer-science/programming/Windows 定时任务设置、批处理(.bat)命令详解和通过conda虚拟环境定时运行Python程序.md' },
                { text: 'Docker实践', link: '/computer-science/programming/AI开发者的Docker实践：汉化（中文），更换镜像源，Dockerfile，部署Python项目.md' },
                { text: 'GitHub博客搭建', link: '/computer-science/programming/基于Hexo的主题Fluid搭建Github博客.md' },
                { text: 'Web开发HTML', link: '/computer-science/programming/Web之HTML笔记.md' },
                { text: 'Web开发CSS', link: '/computer-science/programming/Web之CSS笔记.md' },
                { text: 'Web开发JavaScript', link: '/computer-science/programming/Web之JavaScript(jQuery)笔记.md' },
                { text: '大数据Spark基础', link: '/computer-science/programming/Spark笔记（pyspark）.md' },
                { text: '大数据PySpark', link: '/computer-science/programming/pyspark笔记（RDD,DataFrame和Spark SQL）.md' },
                { text: '云存储与分布式文件', link: '/computer-science/programming/aws s3命令与hdfs dfs命令_hadoop fs命令（笔记）.md' },
                { text: 'AGV工业场景Baseline', link: '/computer-science/programming/AGV分拣工业场景Baseline.md' },
                { text: '工具配置技巧', link: '/computer-science/programming/笔记-pd.set_option()、warnings、np.set_printoptions参数详解.md' },
                { text: '命令行工具', link: '/computer-science/programming/如何在cmd中打开指定文件夹路径（三种方法）.md' },
                { text: 'XML处理', link: '/computer-science/programming/Python读写xml（xml，lxml）Edge 浏览器插件 WebTab - 免费ChatGPT.md' },
                { text: 'Scikit-learn扩展', link: '/computer-science/programming/Scikit-learn使用和扩展之mlxtend（Stacking...）.md' },
                { text: 'Numpy reshape详解', link: '/computer-science/programming/Python的reshape的用法和reshape(1,-1)、reshape(-1,1).md' }
              ]
            }
          ]
        }
      ],
      '/links/': [
        {
          text: '相关链接',
          items: [
            { text: 'CSDN博客推荐', link: '/links/csdn.md' },
            { text: '知乎精选', link: '/links/zhihu.md' },
            { text: '微信公众号', link: '/links/wechat.md' },
          ]
        }
      ]
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/QInzhengk' }
    ],

    footer: {
      message: 'Released under the MIT License.',
      copyright: 'Copyright © 2025 数学建模与人工智能'
    },

    search: {
      provider: 'local'
    },

    editLink: {
      pattern: 'https://github.com/QInzhengk/the-milky-way/edit/main/docs/:path',
      text: '在 GitHub 上编辑此页面'
    },

    docFooter: {
      prev: '上一页',
      next: '下一页'
    },

    outline: {
      label: '页面导航'
    },

    returnToTopLabel: '回到顶部',

    sidebarMenuLabel: '菜单',

    darkModeSwitchLabel: '主题',

    lastUpdated: {
      text: '最后更新于',
      formatOptions: {
        dateStyle: 'full',
        timeStyle: 'medium'
      }
    },

    carbonAds: {
      code: 'your-carbon-code',
      placement: 'your-carbon-placement'
    }
  },

  vite: {
    ssr: {
      noExternal: []
    }
  }
})
