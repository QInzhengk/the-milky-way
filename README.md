# 数学建模与人工智能博客

一个专注于数学建模、人工智能和计算机科学的技术博客网站。

## 🚀 项目特色

### 📊 结构化知识体系
- **数学建模**：涵盖数学竞赛、大学生和研究生建模竞赛
- **人工智能**：深入机器学习、深度学习和大模型技术
- **计算机科学**：包含算法、数据结构和编程实践
- **相关链接**：精选CSDN、知乎和微信公众号资源

### 🎯 学习路径清晰
- 系统化的内容组织
- 从基础到进阶的学习路径
- 理论与实践相结合
- 丰富的代码示例和案例分析

### 💡 实战导向
- 竞赛真题解析
- 项目实战案例
- 最佳实践分享
- 工具使用指南

## 🛠 技术栈

- **框架**：VitePress
- **语言**：TypeScript/JavaScript
- **样式**：CSS3
- **构建工具**：Vite
- **部署**：GitHub Pages

## 📁 项目结构

```
the-milky-way/
├── docs/                          # 文档目录
│   ├── .vitepress/               # VitePress配置
│   │   ├── config.mts            # 网站配置
│   │   └── theme/                # 自定义主题
│   ├── mathematical-modeling/     # 数学建模
│   │   ├── math-competition/      # 数学竞赛
│   │   ├── undergraduate-modeling/ # 大学生建模
│   │   └── graduate-modeling/    # 研究生建模
│   ├── artificial-intelligence/    # 人工智能
│   │   ├── machine-learning/      # 机器学习
│   │   ├── deep-learning/        # 深度学习
│   │   └── large-models/         # 大模型
│   ├── computer-science/          # 计算机科学
│   │   ├── algorithms/            # 算法
│   │   ├── data-structures/       # 数据结构
│   │   └── programming/           # 编程
│   ├── links/                    # 相关链接
│   │   ├── csdn.md               # CSDN推荐
│   │   ├── zhihu.md              # 知乎精选
│   │   └── wechat.md             # 微信公众号
│   └── index.md                  # 首页
├── package.json                   # 项目配置
└── README.md                      # 项目说明
```

## 🚦 快速开始

### 环境要求
- Node.js >= 22
- npm 或 yarn

### 安装依赖
```bash
npm install
```

### 开发服务器
```bash
npm run docs:dev
```

### 构建生产版本
```bash
npm run docs:build
```

### 预览生产版本
```bash
npm run docs:preview
```

## 📚 内容特色

### 数学建模板块
- **基础入门**：数学建模基本概念和方法
- **竞赛指导**：国赛、美赛备赛指南
- **算法详解**：常用数学建模算法实现
- **案例实战**：经典竞赛案例分析

### 人工智能板块
- **机器学习**：监督学习、非监督学习、强化学习
- **深度学习**：神经网络、CNN、RNN、Transformer
- **大模型**：预训练模型、微调技术、应用实践
- **工具框架**：PyTorch、TensorFlow实战

### 计算机科学板块
- **算法设计**：排序、搜索、动态规划、图算法
- **数据结构**：线性结构、树形结构、图结构
- **编程实践**：Python、C++、Java高级技巧
- **系统设计**：架构设计、性能优化、最佳实践

## 🎨 设计特色

### 响应式设计
- 支持桌面端、平板和移动设备
- 自适应布局，最佳阅读体验
- 触摸友好的交互设计

### 主题系统
- 支持明暗主题切换
- 优雅的配色方案
- 清晰的视觉层次

### 用户体验
- 快速搜索功能
- 清晰的导航结构
- 代码高亮显示
- 友好的交互反馈

## 🔧 自定义配置

### 网站配置
编辑 `docs/.vitepress/config.mts` 文件可以自定义：
- 网站标题和描述
- 导航菜单结构
- 侧边栏配置
- 社交链接
- 主题样式

### 样式定制
在 `docs/.vitepress/theme/custom.css` 中添加自定义样式：
```css
:root {
  --vp-c-brand-1: #3b82f6;
  /* 更多自定义变量 */
}
```

## 📈 性能优化

### 加载性能
- 代码分割和懒加载
- 图片优化和压缩
- CDN加速（生产环境）
- 缓存策略优化

### SEO优化
- 语义化HTML结构
- 合理的meta标签
- sitemap生成
- 搜索引擎友好

## 🚀 部署指南

### GitHub Pages (推荐)

#### 自动部署（GitHub Actions）
1. 项目已配置好GitHub Actions工作流
2. 只需将代码推送到`main`或者`master`分支
3. GitHub Actions会自动：
   - 安装依赖
   - 构建站点
   - 部署到GitHub Pages
4. 部署完成后访问：`https://<your-username>.github.io/the-milky-way/`

#### 手动部署
1. 构建站点：`npm run docs:build`
2. 安装部署工具：`npm install -g gh-pages`
3. 部署：`gh-pages -d docs/.vitepress/dist`

### 本地预览
```bash
npm run docs:preview
```

## 📄 开发规范

### .gitignore配置
项目已配置好标准的.gitignore文件，包含：
- 依赖目录（node_modules）
- 构建输出（dist, cache）
- 日志文件
- 编辑器配置
- 环境变量文件
- OS生成文件

### 提交规范
- 使用语义化提交信息
- 保持提交粒度合理
- 定期同步主分支

### 代码风格
- 遵循JavaScript/TypeScript最佳实践
- 保持代码简洁清晰
- 添加必要的注释
- 确保文档与代码同步

## 🤝 贡献指南

### 参与方式
- 📝 提交内容建议
- 🐛 报告问题
- 💡 提出改进意见
- 🔧 贡献代码

### 内容贡献
1. Fork 本仓库
2. 创建特性分支
3. 编写或修改内容
4. 提交Pull Request

### 写作规范
- 使用Markdown格式
- 遵循现有目录结构
- 添加必要的代码示例
- 保持内容准确性

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

感谢以下开源项目：
- [VitePress](https://vitepress.dev/) - 静态网站生成器
- [Vue.js](https://vuejs.org/) - 响应式框架
- [Vite](https://vitejs.dev/) - 构建工具

## 📞 联系方式

- 💬 微信公众号：数学建模与人工智能
- 🔗 GitHub：[数学建模与人工智能](https://github.com/QInzhengk)

---

⭐ 如果这个项目对您有帮助，请给我们一个星标！

📚 持续学习，共同进步！