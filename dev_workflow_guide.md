# 工程开发学习指南

> 适用于：第一次做相对完整项目的开发者

---

## 一、项目启动前的准备

### 1. 搭好环境与工具链

```bash
# 用 venv 隔离环境（每个项目独立，不污染系统）
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows

# 安装依赖，并立刻记录
pip install fastapi uvicorn pynvml psutil
pip freeze > requirements.txt    # 这个文件要提交到 git
```

### 2. Git 从第一天开始用

```bash
git init

# 创建 .gitignore（重要！避免把垃圾文件提交进去）
echo ".venv/\n__pycache__/\n*.pyc\n.env" > .gitignore

git add .
git commit -m "init: project scaffold"
```

**核心习惯：每完成一个小功能就 commit，message 说清楚做了什么。**

---

## 二、如何学习一个陌生的 Package

以 `pynvml` 为例，这套方法适用于任何新库。

### Step 1：看官方文档 / README，建立整体认知

```
搜索：pynvml github          →  看 README，了解能做什么
搜索：NVML API reference     →  了解底层数据来源
```

### Step 2：用交互式环境快速实验（最重要的习惯）

**不要直接写进项目文件！先在 ipython 或 jupyter 里探索。**

```python
import pynvml
pynvml.nvmlInit()

# 不知道有什么函数？用 dir() 看
print(dir(pynvml))

# 不知道函数怎么用？用 help()
help(pynvml.nvmlDeviceGetMemoryInfo)

# 直接跑，观察输出结构
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
print(type(mem))              # 看类型
print(mem)                    # 看完整内容
print(mem.total, mem.used, mem.free)   # 逐个探索字段
```

### Step 3：把探索结果固化成自己的笔记

```python
# notes/pynvml_exploration.py  ← 学习记录，不进生产代码

"""
pynvml 探索笔记
----------------
nvmlDeviceGetMemoryInfo(handle)
  → 返回 struct { total, used, free }，单位 bytes

nvmlDeviceGetUtilizationRates(handle)
  → 返回 struct { gpu, memory }，范围 0-100 (百分比)

nvmlDeviceGetComputeRunningProcesses(handle)
  → 返回 [ struct { pid, usedGpuMemory } ]
  ⚠️ 注意：只有显存占用，没有 per-process SM 利用率！

nvmlDeviceGetCount()
  → 返回 GPU 数量（多卡环境用得到）
"""
```

---

## 三、代码组织方式（工程思维）

### 核心原则：每个文件只做一件事

```
gpu-monitor/
├── backend/
│   ├── main.py          # 只负责：启动 app，注册路由
│   ├── collector.py     # 只负责：和 GPU 硬件交互，返回数据
│   ├── models.py        # 只负责：定义数据结构（Pydantic）
│   └── api/
│       ├── stats.py     # 只负责：/api/stats 相关路由
│       └── process.py   # 只负责：/api/processes 相关路由
├── frontend/
│   ├── index.html
│   ├── dashboard.js
│   └── style.css
├── notes/               # 探索笔记，不进生产
├── tests/               # 测试文件
├── .gitignore
├── requirements.txt
└── TODO.md
```

### 写代码的顺序：从内向外

```
数据结构  →  采集层  →  API层  →  前端
(models)    (collector)  (routes)   (HTML/JS)
```

**永远先定义好数据长什么样，再写逻辑。**

```python
# models.py ← 第一步先写这个，定义清楚数据的形状

from pydantic import BaseModel

class ProcessInfo(BaseModel):
    pid: int
    name: str
    gpu_memory_mb: float
    cpu_percent: float

class GpuStats(BaseModel):
    gpu_index: int
    utilization_percent: int
    memory_used_mb: float
    memory_total_mb: float
    processes: list[ProcessInfo]
```

---

## 四、开发节奏

### 用 TODO.md 驱动开发，不要想着一口气写完

```markdown
## 进行中

- [ ] collector.py: 实现 get_gpu_stats() 返回 GpuStats

## 待办

- [ ] collector.py: 实现 get_process_list()
- [ ] main.py: GET /api/stats 接口
- [ ] main.py: WebSocket /ws/live
- [ ] main.py: POST /api/processes/{pid}/kill
- [ ] frontend: 基础页面框架
- [ ] frontend: Chart.js 折线图

## 完成

- [x] 项目结构初始化
- [x] pynvml 探索笔记
```

**每次只专注一个 checkbox，完成就 commit。**

### 一个功能的完整开发流程

```
1. 在 notes/ 或 jupyter 里先探索 API，跑通
         ↓
2. 在 models.py 定义数据结构
         ↓
3. 在 collector.py 实现采集函数
         ↓
4. 直接 python collector.py 跑一下，print 验证输出
         ↓
5. 写 FastAPI 路由，打开 localhost:8000/docs 在线测试
         ↓
6. 没问题，git commit
```

---

## 五、调试方法

### 不要只会 print，学会用 debugger

```python
# 在想暂停的地方插入这一行
import pdb; pdb.set_trace()

# 程序运行到这里会暂停，进入交互模式
# 常用命令：
#   n        下一行
#   s        进入函数内部
#   p 变量名  打印变量值
#   q        退出调试
```

### FastAPI 自带的调试神器

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

打开 `http://localhost:8000/docs`，可以直接在浏览器里测试每个接口，**不需要写前端就能验证后端逻辑是否正确**。

---

## 六、遇到问题的解决路径

### 查资料的优先级

```
1. 官方文档                 ← 最权威，先查这里
2. 官方 GitHub Issues       ← 别人遇到过同样的 bug
3. 精准 Google 搜索         ← 加上库名 + 函数名 + 报错信息
4. Claude / ChatGPT         ← 解释概念、review 代码
5. Stack Overflow           ← 通用编程问题
```

### 问 AI 的正确方式

```
❌  差："pynvml 怎么用"

✅  好："我用 pynvml 的 nvmlDeviceGetComputeRunningProcesses
        返回了空列表，但 nvidia-smi 能看到进程。
        我的代码是：[贴代码]
        报错信息是：[贴报错]
        可能是什么原因？"
```

**标准格式：你做了什么 + 期望结果 + 实际结果 + 相关代码**

---

## 七、行动计划

| 天数    | 任务                                          |
| ------- | --------------------------------------------- |
| Day 1   | 搭环境，git init，建目录结构，装依赖          |
| Day 2   | jupyter 里探索 pynvml，写 notes，写 models.py |
| Day 3   | 写 collector.py，CLI 跑通验证                 |
| Day 4-5 | 写 FastAPI 接口，/docs 里测试通               |
| Day 6-7 | WebSocket 推送 + 前端基础页面                 |

---
