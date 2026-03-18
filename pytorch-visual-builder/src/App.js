import { useState, useRef, useCallback, useEffect, useMemo } from "react";

const T = {
  zh: {
    title: "PyTorch 可视化构建器",
    builder: "网络构建",
    modelStore: "模型商店",
    datasetStore: "数据集商店",
    training: "训练设置",
    uiSettings: "界面设置",
    layersTab: "层素材",
    modelsTab: "模型库",
    export: "导出",
    import: "导入",
    emptyCanvas: "拖拽左侧层 / 模型到此区域开始构建",
    props: "层属性",
    selectLayer: "点击画布中的层查看属性",
    delete: "删除",
    clear: "清空画布",
    exportPy: ".py",
    exportIpynb: ".ipynb",
    importFile: "导入",
    search: "搜索...",
    categories: {
      conv: "卷积",
      act: "激活",
      norm: "归一化",
      pool: "池化",
      linear: "线性",
      rnn: "循环",
      reg: "正则化",
      util: "工具",
      attn: "注意力",
      container: "容器",
    },
    in: "输入",
    out: "输出",
    moveUp: "↑ 上移",
    moveDown: "↓ 下移",
    totalParams: "参数量",
    totalLayers: "层数",
    modelDesc: "点击下载按钮将模型加入左侧模型库",
    datasetDesc: "点击查看数据集下载代码",
    loadModel: "下载到模型库",
    loaded: "已在模型库",
    samples: "样本",
    classes: "类别",
    size: "大小",
    task: "任务",
    accuracy: "精度",
    inputShape: "输入张量",
    shapeAfter: "输出形状",
    trainTitle: "训练与验证配置",
    optimizer: "优化器",
    lr: "学习率",
    weightDecay: "权重衰减",
    scheduler: "调度器",
    lossFunc: "损失函数",
    epochs: "轮数",
    batchSizeTrain: "批大小",
    valSplit: "验证比例",
    earlyStop: "早停patience",
    genTrainCode: "生成训练代码",
    uiTitle: "界面自定义",
    theme: "主题",
    dark: "深色",
    light: "浅色",
    fontSize: "字体大小",
    bgImage: "背景图URL",
    bgOpacity: "背景透明度",
    accentColor: "主题色",
    resetUI: "重置默认",
    importSuccess: "导入成功！",
    exportSuccess: "导出成功！",
    image: "图像",
    sequence: "序列",
    tabular: "表格",
    text: "文本",
    noModels: "模型库为空，请前往模型商店下载",
    useModel: "拖拽或双击加载到画布",
    layerCount: "层",
  },
  en: {
    title: "PyTorch Visual Builder",
    builder: "Builder",
    modelStore: "Model Store",
    datasetStore: "Datasets",
    training: "Training",
    uiSettings: "UI",
    layersTab: "Layers",
    modelsTab: "Models",
    export: "Export",
    import: "Import",
    emptyCanvas: "Drag layers or models here to start",
    props: "Properties",
    selectLayer: "Click a layer on canvas",
    delete: "Delete",
    clear: "Clear All",
    exportPy: ".py",
    exportIpynb: ".ipynb",
    importFile: "Import",
    search: "Search...",
    categories: {
      conv: "Conv",
      act: "Activation",
      norm: "Norm",
      pool: "Pool",
      linear: "Linear",
      rnn: "RNN",
      reg: "Reg",
      util: "Util",
      attn: "Attn",
      container: "Container",
    },
    in: "In",
    out: "Out",
    moveUp: "↑ Up",
    moveDown: "↓ Down",
    totalParams: "Params",
    totalLayers: "Layers",
    modelDesc: "Click download to add models to your library",
    datasetDesc: "View dataset download code",
    loadModel: "Add to Library",
    loaded: "In Library",
    samples: "Samples",
    classes: "Classes",
    size: "Size",
    task: "Task",
    accuracy: "Acc",
    inputShape: "Input Tensor",
    shapeAfter: "Output Shape",
    trainTitle: "Training & Validation",
    optimizer: "Optimizer",
    lr: "Learning Rate",
    weightDecay: "Weight Decay",
    scheduler: "Scheduler",
    lossFunc: "Loss",
    epochs: "Epochs",
    batchSizeTrain: "Batch Size",
    valSplit: "Val Split",
    earlyStop: "Early Stop",
    genTrainCode: "Generate Code",
    uiTitle: "UI Customization",
    theme: "Theme",
    dark: "Dark",
    light: "Light",
    fontSize: "Font Size",
    bgImage: "Background URL",
    bgOpacity: "BG Opacity",
    accentColor: "Accent Color",
    resetUI: "Reset",
    importSuccess: "Imported!",
    exportSuccess: "Exported!",
    image: "Image",
    sequence: "Sequence",
    tabular: "Tabular",
    text: "Text",
    noModels: "Library empty — download from Model Store",
    useModel: "Drag or double-click to load",
    layerCount: "layers",
  },
};

const calcShape = (layerId, params, inShape) => {
  if (!inShape || !inShape.length) return inShape;
  const s = [...inShape];
  try {
    switch (layerId) {
      case "conv2d":
      case "convT2d": {
        const [N, C, H, W] =
          s.length >= 4
            ? s
            : [s[0], params.in_channels, s[2] || 32, s[3] || 32];
        const k = params.kernel_size,
          st = params.stride,
          p = params.padding;
        if (layerId === "conv2d")
          return [
            N,
            params.out_channels,
            Math.floor((H + 2 * p - k) / st) + 1,
            Math.floor((W + 2 * p - k) / st) + 1,
          ];
        return [
          N,
          params.out_channels,
          (H - 1) * st - 2 * p + k,
          (W - 1) * st - 2 * p + k,
        ];
      }
      case "conv1d": {
        const [N, C, L] = s.length >= 3 ? s : [s[0], params.in_channels, 100];
        return [
          N,
          params.out_channels,
          Math.floor(
            (L + 2 * params.padding - params.kernel_size) / params.stride,
          ) + 1,
        ];
      }
      case "conv3d": {
        const [N, C, D, H, W] =
          s.length >= 5 ? s : [s[0], params.in_channels, 16, 16, 16];
        const f = (x) =>
          Math.floor(
            (x + 2 * params.padding - params.kernel_size) / params.stride,
          ) + 1;
        return [N, params.out_channels, f(D), f(H), f(W)];
      }
      case "dwconv": {
        const [N, C, H, W] = s;
        const k = params.kernel_size,
          st = params.stride,
          p = params.padding;
        return [
          N,
          C,
          Math.floor((H + 2 * p - k) / st) + 1,
          Math.floor((W + 2 * p - k) / st) + 1,
        ];
      }
      case "maxp2d":
      case "avgp2d": {
        const [N, C, H, W] = s;
        const k = params.kernel_size,
          st = params.stride,
          p = params.padding;
        return [
          N,
          C,
          Math.floor((H + 2 * p - k) / st) + 1,
          Math.floor((W + 2 * p - k) / st) + 1,
        ];
      }
      case "adapavg":
        return [s[0], s[1], params.output_size, params.output_size];
      case "flatten": {
        const sd =
          params.start_dim < 0 ? s.length + params.start_dim : params.start_dim;
        return [...s.slice(0, sd), s.slice(sd).reduce((a, b) => a * b, 1)];
      }
      case "linear":
        return [...s.slice(0, -1), params.out_features];
      case "lstm":
      case "gru":
        return [
          s[0],
          s[1],
          params.hidden_size * (params.bidirectional ? 2 : 1),
        ];
      case "embed":
        return [...s, params.embedding_dim];
      case "upsample":
        return [
          s[0],
          s[1],
          s[2] * params.scale_factor,
          s[3] * params.scale_factor,
        ];
      default:
        return [...s];
    }
  } catch {
    return s;
  }
};

const LAYERS = [
  {
    id: "conv1d",
    name: "Conv1d",
    cat: "conv",
    color: "#3B82F6",
    inF: "(N,C_in,L)",
    outF: "(N,C_out,L')",
    params: {
      in_channels: 1,
      out_channels: 16,
      kernel_size: 3,
      stride: 1,
      padding: 0,
    },
    pCount: (p) =>
      p.in_channels * p.out_channels * p.kernel_size + p.out_channels,
  },
  {
    id: "conv2d",
    name: "Conv2d",
    cat: "conv",
    color: "#3B82F6",
    inF: "(N,C_in,H,W)",
    outF: "(N,C_out,H',W')",
    params: {
      in_channels: 3,
      out_channels: 64,
      kernel_size: 3,
      stride: 1,
      padding: 1,
    },
    pCount: (p) =>
      p.in_channels * p.out_channels * p.kernel_size * p.kernel_size +
      p.out_channels,
  },
  {
    id: "conv3d",
    name: "Conv3d",
    cat: "conv",
    color: "#3B82F6",
    inF: "(N,C_in,D,H,W)",
    outF: "(N,C_out,D',H',W')",
    params: {
      in_channels: 1,
      out_channels: 32,
      kernel_size: 3,
      stride: 1,
      padding: 1,
    },
    pCount: (p) =>
      p.in_channels * p.out_channels * Math.pow(p.kernel_size, 3) +
      p.out_channels,
  },
  {
    id: "convT2d",
    name: "ConvTranspose2d",
    cat: "conv",
    color: "#3B82F6",
    inF: "(N,C_in,H,W)",
    outF: "(N,C_out,H',W')",
    params: {
      in_channels: 64,
      out_channels: 32,
      kernel_size: 4,
      stride: 2,
      padding: 1,
    },
    pCount: (p) =>
      p.in_channels * p.out_channels * p.kernel_size * p.kernel_size +
      p.out_channels,
  },
  {
    id: "dwconv",
    name: "DepthwiseConv2d",
    cat: "conv",
    color: "#3B82F6",
    inF: "(N,C,H,W)",
    outF: "(N,C,H',W')",
    params: { channels: 64, kernel_size: 3, stride: 1, padding: 1 },
    pCount: (p) => p.channels * p.kernel_size * p.kernel_size + p.channels,
  },
  {
    id: "relu",
    name: "ReLU",
    cat: "act",
    color: "#EF4444",
    inF: "(*)",
    outF: "(*)",
    params: { inplace: false },
    pCount: () => 0,
  },
  {
    id: "leaky",
    name: "LeakyReLU",
    cat: "act",
    color: "#EF4444",
    inF: "(*)",
    outF: "(*)",
    params: { negative_slope: 0.01 },
    pCount: () => 0,
  },
  {
    id: "gelu",
    name: "GELU",
    cat: "act",
    color: "#EF4444",
    inF: "(*)",
    outF: "(*)",
    params: {},
    pCount: () => 0,
  },
  {
    id: "silu",
    name: "SiLU",
    cat: "act",
    color: "#EF4444",
    inF: "(*)",
    outF: "(*)",
    params: {},
    pCount: () => 0,
  },
  {
    id: "sigmoid",
    name: "Sigmoid",
    cat: "act",
    color: "#EF4444",
    inF: "(*)",
    outF: "(*)",
    params: {},
    pCount: () => 0,
  },
  {
    id: "tanh",
    name: "Tanh",
    cat: "act",
    color: "#EF4444",
    inF: "(*)",
    outF: "(*)",
    params: {},
    pCount: () => 0,
  },
  {
    id: "softmax",
    name: "Softmax",
    cat: "act",
    color: "#EF4444",
    inF: "(*)",
    outF: "(*)",
    params: { dim: 1 },
    pCount: () => 0,
  },
  {
    id: "bn1d",
    name: "BatchNorm1d",
    cat: "norm",
    color: "#8B5CF6",
    inF: "(N,C)",
    outF: "same",
    params: { num_features: 64 },
    pCount: (p) => p.num_features * 4,
  },
  {
    id: "bn2d",
    name: "BatchNorm2d",
    cat: "norm",
    color: "#8B5CF6",
    inF: "(N,C,H,W)",
    outF: "same",
    params: { num_features: 64 },
    pCount: (p) => p.num_features * 4,
  },
  {
    id: "ln",
    name: "LayerNorm",
    cat: "norm",
    color: "#8B5CF6",
    inF: "(N,*,norm)",
    outF: "same",
    params: { normalized_shape: 256 },
    pCount: (p) => p.normalized_shape * 2,
  },
  {
    id: "gn",
    name: "GroupNorm",
    cat: "norm",
    color: "#8B5CF6",
    inF: "(N,C,*)",
    outF: "same",
    params: { num_groups: 8, num_channels: 64 },
    pCount: (p) => p.num_channels * 2,
  },
  {
    id: "maxp2d",
    name: "MaxPool2d",
    cat: "pool",
    color: "#F59E0B",
    inF: "(N,C,H,W)",
    outF: "(N,C,H',W')",
    params: { kernel_size: 2, stride: 2, padding: 0 },
    pCount: () => 0,
  },
  {
    id: "avgp2d",
    name: "AvgPool2d",
    cat: "pool",
    color: "#F59E0B",
    inF: "(N,C,H,W)",
    outF: "(N,C,H',W')",
    params: { kernel_size: 2, stride: 2, padding: 0 },
    pCount: () => 0,
  },
  {
    id: "adapavg",
    name: "AdaptiveAvgPool2d",
    cat: "pool",
    color: "#F59E0B",
    inF: "(N,C,H,W)",
    outF: "(N,C,o,o)",
    params: { output_size: 1 },
    pCount: () => 0,
  },
  {
    id: "linear",
    name: "Linear",
    cat: "linear",
    color: "#10B981",
    inF: "(N,*,in)",
    outF: "(N,*,out)",
    params: { in_features: 512, out_features: 256, bias: true },
    pCount: (p) =>
      p.in_features * p.out_features + (p.bias ? p.out_features : 0),
  },
  {
    id: "lstm",
    name: "LSTM",
    cat: "rnn",
    color: "#EC4899",
    inF: "(seq,N,in)",
    outF: "(seq,N,hid)",
    params: {
      input_size: 128,
      hidden_size: 256,
      num_layers: 1,
      bidirectional: false,
    },
    pCount: (p) => {
      const d = p.bidirectional ? 2 : 1;
      return (
        d *
        p.num_layers *
        4 *
        (p.input_size * p.hidden_size +
          p.hidden_size * p.hidden_size +
          p.hidden_size)
      );
    },
  },
  {
    id: "gru",
    name: "GRU",
    cat: "rnn",
    color: "#EC4899",
    inF: "(seq,N,in)",
    outF: "(seq,N,hid)",
    params: { input_size: 128, hidden_size: 256, num_layers: 1 },
    pCount: (p) =>
      p.num_layers *
      3 *
      (p.input_size * p.hidden_size +
        p.hidden_size * p.hidden_size +
        p.hidden_size),
  },
  {
    id: "tfenc",
    name: "TransformerEncoderLayer",
    cat: "attn",
    color: "#06B6D4",
    inF: "(seq,N,d)",
    outF: "(seq,N,d)",
    params: { d_model: 512, nhead: 8, dim_feedforward: 2048, dropout: 0.1 },
    pCount: (p) =>
      4 * p.d_model * p.d_model +
      2 * p.d_model * p.dim_feedforward +
      4 * p.d_model +
      2 * p.dim_feedforward,
  },
  {
    id: "mha",
    name: "MultiheadAttention",
    cat: "attn",
    color: "#06B6D4",
    inF: "(seq,N,e)",
    outF: "(seq,N,e)",
    params: { embed_dim: 512, num_heads: 8 },
    pCount: (p) => 4 * p.embed_dim * p.embed_dim + 4 * p.embed_dim,
  },
  {
    id: "dropout",
    name: "Dropout",
    cat: "reg",
    color: "#6B7280",
    inF: "(*)",
    outF: "(*)",
    params: { p: 0.5 },
    pCount: () => 0,
  },
  {
    id: "drop2d",
    name: "Dropout2d",
    cat: "reg",
    color: "#6B7280",
    inF: "(N,C,H,W)",
    outF: "same",
    params: { p: 0.5 },
    pCount: () => 0,
  },
  {
    id: "flatten",
    name: "Flatten",
    cat: "util",
    color: "#A3A3A3",
    inF: "(N,*)",
    outF: "(N,flat)",
    params: { start_dim: 1, end_dim: -1 },
    pCount: () => 0,
  },
  {
    id: "embed",
    name: "Embedding",
    cat: "util",
    color: "#A3A3A3",
    inF: "(*) Long",
    outF: "(*,dim)",
    params: { num_embeddings: 10000, embedding_dim: 256 },
    pCount: (p) => p.num_embeddings * p.embedding_dim,
  },
  {
    id: "upsample",
    name: "Upsample",
    cat: "util",
    color: "#A3A3A3",
    inF: "(N,C,H,W)",
    outF: "(N,C,sH,sW)",
    params: { scale_factor: 2, mode: "nearest" },
    pCount: () => 0,
  },
  {
    id: "sequential",
    name: "Sequential",
    cat: "container",
    color: "#F97316",
    inF: "(*)",
    outF: "(*)",
    params: { note: "container" },
    pCount: () => 0,
  },
  {
    id: "modulelist",
    name: "ModuleList",
    cat: "container",
    color: "#F97316",
    inF: "(*)",
    outF: "(*)",
    params: { note: "list" },
    pCount: () => 0,
  },
  {
    id: "moduledict",
    name: "ModuleDict",
    cat: "container",
    color: "#F97316",
    inF: "(*)",
    outF: "(*)",
    params: { note: "dict" },
    pCount: () => 0,
  },
];

const MODELS_DB = [
  {
    name: "LeNet-5",
    desc: "Classic CNN for digit recognition",
    size: "0.06M",
    task: "Classification",
    acc: "99.2% MNIST",
    year: 1998,
    inputType: "image",
    inputCfg: {
      batch: 1,
      channels: 1,
      height: 28,
      width: 28,
      seqLen: 50,
      features: 128,
    },
    color: "#3B82F6",
    layers: [
      {
        id: "conv2d",
        p: {
          in_channels: 1,
          out_channels: 6,
          kernel_size: 5,
          stride: 1,
          padding: 2,
        },
      },
      { id: "relu", p: { inplace: true } },
      { id: "maxp2d", p: { kernel_size: 2, stride: 2, padding: 0 } },
      {
        id: "conv2d",
        p: {
          in_channels: 6,
          out_channels: 16,
          kernel_size: 5,
          stride: 1,
          padding: 0,
        },
      },
      { id: "relu", p: { inplace: true } },
      { id: "maxp2d", p: { kernel_size: 2, stride: 2, padding: 0 } },
      { id: "flatten", p: { start_dim: 1, end_dim: -1 } },
      { id: "linear", p: { in_features: 400, out_features: 120, bias: true } },
      { id: "relu", p: { inplace: true } },
      { id: "linear", p: { in_features: 120, out_features: 84, bias: true } },
      { id: "relu", p: { inplace: true } },
      { id: "linear", p: { in_features: 84, out_features: 10, bias: true } },
    ],
  },
  {
    name: "AlexNet",
    desc: "Deep CNN, ImageNet 2012 winner",
    size: "61M",
    task: "Classification",
    acc: "56.5% Top-1",
    year: 2012,
    inputType: "image",
    inputCfg: {
      batch: 1,
      channels: 3,
      height: 224,
      width: 224,
      seqLen: 50,
      features: 128,
    },
    color: "#8B5CF6",
    layers: [
      {
        id: "conv2d",
        p: {
          in_channels: 3,
          out_channels: 64,
          kernel_size: 11,
          stride: 4,
          padding: 2,
        },
      },
      { id: "relu", p: { inplace: true } },
      { id: "maxp2d", p: { kernel_size: 3, stride: 2, padding: 0 } },
      {
        id: "conv2d",
        p: {
          in_channels: 64,
          out_channels: 192,
          kernel_size: 5,
          stride: 1,
          padding: 2,
        },
      },
      { id: "relu", p: { inplace: true } },
      { id: "maxp2d", p: { kernel_size: 3, stride: 2, padding: 0 } },
      {
        id: "conv2d",
        p: {
          in_channels: 192,
          out_channels: 384,
          kernel_size: 3,
          stride: 1,
          padding: 1,
        },
      },
      { id: "relu", p: { inplace: true } },
      {
        id: "conv2d",
        p: {
          in_channels: 384,
          out_channels: 256,
          kernel_size: 3,
          stride: 1,
          padding: 1,
        },
      },
      { id: "relu", p: { inplace: true } },
      {
        id: "conv2d",
        p: {
          in_channels: 256,
          out_channels: 256,
          kernel_size: 3,
          stride: 1,
          padding: 1,
        },
      },
      { id: "relu", p: { inplace: true } },
      { id: "maxp2d", p: { kernel_size: 3, stride: 2, padding: 0 } },
      { id: "adapavg", p: { output_size: 6 } },
      { id: "flatten", p: { start_dim: 1, end_dim: -1 } },
      { id: "dropout", p: { p: 0.5 } },
      {
        id: "linear",
        p: { in_features: 9216, out_features: 4096, bias: true },
      },
      { id: "relu", p: { inplace: true } },
      { id: "dropout", p: { p: 0.5 } },
      {
        id: "linear",
        p: { in_features: 4096, out_features: 4096, bias: true },
      },
      { id: "relu", p: { inplace: true } },
      {
        id: "linear",
        p: { in_features: 4096, out_features: 1000, bias: true },
      },
    ],
  },
  {
    name: "VGG-16",
    desc: "Very deep 3×3 filter network",
    size: "138M",
    task: "Classification",
    acc: "71.5% Top-1",
    year: 2014,
    inputType: "image",
    inputCfg: {
      batch: 1,
      channels: 3,
      height: 224,
      width: 224,
      seqLen: 50,
      features: 128,
    },
    color: "#10B981",
    layers: [
      {
        id: "conv2d",
        p: {
          in_channels: 3,
          out_channels: 64,
          kernel_size: 3,
          stride: 1,
          padding: 1,
        },
      },
      { id: "relu", p: { inplace: true } },
      {
        id: "conv2d",
        p: {
          in_channels: 64,
          out_channels: 64,
          kernel_size: 3,
          stride: 1,
          padding: 1,
        },
      },
      { id: "relu", p: { inplace: true } },
      { id: "maxp2d", p: { kernel_size: 2, stride: 2, padding: 0 } },
      {
        id: "conv2d",
        p: {
          in_channels: 64,
          out_channels: 128,
          kernel_size: 3,
          stride: 1,
          padding: 1,
        },
      },
      { id: "relu", p: { inplace: true } },
      {
        id: "conv2d",
        p: {
          in_channels: 128,
          out_channels: 128,
          kernel_size: 3,
          stride: 1,
          padding: 1,
        },
      },
      { id: "relu", p: { inplace: true } },
      { id: "maxp2d", p: { kernel_size: 2, stride: 2, padding: 0 } },
      {
        id: "conv2d",
        p: {
          in_channels: 128,
          out_channels: 256,
          kernel_size: 3,
          stride: 1,
          padding: 1,
        },
      },
      { id: "relu", p: { inplace: true } },
      {
        id: "conv2d",
        p: {
          in_channels: 256,
          out_channels: 256,
          kernel_size: 3,
          stride: 1,
          padding: 1,
        },
      },
      { id: "relu", p: { inplace: true } },
      {
        id: "conv2d",
        p: {
          in_channels: 256,
          out_channels: 256,
          kernel_size: 3,
          stride: 1,
          padding: 1,
        },
      },
      { id: "relu", p: { inplace: true } },
      { id: "maxp2d", p: { kernel_size: 2, stride: 2, padding: 0 } },
      {
        id: "conv2d",
        p: {
          in_channels: 256,
          out_channels: 512,
          kernel_size: 3,
          stride: 1,
          padding: 1,
        },
      },
      { id: "relu", p: { inplace: true } },
      {
        id: "conv2d",
        p: {
          in_channels: 512,
          out_channels: 512,
          kernel_size: 3,
          stride: 1,
          padding: 1,
        },
      },
      { id: "relu", p: { inplace: true } },
      {
        id: "conv2d",
        p: {
          in_channels: 512,
          out_channels: 512,
          kernel_size: 3,
          stride: 1,
          padding: 1,
        },
      },
      { id: "relu", p: { inplace: true } },
      { id: "maxp2d", p: { kernel_size: 2, stride: 2, padding: 0 } },
      {
        id: "conv2d",
        p: {
          in_channels: 512,
          out_channels: 512,
          kernel_size: 3,
          stride: 1,
          padding: 1,
        },
      },
      { id: "relu", p: { inplace: true } },
      {
        id: "conv2d",
        p: {
          in_channels: 512,
          out_channels: 512,
          kernel_size: 3,
          stride: 1,
          padding: 1,
        },
      },
      { id: "relu", p: { inplace: true } },
      {
        id: "conv2d",
        p: {
          in_channels: 512,
          out_channels: 512,
          kernel_size: 3,
          stride: 1,
          padding: 1,
        },
      },
      { id: "relu", p: { inplace: true } },
      { id: "maxp2d", p: { kernel_size: 2, stride: 2, padding: 0 } },
      { id: "adapavg", p: { output_size: 7 } },
      { id: "flatten", p: { start_dim: 1, end_dim: -1 } },
      {
        id: "linear",
        p: { in_features: 25088, out_features: 4096, bias: true },
      },
      { id: "relu", p: { inplace: true } },
      { id: "dropout", p: { p: 0.5 } },
      {
        id: "linear",
        p: { in_features: 4096, out_features: 4096, bias: true },
      },
      { id: "relu", p: { inplace: true } },
      { id: "dropout", p: { p: 0.5 } },
      {
        id: "linear",
        p: { in_features: 4096, out_features: 1000, bias: true },
      },
    ],
  },
  {
    name: "ResNet-18",
    desc: "Residual blocks with skip connections",
    size: "11.7M",
    task: "Classification",
    acc: "69.8% Top-1",
    year: 2015,
    inputType: "image",
    inputCfg: {
      batch: 1,
      channels: 3,
      height: 224,
      width: 224,
      seqLen: 50,
      features: 128,
    },
    color: "#F59E0B",
    layers: [
      {
        id: "conv2d",
        p: {
          in_channels: 3,
          out_channels: 64,
          kernel_size: 7,
          stride: 2,
          padding: 3,
        },
      },
      { id: "bn2d", p: { num_features: 64 } },
      { id: "relu", p: { inplace: true } },
      { id: "maxp2d", p: { kernel_size: 3, stride: 2, padding: 1 } },
      {
        id: "conv2d",
        p: {
          in_channels: 64,
          out_channels: 64,
          kernel_size: 3,
          stride: 1,
          padding: 1,
        },
      },
      { id: "bn2d", p: { num_features: 64 } },
      { id: "relu", p: { inplace: true } },
      {
        id: "conv2d",
        p: {
          in_channels: 64,
          out_channels: 64,
          kernel_size: 3,
          stride: 1,
          padding: 1,
        },
      },
      { id: "bn2d", p: { num_features: 64 } },
      { id: "relu", p: { inplace: true } },
      {
        id: "conv2d",
        p: {
          in_channels: 64,
          out_channels: 128,
          kernel_size: 3,
          stride: 2,
          padding: 1,
        },
      },
      { id: "bn2d", p: { num_features: 128 } },
      { id: "relu", p: { inplace: true } },
      {
        id: "conv2d",
        p: {
          in_channels: 128,
          out_channels: 128,
          kernel_size: 3,
          stride: 1,
          padding: 1,
        },
      },
      { id: "bn2d", p: { num_features: 128 } },
      { id: "relu", p: { inplace: true } },
      { id: "adapavg", p: { output_size: 1 } },
      { id: "flatten", p: { start_dim: 1, end_dim: -1 } },
      { id: "linear", p: { in_features: 128, out_features: 1000, bias: true } },
    ],
  },
  {
    name: "YOLOv3-tiny",
    desc: "Real-time object detection",
    size: "8.7M",
    task: "Detection",
    acc: "33.1 mAP",
    year: 2018,
    inputType: "image",
    inputCfg: {
      batch: 1,
      channels: 3,
      height: 416,
      width: 416,
      seqLen: 50,
      features: 128,
    },
    color: "#EF4444",
    layers: [
      {
        id: "conv2d",
        p: {
          in_channels: 3,
          out_channels: 16,
          kernel_size: 3,
          stride: 1,
          padding: 1,
        },
      },
      { id: "bn2d", p: { num_features: 16 } },
      { id: "leaky", p: { negative_slope: 0.1 } },
      { id: "maxp2d", p: { kernel_size: 2, stride: 2, padding: 0 } },
      {
        id: "conv2d",
        p: {
          in_channels: 16,
          out_channels: 32,
          kernel_size: 3,
          stride: 1,
          padding: 1,
        },
      },
      { id: "bn2d", p: { num_features: 32 } },
      { id: "leaky", p: { negative_slope: 0.1 } },
      { id: "maxp2d", p: { kernel_size: 2, stride: 2, padding: 0 } },
      {
        id: "conv2d",
        p: {
          in_channels: 32,
          out_channels: 64,
          kernel_size: 3,
          stride: 1,
          padding: 1,
        },
      },
      { id: "bn2d", p: { num_features: 64 } },
      { id: "leaky", p: { negative_slope: 0.1 } },
      { id: "maxp2d", p: { kernel_size: 2, stride: 2, padding: 0 } },
      {
        id: "conv2d",
        p: {
          in_channels: 64,
          out_channels: 128,
          kernel_size: 3,
          stride: 1,
          padding: 1,
        },
      },
      { id: "bn2d", p: { num_features: 128 } },
      { id: "leaky", p: { negative_slope: 0.1 } },
      { id: "maxp2d", p: { kernel_size: 2, stride: 2, padding: 0 } },
      {
        id: "conv2d",
        p: {
          in_channels: 128,
          out_channels: 256,
          kernel_size: 3,
          stride: 1,
          padding: 1,
        },
      },
      { id: "bn2d", p: { num_features: 256 } },
      { id: "leaky", p: { negative_slope: 0.1 } },
      { id: "maxp2d", p: { kernel_size: 2, stride: 2, padding: 0 } },
      {
        id: "conv2d",
        p: {
          in_channels: 256,
          out_channels: 512,
          kernel_size: 3,
          stride: 1,
          padding: 1,
        },
      },
      { id: "bn2d", p: { num_features: 512 } },
      { id: "leaky", p: { negative_slope: 0.1 } },
      {
        id: "conv2d",
        p: {
          in_channels: 512,
          out_channels: 1024,
          kernel_size: 3,
          stride: 1,
          padding: 1,
        },
      },
      { id: "bn2d", p: { num_features: 1024 } },
      { id: "leaky", p: { negative_slope: 0.1 } },
      {
        id: "conv2d",
        p: {
          in_channels: 1024,
          out_channels: 256,
          kernel_size: 1,
          stride: 1,
          padding: 0,
        },
      },
      {
        id: "conv2d",
        p: {
          in_channels: 256,
          out_channels: 512,
          kernel_size: 3,
          stride: 1,
          padding: 1,
        },
      },
      {
        id: "conv2d",
        p: {
          in_channels: 512,
          out_channels: 255,
          kernel_size: 1,
          stride: 1,
          padding: 0,
        },
      },
    ],
  },
  {
    name: "U-Net",
    desc: "Encoder-decoder segmentation",
    size: "31M",
    task: "Segmentation",
    acc: "IoU 0.92",
    year: 2015,
    inputType: "image",
    inputCfg: {
      batch: 1,
      channels: 1,
      height: 256,
      width: 256,
      seqLen: 50,
      features: 128,
    },
    color: "#06B6D4",
    layers: [
      {
        id: "conv2d",
        p: {
          in_channels: 1,
          out_channels: 64,
          kernel_size: 3,
          stride: 1,
          padding: 1,
        },
      },
      { id: "relu", p: { inplace: true } },
      {
        id: "conv2d",
        p: {
          in_channels: 64,
          out_channels: 64,
          kernel_size: 3,
          stride: 1,
          padding: 1,
        },
      },
      { id: "relu", p: { inplace: true } },
      { id: "maxp2d", p: { kernel_size: 2, stride: 2, padding: 0 } },
      {
        id: "conv2d",
        p: {
          in_channels: 64,
          out_channels: 128,
          kernel_size: 3,
          stride: 1,
          padding: 1,
        },
      },
      { id: "relu", p: { inplace: true } },
      {
        id: "conv2d",
        p: {
          in_channels: 128,
          out_channels: 128,
          kernel_size: 3,
          stride: 1,
          padding: 1,
        },
      },
      { id: "relu", p: { inplace: true } },
      { id: "maxp2d", p: { kernel_size: 2, stride: 2, padding: 0 } },
      {
        id: "conv2d",
        p: {
          in_channels: 128,
          out_channels: 256,
          kernel_size: 3,
          stride: 1,
          padding: 1,
        },
      },
      { id: "relu", p: { inplace: true } },
      {
        id: "conv2d",
        p: {
          in_channels: 256,
          out_channels: 256,
          kernel_size: 3,
          stride: 1,
          padding: 1,
        },
      },
      { id: "relu", p: { inplace: true } },
      {
        id: "convT2d",
        p: {
          in_channels: 256,
          out_channels: 128,
          kernel_size: 2,
          stride: 2,
          padding: 0,
        },
      },
      {
        id: "conv2d",
        p: {
          in_channels: 256,
          out_channels: 128,
          kernel_size: 3,
          stride: 1,
          padding: 1,
        },
      },
      { id: "relu", p: { inplace: true } },
      {
        id: "conv2d",
        p: {
          in_channels: 128,
          out_channels: 128,
          kernel_size: 3,
          stride: 1,
          padding: 1,
        },
      },
      { id: "relu", p: { inplace: true } },
      {
        id: "convT2d",
        p: {
          in_channels: 128,
          out_channels: 64,
          kernel_size: 2,
          stride: 2,
          padding: 0,
        },
      },
      {
        id: "conv2d",
        p: {
          in_channels: 128,
          out_channels: 64,
          kernel_size: 3,
          stride: 1,
          padding: 1,
        },
      },
      { id: "relu", p: { inplace: true } },
      {
        id: "conv2d",
        p: {
          in_channels: 64,
          out_channels: 64,
          kernel_size: 3,
          stride: 1,
          padding: 1,
        },
      },
      { id: "relu", p: { inplace: true } },
      {
        id: "conv2d",
        p: {
          in_channels: 64,
          out_channels: 2,
          kernel_size: 1,
          stride: 1,
          padding: 0,
        },
      },
    ],
  },
  {
    name: "Transformer",
    desc: "6-layer encoder (Vaswani)",
    size: "65M",
    task: "Seq2Seq",
    acc: "BLEU 27.3",
    year: 2017,
    inputType: "sequence",
    inputCfg: {
      batch: 1,
      channels: 3,
      height: 224,
      width: 224,
      seqLen: 50,
      features: 512,
    },
    color: "#EC4899",
    layers: [
      { id: "embed", p: { num_embeddings: 30000, embedding_dim: 512 } },
      {
        id: "tfenc",
        p: { d_model: 512, nhead: 8, dim_feedforward: 2048, dropout: 0.1 },
      },
      {
        id: "tfenc",
        p: { d_model: 512, nhead: 8, dim_feedforward: 2048, dropout: 0.1 },
      },
      {
        id: "tfenc",
        p: { d_model: 512, nhead: 8, dim_feedforward: 2048, dropout: 0.1 },
      },
      {
        id: "tfenc",
        p: { d_model: 512, nhead: 8, dim_feedforward: 2048, dropout: 0.1 },
      },
      {
        id: "tfenc",
        p: { d_model: 512, nhead: 8, dim_feedforward: 2048, dropout: 0.1 },
      },
      {
        id: "tfenc",
        p: { d_model: 512, nhead: 8, dim_feedforward: 2048, dropout: 0.1 },
      },
      {
        id: "linear",
        p: { in_features: 512, out_features: 30000, bias: true },
      },
    ],
  },
  {
    name: "GPT-2 Small",
    desc: "Autoregressive decoder (simplified)",
    size: "124M",
    task: "Generation",
    acc: "PPL 29.4",
    year: 2019,
    inputType: "sequence",
    inputCfg: {
      batch: 1,
      channels: 3,
      height: 224,
      width: 224,
      seqLen: 128,
      features: 768,
    },
    color: "#F97316",
    layers: [
      { id: "embed", p: { num_embeddings: 50257, embedding_dim: 768 } },
      { id: "ln", p: { normalized_shape: 768 } },
      { id: "mha", p: { embed_dim: 768, num_heads: 12 } },
      { id: "ln", p: { normalized_shape: 768 } },
      { id: "linear", p: { in_features: 768, out_features: 3072, bias: true } },
      { id: "gelu", p: {} },
      { id: "linear", p: { in_features: 3072, out_features: 768, bias: true } },
      { id: "ln", p: { normalized_shape: 768 } },
      { id: "mha", p: { embed_dim: 768, num_heads: 12 } },
      { id: "ln", p: { normalized_shape: 768 } },
      { id: "linear", p: { in_features: 768, out_features: 3072, bias: true } },
      { id: "gelu", p: {} },
      { id: "linear", p: { in_features: 3072, out_features: 768, bias: true } },
      { id: "ln", p: { normalized_shape: 768 } },
      {
        id: "linear",
        p: { in_features: 768, out_features: 50257, bias: true },
      },
    ],
  },
];

const DATASETS = [
  {
    name: "MNIST",
    desc: "Handwritten digits 28×28",
    samples: "70K",
    classes: 10,
    size: "11 MB",
    cmd: "torchvision.datasets.MNIST('data/',download=True)",
  },
  {
    name: "CIFAR-10",
    desc: "10-class color images 32×32",
    samples: "60K",
    classes: 10,
    size: "163 MB",
    cmd: "torchvision.datasets.CIFAR10('data/',download=True)",
  },
  {
    name: "CIFAR-100",
    desc: "100-class color images 32×32",
    samples: "60K",
    classes: 100,
    size: "163 MB",
    cmd: "torchvision.datasets.CIFAR100('data/',download=True)",
  },
  {
    name: "ImageNet",
    desc: "1000-class large scale",
    samples: "1.2M",
    classes: 1000,
    size: "~150 GB",
    cmd: "torchvision.datasets.ImageNet('data/')",
  },
  {
    name: "COCO 2017",
    desc: "Detection & segmentation",
    samples: "330K",
    classes: 80,
    size: "~25 GB",
    cmd: "torchvision.datasets.CocoDetection('data/')",
  },
  {
    name: "Fashion-MNIST",
    desc: "Clothing items 28×28",
    samples: "70K",
    classes: 10,
    size: "30 MB",
    cmd: "torchvision.datasets.FashionMNIST('data/',download=True)",
  },
  {
    name: "VOC 2012",
    desc: "Detection & segmentation",
    samples: "11.5K",
    classes: 20,
    size: "~2 GB",
    cmd: "torchvision.datasets.VOCDetection('data/',download=True)",
  },
  {
    name: "WikiText-2",
    desc: "Language modeling corpus",
    samples: "2M tok",
    classes: "-",
    size: "12 MB",
    cmd: "torchtext.datasets.WikiText2(root='data/')",
  },
  {
    name: "STL-10",
    desc: "96×96 image recognition",
    samples: "113K",
    classes: 10,
    size: "2.6 GB",
    cmd: "torchvision.datasets.STL10('data/',download=True)",
  },
  {
    name: "AG News",
    desc: "News article classification",
    samples: "127K",
    classes: 4,
    size: "30 MB",
    cmd: "torchtext.datasets.AG_NEWS(root='data/')",
  },
];

const catOrder = [
  "conv",
  "act",
  "norm",
  "pool",
  "linear",
  "rnn",
  "attn",
  "reg",
  "util",
  "container",
];
const catIcons = {
  conv: "⊞",
  act: "⚡",
  norm: "≋",
  pool: "▣",
  linear: "→",
  rnn: "↻",
  attn: "◎",
  reg: "✦",
  util: "⚙",
  container: "☰",
};
let uid = 0;
const mkId = () => `l_${++uid}_${Date.now()}`;
const fmtP = (n) =>
  n >= 1e6
    ? (n / 1e6).toFixed(2) + "M"
    : n >= 1e3
      ? (n / 1e3).toFixed(1) + "K"
      : String(n);
const shapeStr = (s) => (s ? `(${s.join(", ")})` : "?");

const themes = {
  dark: {
    bg: "#0f1117",
    bg2: "#161822",
    bg3: "#1e2030",
    border: "#2a2d3a",
    text: "#e2e8f0",
    textMuted: "#94a3b8",
    textDim: "#64748b",
  },
  light: {
    bg: "#f1f5f9",
    bg2: "#ffffff",
    bg3: "#e2e8f0",
    border: "#cbd5e1",
    text: "#1e293b",
    textMuted: "#475569",
    textDim: "#94a3b8",
  },
};

const genPyCode = (network, trainCfg, cls = "MyNetwork") => {
  const inits = network.map((n, i) => {
    const L = LAYERS.find((l) => l.id === n.layerId);
    if (["sequential", "modulelist", "moduledict"].includes(n.layerId))
      return `        # self.${n.layerId}_${i} = nn.${L.name}(...)`;
    const ps = Object.entries(n.params)
      .map(([k, v]) => `${k}=${typeof v === "string" ? `"${v}"` : v}`)
      .join(", ");
    return `        self.layer${i} = nn.${L.name}(${ps})`;
  });
  const fwds = network.map((n, i) =>
    ["sequential", "modulelist", "moduledict"].includes(n.layerId)
      ? `        # x = self.${n.layerId}_${i}(x)`
      : `        x = self.layer${i}(x)`,
  );
  let code = `import torch\nimport torch.nn as nn\nimport torch.optim as optim\nfrom torch.utils.data import DataLoader, random_split\n\n\nclass ${cls}(nn.Module):\n    def __init__(self):\n        super().__init__()\n${inits.join("\n")}\n\n    def forward(self, x):\n${fwds.join("\n")}\n        return x\n\n\ndef train():\n    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")\n    model = ${cls}().to(device)\n    print(model)\n    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")\n\n    criterion = nn.${trainCfg.loss}()\n    optimizer = optim.${trainCfg.optimizer}(model.parameters(), lr=${trainCfg.lr}, weight_decay=${trainCfg.weightDecay})\n`;
  if (trainCfg.scheduler !== "None") {
    const m = {
      StepLR: "optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)",
      CosineAnnealingLR:
        "optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)",
      ReduceLROnPlateau:
        "optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)",
    };
    code += `    scheduler = ${m[trainCfg.scheduler] || "None"}\n`;
  }
  code += `\n    # TODO: Replace with your dataset\n    # dataset = YourDataset(...)\n    # train_size = int(${1 - trainCfg.valSplit} * len(dataset))\n    # val_size = len(dataset) - train_size\n    # train_set, val_set = random_split(dataset, [train_size, val_size])\n    # train_loader = DataLoader(train_set, batch_size=${trainCfg.batchSize}, shuffle=True)\n    # val_loader = DataLoader(val_set, batch_size=${trainCfg.batchSize})\n\n    best_val = float('inf')\n    patience = 0\n\n    for epoch in range(${trainCfg.epochs}):\n        model.train()\n        # for x, y in train_loader:\n        #     x, y = x.to(device), y.to(device)\n        #     optimizer.zero_grad()\n        #     loss = criterion(model(x), y)\n        #     loss.backward()\n        #     optimizer.step()\n\n        # model.eval()\n        # val_loss = 0\n        # with torch.no_grad():\n        #     for x, y in val_loader:\n        #         val_loss += criterion(model(x.to(device)), y.to(device)).item()\n${trainCfg.scheduler !== "None" ? "        # scheduler.step()\n" : ""}\n        # Early stopping\n        # if val_loss < best_val:\n        #     best_val = val_loss; patience = 0\n        #     torch.save(model.state_dict(), "best.pth")\n        # else:\n        #     patience += 1\n        #     if patience >= ${trainCfg.earlyStop}: break\n\n\nif __name__ == "__main__":\n    train()\n`;
  return code;
};
const genIpynb = (c) =>
  JSON.stringify(
    {
      nbformat: 4,
      nbformat_minor: 5,
      metadata: {
        kernelspec: {
          display_name: "Python 3",
          language: "python",
          name: "python3",
        },
      },
      cells: [
        {
          cell_type: "markdown",
          metadata: {},
          source: ["# PyTorch Network\n"],
        },
        {
          cell_type: "code",
          execution_count: null,
          metadata: {},
          outputs: [],
          source: ["!pip install torch torchvision -q"],
        },
        {
          cell_type: "code",
          execution_count: null,
          metadata: {},
          outputs: [],
          source: [c],
        },
      ],
    },
    null,
    2,
  );
const downloadFile = (c, n, tp = "text/plain") => {
  const u = URL.createObjectURL(new Blob([c], { type: tp }));
  const a = document.createElement("a");
  a.href = u;
  a.download = n;
  a.click();
  URL.revokeObjectURL(u);
};

export default function App() {
  const [lang, setLang] = useState("zh");
  const [tab, setTab] = useState("builder");
  const [network, setNetwork] = useState([]);
  const [selIdx, setSelIdx] = useState(-1);
  const [dragLayer, setDragLayer] = useState(null);
  const [dragModel, setDragModel] = useState(null);
  const [search, setSearch] = useState("");
  const [toast, setToast] = useState("");
  const [catFilter, setCatFilter] = useState("all");
  const [sideTab, setSideTab] = useState("layers");
  const [myModels, setMyModels] = useState([]);
  const fileRef = useRef();
  const t = T[lang];

  const [inputType, setInputType] = useState("image");
  const [inputCfg, setInputCfg] = useState({
    batch: 1,
    channels: 3,
    height: 224,
    width: 224,
    seqLen: 50,
    features: 128,
  });
  const [trainCfg, setTrainCfg] = useState({
    optimizer: "Adam",
    lr: 0.001,
    weightDecay: 1e-4,
    scheduler: "None",
    loss: "CrossEntropyLoss",
    epochs: 50,
    batchSize: 32,
    valSplit: 0.2,
    earlyStop: 10,
  });
  const [uiCfg, setUiCfg] = useState({
    theme: "dark",
    fontSize: 13,
    bgImage: "https://invoker-pray.github.io/img/cover/toysandtools-cover.jpg",
    bgOpacity: 0.08,
    accent: "#3B82F6",
  });
  const th = themes[uiCfg.theme];

  useEffect(() => {
    const id = "no-spin";
    if (!document.getElementById(id)) {
      const st = document.createElement("style");
      st.id = id;
      st.textContent =
        "input[type=number]::-webkit-outer-spin-button,input[type=number]::-webkit-inner-spin-button{-webkit-appearance:none;margin:0}input[type=number]{-moz-appearance:textfield;appearance:textfield}";
      document.head.appendChild(st);
    }
  }, []);

  const showToast = (m) => {
    setToast(m);
    setTimeout(() => setToast(""), 2500);
  };

  const shapes = useMemo(() => {
    let s;
    if (inputType === "image")
      s = [inputCfg.batch, inputCfg.channels, inputCfg.height, inputCfg.width];
    else if (inputType === "sequence")
      s = [inputCfg.seqLen, inputCfg.batch, inputCfg.features];
    else if (inputType === "tabular") s = [inputCfg.batch, inputCfg.features];
    else s = [inputCfg.batch, inputCfg.seqLen];
    const r = [s];
    for (const n of network) {
      s = calcShape(n.layerId, n.params, s);
      r.push(s);
    }
    return r;
  }, [network, inputType, inputCfg]);

  const addLayer = useCallback((lid) => {
    const L = LAYERS.find((l) => l.id === lid);
    setNetwork((p) => [
      ...p,
      { uid: mkId(), layerId: lid, params: { ...L.params } },
    ]);
  }, []);

  const loadModelToCanvas = (model) => {
    const layers = model.layers.map((item) => ({
      uid: mkId(),
      layerId: item.id,
      params: { ...item.p },
    }));
    setNetwork(layers);
    setSelIdx(-1);
    if (model.inputType) setInputType(model.inputType);
    if (model.inputCfg) setInputCfg((p) => ({ ...p, ...model.inputCfg }));
    setTab("builder");
    showToast(
      `${model.name} ${lang === "zh" ? "已加载到画布" : "loaded to canvas"}!`,
    );
  };

  const addToLibrary = (model) => {
    if (myModels.find((m) => m.name === model.name)) return;
    setMyModels((p) => [...p, model]);
    showToast(
      `${model.name} ${lang === "zh" ? "已添加到模型库" : "added to library"}!`,
    );
  };

  const removeLayer = (i) => {
    setNetwork((p) => p.filter((_, j) => j !== i));
    setSelIdx(-1);
  };
  const moveLayer = (i, d) => {
    setNetwork((p) => {
      const n = [...p];
      const t2 = i + d;
      if (t2 < 0 || t2 >= n.length) return n;
      [n[i], n[t2]] = [n[t2], n[i]];
      setSelIdx(t2);
      return n;
    });
  };
  const updateParam = (i, k, v) => {
    setNetwork((p) => {
      const n = [...p];
      n[i] = { ...n[i], params: { ...n[i].params, [k]: v } };
      return n;
    });
  };

  const handleExport = (type) => {
    if (!network.length) return;
    const code = genPyCode(network, trainCfg);
    if (type === "py") downloadFile(code, "model.py", "text/x-python");
    else downloadFile(genIpynb(code), "model.ipynb", "application/json");
    showToast(t.exportSuccess);
  };

  const handleImport = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      let text = ev.target.result;
      if (file.name.endsWith(".ipynb")) {
        try {
          const nb = JSON.parse(text);
          text = nb.cells
            .filter((c) => c.cell_type === "code")
            .map((c) =>
              Array.isArray(c.source) ? c.source.join("") : c.source,
            )
            .join("\n");
        } catch {
          return;
        }
      }
      const parsed = [],
        re = /nn\.(\w+)\(([^)]*)\)/g;
      let m;
      while ((m = re.exec(text)) !== null) {
        const L = LAYERS.find((l) => l.name === m[1]);
        if (L) {
          const p = { ...L.params };
          const args = m[2]
            .split(",")
            .map((s) => s.trim())
            .filter(Boolean);
          const keys = Object.keys(p);
          args.forEach((v, i) => {
            if (v.includes("=")) {
              const [k, val] = v.split("=");
              if (k.trim() in p)
                p[k.trim()] = isNaN(val)
                  ? val.replace(/['"]/g, "")
                  : Number(val);
            } else if (i < keys.length)
              p[keys[i]] = isNaN(v) ? v.replace(/['"]/g, "") : Number(v);
          });
          parsed.push({ uid: mkId(), layerId: L.id, params: p });
        }
      }
      if (parsed.length) {
        setNetwork(parsed);
        setSelIdx(-1);
        setTab("builder");
        showToast(t.importSuccess);
      }
    };
    reader.readAsText(file);
    e.target.value = "";
  };

  const totalParams = network.reduce((s, n) => {
    const L = LAYERS.find((l) => l.id === n.layerId);
    return s + (L.pCount ? L.pCount(n.params) : 0);
  }, 0);
  const filteredLayers = LAYERS.filter(
    (l) =>
      (catFilter === "all" || l.cat === catFilter) &&
      (!search || l.name.toLowerCase().includes(search.toLowerCase())),
  );
  const sel = selIdx >= 0 && selIdx < network.length ? network[selIdx] : null;
  const selDef = sel ? LAYERS.find((l) => l.id === sel.layerId) : null;

  const accent = uiCfg.accent;
  const fs = uiCfg.fontSize;
  const S = {
    input: {
      width: "100%",
      padding: "4px 8px",
      borderRadius: 4,
      border: `1px solid ${th.border}`,
      background: th.bg3,
      color: th.text,
      fontSize: fs - 2,
      outline: "none",
      boxSizing: "border-box",
    },
    btn: (bg = th.bg3, c = th.text) => ({
      padding: "4px 10px",
      borderRadius: 6,
      border: "none",
      cursor: "pointer",
      fontSize: fs - 2,
      fontWeight: 500,
      background: bg,
      color: c,
      transition: "all .15s",
      whiteSpace: "nowrap",
    }),
    tag: (bg) => ({
      display: "inline-block",
      padding: "1px 5px",
      borderRadius: 3,
      fontSize: fs - 4,
      fontWeight: 600,
      background: `${bg}22`,
      color: bg,
    }),
    shapeTag: {
      fontSize: fs - 3,
      padding: "2px 6px",
      borderRadius: 4,
      background: `${accent}18`,
      color: accent,
      fontFamily: "'Courier New',monospace",
      fontWeight: 700,
      display: "inline-block",
      letterSpacing: 0.3,
    },
    card: {
      background: th.bg2,
      border: `1px solid ${th.border}`,
      borderRadius: 10,
      padding: 14,
    },
  };
  const Field = ({ label, children }) => (
    <div style={{ marginBottom: 8 }}>
      <label
        style={{
          fontSize: fs - 3,
          color: th.textDim,
          display: "block",
          marginBottom: 2,
        }}
      >
        {label}
      </label>
      {children}
    </div>
  );
  const Sel = ({ value, onChange, options }) => (
    <select
      style={{ ...S.input, cursor: "pointer" }}
      value={value}
      onChange={(e) => onChange(e.target.value)}
    >
      {options.map((o) => (
        <option key={o} value={o}>
          {o}
        </option>
      ))}
    </select>
  );

  return (
    <div
      style={{
        fontFamily: "'Inter',system-ui,sans-serif",
        height: "100vh",
        display: "flex",
        flexDirection: "column",
        background: th.bg,
        color: th.text,
        overflow: "hidden",
        fontSize: fs,
        position: "relative",
      }}
    >
      {/* BG */}
      {uiCfg.bgImage && (
        <div
          style={{
            position: "absolute",
            inset: 0,
            backgroundImage: `url(${uiCfg.bgImage})`,
            backgroundSize: "cover",
            backgroundPosition: "center",
            opacity: uiCfg.bgOpacity,
            pointerEvents: "none",
            zIndex: 0,
          }}
        />
      )}
      {toast && (
        <div
          style={{
            position: "fixed",
            top: 52,
            right: 16,
            background: "#10B981",
            color: "#fff",
            padding: "7px 16px",
            borderRadius: 8,
            fontSize: fs - 1,
            fontWeight: 500,
            zIndex: 1000,
            boxShadow: "0 4px 20px #0006",
          }}
        >
          {toast}
        </div>
      )}

      {/* HEADER */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "6px 14px",
          background: `${th.bg2}ee`,
          borderBottom: `1px solid ${th.border}`,
          flexShrink: 0,
          minHeight: 44,
          zIndex: 2,
          gap: 8,
          flexWrap: "wrap",
          backdropFilter: "blur(12px)",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <div
            style={{
              width: 28,
              height: 28,
              borderRadius: 6,
              background: `linear-gradient(135deg,${accent},#8B5CF6)`,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: 12,
              fontWeight: 800,
              color: "#fff",
            }}
          >
            PT
          </div>
          <span style={{ fontWeight: 700, fontSize: fs + 2 }}>{t.title}</span>
        </div>
        <div
          style={{
            display: "flex",
            gap: 2,
            background: `${th.bg3}88`,
            borderRadius: 8,
            padding: 2,
            backdropFilter: "blur(8px)",
          }}
        >
          {[
            ["builder", t.builder],
            ["models", t.modelStore],
            ["datasets", t.datasetStore],
            ["training", t.training],
            ["ui", t.uiSettings],
          ].map(([k, v]) => (
            <button
              key={k}
              style={{
                padding: "5px 12px",
                borderRadius: 6,
                border: "none",
                cursor: "pointer",
                fontSize: fs - 1,
                fontWeight: 600,
                background: tab === k ? accent : "transparent",
                color: tab === k ? "#fff" : th.textMuted,
                transition: "all .2s",
              }}
              onClick={() => setTab(k)}
            >
              {v}
            </button>
          ))}
        </div>
        <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
          <button style={S.btn()} onClick={() => fileRef.current?.click()}>
            {t.importFile}
          </button>
          <input
            ref={fileRef}
            type="file"
            accept=".py,.ipynb"
            style={{ display: "none" }}
            onChange={handleImport}
          />
          <button
            style={S.btn(accent, "#fff")}
            onClick={() => handleExport("py")}
          >
            {t.exportPy}
          </button>
          <button
            style={S.btn("#7c3aed", "#fff")}
            onClick={() => handleExport("ipynb")}
          >
            {t.exportIpynb}
          </button>
          <button
            style={{
              ...S.btn(),
              fontWeight: 700,
              fontSize: fs,
              minWidth: 32,
              textAlign: "center",
            }}
            onClick={() => setLang((l) => (l === "zh" ? "en" : "zh"))}
          >
            {lang === "zh" ? "EN" : "中"}
          </button>
        </div>
      </div>

      {/* ═══ BUILDER ═══ */}
      {tab === "builder" && (
        <div
          style={{ display: "flex", flex: 1, overflow: "hidden", zIndex: 1 }}
        >
          {/* LEFT SIDEBAR */}
          <div
            style={{
              width: 224,
              background: `${th.bg2}dd`,
              borderRight: `1px solid ${th.border}`,
              display: "flex",
              flexDirection: "column",
              flexShrink: 0,
              backdropFilter: "blur(10px)",
            }}
          >
            {/* sidebar tabs */}
            <div
              style={{
                display: "flex",
                borderBottom: `1px solid ${th.border}`,
              }}
            >
              {["layers", "models"].map((st) => (
                <button
                  key={st}
                  onClick={() => setSideTab(st)}
                  style={{
                    flex: 1,
                    padding: "8px 0",
                    border: "none",
                    cursor: "pointer",
                    fontSize: fs - 1,
                    fontWeight: 600,
                    background: sideTab === st ? `${accent}22` : "transparent",
                    color: sideTab === st ? accent : th.textMuted,
                    borderBottom:
                      sideTab === st
                        ? `2px solid ${accent}`
                        : "2px solid transparent",
                    transition: "all .2s",
                  }}
                >
                  {st === "layers" ? t.layersTab : t.modelsTab}
                </button>
              ))}
            </div>

            {sideTab === "layers" && (
              <>
                <div style={{ padding: "8px 8px 4px" }}>
                  <input
                    style={S.input}
                    placeholder={t.search}
                    value={search}
                    onChange={(e) => setSearch(e.target.value)}
                  />
                </div>
                <div
                  style={{
                    display: "flex",
                    flexWrap: "wrap",
                    gap: 2,
                    padding: "4px 8px 6px",
                    borderBottom: `1px solid ${th.border}`,
                  }}
                >
                  <button
                    style={{
                      ...S.btn(
                        catFilter === "all" ? accent : th.bg3,
                        catFilter === "all" ? "#fff" : th.textMuted,
                      ),
                      fontSize: fs - 4,
                      padding: "2px 5px",
                    }}
                    onClick={() => setCatFilter("all")}
                  >
                    All
                  </button>
                  {catOrder.map((c) => (
                    <button
                      key={c}
                      style={{
                        ...S.btn(
                          catFilter === c ? accent : th.bg3,
                          catFilter === c ? "#fff" : th.textMuted,
                        ),
                        fontSize: fs - 4,
                        padding: "2px 5px",
                      }}
                      onClick={() => setCatFilter(c)}
                    >
                      {catIcons[c]} {t.categories[c]}
                    </button>
                  ))}
                </div>
                <div style={{ flex: 1, overflow: "auto", padding: 5 }}>
                  {filteredLayers.map((L) => (
                    <div
                      key={L.id}
                      draggable
                      onDragStart={() => setDragLayer(L.id)}
                      onDragEnd={() => setDragLayer(null)}
                      onDoubleClick={() => addLayer(L.id)}
                      style={{
                        padding: "5px 8px",
                        marginBottom: 3,
                        borderRadius: 6,
                        background: `${th.bg3}cc`,
                        border: `1px solid ${th.border}`,
                        cursor: "grab",
                        borderLeft: `3px solid ${L.color}`,
                        fontSize: fs - 2,
                      }}
                    >
                      <div
                        style={{
                          display: "flex",
                          justifyContent: "space-between",
                          alignItems: "center",
                        }}
                      >
                        <span style={{ fontWeight: 600 }}>{L.name}</span>
                        <span style={S.tag(L.color)}>
                          {t.categories[L.cat]}
                        </span>
                      </div>
                      <div
                        style={{
                          color: th.textDim,
                          fontSize: fs - 3,
                          marginTop: 1,
                        }}
                      >
                        {L.inF} → {L.outF}
                      </div>
                    </div>
                  ))}
                </div>
              </>
            )}

            {sideTab === "models" && (
              <div style={{ flex: 1, overflow: "auto", padding: 6 }}>
                {myModels.length === 0 ? (
                  <div
                    style={{
                      textAlign: "center",
                      color: th.textDim,
                      marginTop: 40,
                      fontSize: fs - 2,
                      padding: "0 10px",
                      lineHeight: 1.6,
                    }}
                  >
                    {t.noModels}
                  </div>
                ) : (
                  myModels.map((m) => (
                    <div
                      key={m.name}
                      draggable
                      onDragStart={() => setDragModel(m.name)}
                      onDragEnd={() => setDragModel(null)}
                      onDoubleClick={() => loadModelToCanvas(m)}
                      style={{
                        padding: "8px 10px",
                        marginBottom: 4,
                        borderRadius: 8,
                        background: `${th.bg3}cc`,
                        border: `1px solid ${th.border}`,
                        cursor: "grab",
                        borderLeft: `3px solid ${m.color || accent}`,
                        transition: "all .15s",
                      }}
                    >
                      <div
                        style={{
                          fontWeight: 700,
                          fontSize: fs,
                          color: th.text,
                        }}
                      >
                        {m.name}
                      </div>
                      <div
                        style={{
                          fontSize: fs - 3,
                          color: th.textDim,
                          marginTop: 2,
                        }}
                      >
                        {m.desc}
                      </div>
                      <div
                        style={{
                          display: "flex",
                          gap: 4,
                          marginTop: 4,
                          flexWrap: "wrap",
                        }}
                      >
                        <span style={S.tag(m.color || accent)}>
                          {m.layers.length} {t.layerCount}
                        </span>
                        <span style={S.tag("#F59E0B")}>{m.size}</span>
                        <span style={S.tag("#10B981")}>{m.task}</span>
                      </div>
                      <div
                        style={{
                          fontSize: fs - 4,
                          color: th.textDim,
                          marginTop: 4,
                          fontStyle: "italic",
                        }}
                      >
                        {t.useModel}
                      </div>
                    </div>
                  ))
                )}
              </div>
            )}
          </div>

          {/* CENTER CANVAS */}
          <div
            style={{
              flex: 1,
              display: "flex",
              flexDirection: "column",
              overflow: "hidden",
            }}
          >
            {/* Input bar */}
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
                padding: "6px 12px",
                borderBottom: `1px solid ${th.border}`,
                background: `${th.bg2}dd`,
                flexWrap: "wrap",
                backdropFilter: "blur(8px)",
              }}
            >
              <span
                style={{
                  fontSize: fs - 2,
                  color: th.textMuted,
                  fontWeight: 600,
                }}
              >
                {t.inputShape}:
              </span>
              <select
                style={{ ...S.input, width: "auto" }}
                value={inputType}
                onChange={(e) => setInputType(e.target.value)}
              >
                {["image", "sequence", "tabular", "text"].map((v) => (
                  <option key={v} value={v}>
                    {t[v]}
                  </option>
                ))}
              </select>
              <label style={{ fontSize: fs - 3, color: th.textDim }}>N=</label>
              <input
                type="number"
                style={{ ...S.input, width: 44 }}
                value={inputCfg.batch}
                onChange={(e) =>
                  setInputCfg((p) => ({ ...p, batch: +e.target.value || 1 }))
                }
              />
              {inputType === "image" && (
                <>
                  <label style={{ fontSize: fs - 3, color: th.textDim }}>
                    C=
                  </label>
                  <input
                    type="number"
                    style={{ ...S.input, width: 38 }}
                    value={inputCfg.channels}
                    onChange={(e) =>
                      setInputCfg((p) => ({
                        ...p,
                        channels: +e.target.value || 1,
                      }))
                    }
                  />
                  <label style={{ fontSize: fs - 3, color: th.textDim }}>
                    H=
                  </label>
                  <input
                    type="number"
                    style={{ ...S.input, width: 44 }}
                    value={inputCfg.height}
                    onChange={(e) =>
                      setInputCfg((p) => ({
                        ...p,
                        height: +e.target.value || 1,
                      }))
                    }
                  />
                  <label style={{ fontSize: fs - 3, color: th.textDim }}>
                    W=
                  </label>
                  <input
                    type="number"
                    style={{ ...S.input, width: 44 }}
                    value={inputCfg.width}
                    onChange={(e) =>
                      setInputCfg((p) => ({
                        ...p,
                        width: +e.target.value || 1,
                      }))
                    }
                  />
                </>
              )}
              {(inputType === "sequence" || inputType === "text") && (
                <>
                  <label style={{ fontSize: fs - 3, color: th.textDim }}>
                    Seq=
                  </label>
                  <input
                    type="number"
                    style={{ ...S.input, width: 44 }}
                    value={inputCfg.seqLen}
                    onChange={(e) =>
                      setInputCfg((p) => ({
                        ...p,
                        seqLen: +e.target.value || 1,
                      }))
                    }
                  />
                  <label style={{ fontSize: fs - 3, color: th.textDim }}>
                    F=
                  </label>
                  <input
                    type="number"
                    style={{ ...S.input, width: 44 }}
                    value={inputCfg.features}
                    onChange={(e) =>
                      setInputCfg((p) => ({
                        ...p,
                        features: +e.target.value || 1,
                      }))
                    }
                  />
                </>
              )}
              {inputType === "tabular" && (
                <>
                  <label style={{ fontSize: fs - 3, color: th.textDim }}>
                    F=
                  </label>
                  <input
                    type="number"
                    style={{ ...S.input, width: 52 }}
                    value={inputCfg.features}
                    onChange={(e) =>
                      setInputCfg((p) => ({
                        ...p,
                        features: +e.target.value || 1,
                      }))
                    }
                  />
                </>
              )}
              <span style={S.shapeTag}>{shapeStr(shapes[0])}</span>
              <div
                style={{
                  marginLeft: "auto",
                  display: "flex",
                  gap: 10,
                  fontSize: fs - 2,
                  color: th.textMuted,
                }}
              >
                <span>
                  {t.totalLayers}:{" "}
                  <b style={{ color: th.text }}>{network.length}</b>
                </span>
                <span>
                  {t.totalParams}:{" "}
                  <b style={{ color: th.text }}>{fmtP(totalParams)}</b>
                </span>
              </div>
              <button
                style={S.btn(`${accent}22`, "#ef4444")}
                onClick={() => {
                  setNetwork([]);
                  setSelIdx(-1);
                }}
              >
                {t.clear}
              </button>
            </div>

            {/* Canvas body */}
            <div
              style={{ flex: 1, overflow: "auto", padding: 16 }}
              onDragOver={(e) => e.preventDefault()}
              onDrop={() => {
                if (dragLayer) {
                  addLayer(dragLayer);
                  setDragLayer(null);
                }
                if (dragModel) {
                  const m = myModels.find((x) => x.name === dragModel);
                  if (m) loadModelToCanvas(m);
                  setDragModel(null);
                }
              }}
            >
              {network.length === 0 ? (
                <div
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    justifyContent: "center",
                    height: "100%",
                    color: th.textDim,
                    gap: 6,
                  }}
                >
                  <div style={{ fontSize: 48, opacity: 0.2 }}>⊞</div>
                  <div style={{ fontSize: fs }}>{t.emptyCanvas}</div>
                  <div style={{ fontSize: fs - 3 }}>
                    Drag & drop / double-click
                  </div>
                </div>
              ) : (
                <div style={{ maxWidth: 560, margin: "0 auto" }}>
                  <div style={{ textAlign: "center", marginBottom: 2 }}>
                    <span style={{ ...S.shapeTag, fontSize: fs - 2 }}>
                      Input: {shapeStr(shapes[0])}
                    </span>
                  </div>
                  {network.map((node, idx) => {
                    const L = LAYERS.find((l) => l.id === node.layerId);
                    const pc = L.pCount ? L.pCount(node.params) : 0;
                    const isSel = selIdx === idx;
                    const outShape = shapes[idx + 1];
                    return (
                      <div key={node.uid}>
                        <div
                          style={{
                            display: "flex",
                            flexDirection: "column",
                            alignItems: "center",
                          }}
                        >
                          <div
                            style={{
                              width: 2,
                              height: 12,
                              background: th.border,
                            }}
                          />
                          <div
                            style={{
                              fontSize: fs - 5,
                              color: th.textDim,
                              lineHeight: 1,
                            }}
                          >
                            ▼
                          </div>
                        </div>
                        <div
                          onClick={() => setSelIdx(idx)}
                          style={{
                            display: "flex",
                            alignItems: "stretch",
                            padding: "8px 12px",
                            borderRadius: 8,
                            background: isSel ? `${th.bg3}ee` : `${th.bg2}cc`,
                            border: `1px solid ${isSel ? L.color : th.border}`,
                            cursor: "pointer",
                            transition: "all .15s",
                            boxShadow: isSel ? `0 0 16px ${L.color}33` : "none",
                            backdropFilter: "blur(6px)",
                          }}
                        >
                          <div
                            style={{
                              width: 4,
                              borderRadius: 2,
                              background: L.color,
                              marginRight: 10,
                              flexShrink: 0,
                            }}
                          />
                          <div style={{ flex: 1, minWidth: 0 }}>
                            <div
                              style={{
                                display: "flex",
                                justifyContent: "space-between",
                                alignItems: "center",
                              }}
                            >
                              <span style={{ fontWeight: 700, fontSize: fs }}>
                                {L.name}
                              </span>
                              <div
                                style={{
                                  display: "flex",
                                  gap: 4,
                                  alignItems: "center",
                                }}
                              >
                                {pc > 0 && (
                                  <span
                                    style={{
                                      fontSize: fs - 4,
                                      color: th.textMuted,
                                    }}
                                  >
                                    {fmtP(pc)}
                                  </span>
                                )}
                                <span style={S.tag(L.color)}>{idx}</span>
                              </div>
                            </div>
                            <div
                              style={{
                                display: "flex",
                                flexWrap: "wrap",
                                gap: 3,
                                marginTop: 3,
                              }}
                            >
                              {Object.entries(node.params).map(([k, v]) => (
                                <span
                                  key={k}
                                  style={{
                                    fontSize: fs - 4,
                                    color: th.textDim,
                                    background: `${th.border}66`,
                                    padding: "0 5px",
                                    borderRadius: 3,
                                  }}
                                >
                                  {k}={String(v)}
                                </span>
                              ))}
                            </div>
                            <div
                              style={{
                                marginTop: 4,
                                display: "flex",
                                alignItems: "center",
                                gap: 6,
                              }}
                            >
                              <span
                                style={{ fontSize: fs - 4, color: th.textDim }}
                              >
                                {t.shapeAfter}:
                              </span>
                              <span style={S.shapeTag}>
                                {shapeStr(outShape)}
                              </span>
                            </div>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                  <div
                    style={{
                      display: "flex",
                      flexDirection: "column",
                      alignItems: "center",
                      marginTop: 2,
                    }}
                  >
                    <div
                      style={{ width: 2, height: 12, background: th.border }}
                    />
                    <span style={{ ...S.shapeTag, fontSize: fs - 2 }}>
                      Output: {shapeStr(shapes[shapes.length - 1])}
                    </span>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* RIGHT PROPS */}
          <div
            style={{
              width: 224,
              background: `${th.bg2}dd`,
              borderLeft: `1px solid ${th.border}`,
              display: "flex",
              flexDirection: "column",
              flexShrink: 0,
              overflow: "auto",
              backdropFilter: "blur(10px)",
            }}
          >
            <div
              style={{
                padding: "8px 12px",
                borderBottom: `1px solid ${th.border}`,
                fontWeight: 600,
                fontSize: fs,
              }}
            >
              {t.props}
            </div>
            <div style={{ padding: 10, flex: 1 }}>
              {sel && selDef ? (
                <div>
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: 8,
                      marginBottom: 10,
                    }}
                  >
                    <div
                      style={{
                        width: 4,
                        height: 22,
                        borderRadius: 2,
                        background: selDef.color,
                      }}
                    />
                    <div>
                      <div style={{ fontWeight: 700, fontSize: fs + 2 }}>
                        {selDef.name}
                      </div>
                      <div style={{ fontSize: fs - 3, color: th.textDim }}>
                        Layer {selIdx}
                      </div>
                    </div>
                  </div>
                  <div
                    style={{
                      background: `${th.bg3}aa`,
                      borderRadius: 6,
                      padding: 8,
                      marginBottom: 10,
                      fontSize: fs - 2,
                    }}
                  >
                    <div style={{ color: th.textDim }}>
                      {t.in}:{" "}
                      <span style={{ color: th.textMuted }}>{selDef.inF}</span>
                    </div>
                    <div style={{ color: th.textDim }}>
                      {t.out}:{" "}
                      <span style={{ color: th.textMuted }}>{selDef.outF}</span>
                    </div>
                    <div style={{ marginTop: 4 }}>
                      <span style={{ color: th.textDim }}>
                        {t.shapeAfter}:{" "}
                      </span>
                      <span style={S.shapeTag}>
                        {shapeStr(shapes[selIdx + 1])}
                      </span>
                    </div>
                  </div>
                  {Object.entries(sel.params).map(([k, v]) => (
                    <Field key={k} label={k}>
                      {typeof v === "boolean" ? (
                        <button
                          style={{
                            ...S.btn(v ? accent : th.bg3, v ? "#fff" : th.text),
                            width: "100%",
                            textAlign: "center",
                          }}
                          onClick={() => updateParam(selIdx, k, !v)}
                        >
                          {String(v)}
                        </button>
                      ) : typeof v === "string" ? (
                        <input
                          style={S.input}
                          value={v}
                          onChange={(e) =>
                            updateParam(selIdx, k, e.target.value)
                          }
                        />
                      ) : (
                        <input
                          style={S.input}
                          type="number"
                          value={v}
                          onChange={(e) =>
                            updateParam(selIdx, k, Number(e.target.value) || 0)
                          }
                        />
                      )}
                    </Field>
                  ))}
                  <div style={{ display: "flex", gap: 4, marginTop: 10 }}>
                    <button
                      style={{ ...S.btn(), flex: 1 }}
                      onClick={() => moveLayer(selIdx, -1)}
                    >
                      {t.moveUp}
                    </button>
                    <button
                      style={{ ...S.btn(), flex: 1 }}
                      onClick={() => moveLayer(selIdx, 1)}
                    >
                      {t.moveDown}
                    </button>
                  </div>
                  <button
                    style={{
                      ...S.btn("#dc262622", "#ef4444"),
                      width: "100%",
                      marginTop: 6,
                    }}
                    onClick={() => removeLayer(selIdx)}
                  >
                    {t.delete}
                  </button>
                </div>
              ) : (
                <div
                  style={{
                    textAlign: "center",
                    color: th.textDim,
                    marginTop: 40,
                    fontSize: fs - 1,
                  }}
                >
                  {t.selectLayer}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ═══ MODEL STORE ═══ */}
      {tab === "models" && (
        <div style={{ flex: 1, overflow: "auto", padding: 20, zIndex: 1 }}>
          <div style={{ maxWidth: 900, margin: "0 auto" }}>
            <p
              style={{ color: th.textDim, fontSize: fs - 1, marginBottom: 14 }}
            >
              {t.modelDesc}
            </p>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fill,minmax(260,1fr))",
                gap: 12,
              }}
            >
              {MODELS_DB.map((m) => {
                const inLib = myModels.find((x) => x.name === m.name);
                return (
                  <div
                    key={m.name}
                    style={{
                      ...S.card,
                      borderLeft: `3px solid ${m.color || accent}`,
                    }}
                  >
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "start",
                        marginBottom: 6,
                      }}
                    >
                      <div>
                        <div style={{ fontWeight: 700, fontSize: fs + 2 }}>
                          {m.name}
                        </div>
                        <div style={{ fontSize: fs - 4, color: th.textDim }}>
                          {m.year}
                        </div>
                      </div>
                      <span style={{ ...S.tag(accent), fontSize: fs - 3 }}>
                        {m.size}
                      </span>
                    </div>
                    <p
                      style={{
                        fontSize: fs - 2,
                        color: th.textMuted,
                        margin: "0 0 8px",
                      }}
                    >
                      {m.desc}
                    </p>
                    <div
                      style={{
                        display: "flex",
                        flexWrap: "wrap",
                        gap: 4,
                        marginBottom: 10,
                      }}
                    >
                      <span style={S.tag("#10B981")}>{m.task}</span>
                      <span style={S.tag("#F59E0B")}>{m.acc}</span>
                      <span style={S.tag("#8B5CF6")}>
                        {m.layers.length} layers
                      </span>
                    </div>
                    <div style={{ display: "flex", gap: 6 }}>
                      <button
                        style={{
                          ...S.btn(inLib ? "#10B981" : accent, "#fff"),
                          flex: 1,
                          textAlign: "center",
                          padding: "6px 0",
                        }}
                        onClick={() => addToLibrary(m)}
                      >
                        {inLib ? `✓ ${t.loaded}` : t.loadModel}
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* ═══ DATASET STORE ═══ */}
      {tab === "datasets" && (
        <div style={{ flex: 1, overflow: "auto", padding: 20, zIndex: 1 }}>
          <div style={{ maxWidth: 900, margin: "0 auto" }}>
            <p
              style={{ color: th.textDim, fontSize: fs - 1, marginBottom: 14 }}
            >
              {t.datasetDesc}
            </p>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fill,minmax(260,1fr))",
                gap: 12,
              }}
            >
              {DATASETS.map((d) => (
                <div key={d.name} style={S.card}>
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      marginBottom: 4,
                    }}
                  >
                    <span style={{ fontWeight: 700, fontSize: fs + 1 }}>
                      {d.name}
                    </span>
                    <span style={{ ...S.tag("#F59E0B"), fontSize: fs - 3 }}>
                      {d.size}
                    </span>
                  </div>
                  <p
                    style={{
                      fontSize: fs - 2,
                      color: th.textMuted,
                      margin: "0 0 8px",
                    }}
                  >
                    {d.desc}
                  </p>
                  <div style={{ display: "flex", gap: 6, marginBottom: 8 }}>
                    <span style={S.tag(accent)}>
                      {t.samples}: {d.samples}
                    </span>
                    <span style={S.tag("#8B5CF6")}>
                      {t.classes}: {d.classes}
                    </span>
                  </div>
                  <div
                    style={{
                      background: th.bg3,
                      borderRadius: 6,
                      padding: "6px 8px",
                      fontSize: fs - 3,
                      fontFamily: "monospace",
                      color: "#10B981",
                      wordBreak: "break-all",
                    }}
                  >
                    {d.cmd}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* ═══ TRAINING ═══ */}
      {tab === "training" && (
        <div style={{ flex: 1, overflow: "auto", padding: 20, zIndex: 1 }}>
          <div style={{ maxWidth: 600, margin: "0 auto" }}>
            <div
              style={{ fontWeight: 700, fontSize: fs + 1, marginBottom: 12 }}
            >
              {t.trainTitle}
            </div>
            <div style={{ ...S.card, marginBottom: 12 }}>
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "1fr 1fr",
                  gap: 12,
                }}
              >
                <Field label={t.optimizer}>
                  <Sel
                    value={trainCfg.optimizer}
                    onChange={(v) =>
                      setTrainCfg((p) => ({ ...p, optimizer: v }))
                    }
                    options={["Adam", "AdamW", "SGD", "RMSprop", "Adagrad"]}
                  />
                </Field>
                <Field label={t.lr}>
                  <input
                    style={S.input}
                    type="number"
                    step="0.0001"
                    value={trainCfg.lr}
                    onChange={(e) =>
                      setTrainCfg((p) => ({ ...p, lr: +e.target.value }))
                    }
                  />
                </Field>
                <Field label={t.weightDecay}>
                  <input
                    style={S.input}
                    type="number"
                    step="0.0001"
                    value={trainCfg.weightDecay}
                    onChange={(e) =>
                      setTrainCfg((p) => ({
                        ...p,
                        weightDecay: +e.target.value,
                      }))
                    }
                  />
                </Field>
                <Field label={t.scheduler}>
                  <Sel
                    value={trainCfg.scheduler}
                    onChange={(v) =>
                      setTrainCfg((p) => ({ ...p, scheduler: v }))
                    }
                    options={[
                      "None",
                      "StepLR",
                      "CosineAnnealingLR",
                      "ReduceLROnPlateau",
                    ]}
                  />
                </Field>
                <Field label={t.lossFunc}>
                  <Sel
                    value={trainCfg.loss}
                    onChange={(v) => setTrainCfg((p) => ({ ...p, loss: v }))}
                    options={[
                      "CrossEntropyLoss",
                      "MSELoss",
                      "BCELoss",
                      "BCEWithLogitsLoss",
                      "L1Loss",
                      "SmoothL1Loss",
                      "NLLLoss",
                      "KLDivLoss",
                    ]}
                  />
                </Field>
                <Field label={t.epochs}>
                  <input
                    style={S.input}
                    type="number"
                    value={trainCfg.epochs}
                    onChange={(e) =>
                      setTrainCfg((p) => ({
                        ...p,
                        epochs: +e.target.value || 1,
                      }))
                    }
                  />
                </Field>
                <Field label={t.batchSizeTrain}>
                  <input
                    style={S.input}
                    type="number"
                    value={trainCfg.batchSize}
                    onChange={(e) =>
                      setTrainCfg((p) => ({
                        ...p,
                        batchSize: +e.target.value || 1,
                      }))
                    }
                  />
                </Field>
                <Field label={t.valSplit}>
                  <input
                    style={S.input}
                    type="number"
                    step="0.05"
                    min="0"
                    max="0.5"
                    value={trainCfg.valSplit}
                    onChange={(e) =>
                      setTrainCfg((p) => ({ ...p, valSplit: +e.target.value }))
                    }
                  />
                </Field>
                <Field label={t.earlyStop}>
                  <input
                    style={S.input}
                    type="number"
                    value={trainCfg.earlyStop}
                    onChange={(e) =>
                      setTrainCfg((p) => ({
                        ...p,
                        earlyStop: +e.target.value || 1,
                      }))
                    }
                  />
                </Field>
              </div>
            </div>
            <div
              style={{
                ...S.card,
                fontFamily: "'Courier New',monospace",
                fontSize: fs - 2,
                whiteSpace: "pre-wrap",
                color: th.textMuted,
                maxHeight: 300,
                overflow: "auto",
                lineHeight: 1.6,
              }}
            >
              {network.length > 0
                ? genPyCode(network, trainCfg)
                : `# ${lang === "zh" ? "请先在构建器中添加层" : "Add layers in Builder first"}`}
            </div>
            <div style={{ display: "flex", gap: 8, marginTop: 12 }}>
              <button
                style={{
                  ...S.btn(accent, "#fff"),
                  flex: 1,
                  padding: "8px 0",
                  textAlign: "center",
                }}
                onClick={() => handleExport("py")}
              >
                {t.exportPy}
              </button>
              <button
                style={{
                  ...S.btn("#7c3aed", "#fff"),
                  flex: 1,
                  padding: "8px 0",
                  textAlign: "center",
                }}
                onClick={() => handleExport("ipynb")}
              >
                {t.exportIpynb}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ═══ UI ═══ */}
      {tab === "ui" && (
        <div style={{ flex: 1, overflow: "auto", padding: 20, zIndex: 1 }}>
          <div style={{ maxWidth: 500, margin: "0 auto" }}>
            <div
              style={{ fontWeight: 700, fontSize: fs + 1, marginBottom: 12 }}
            >
              {t.uiTitle}
            </div>
            <div style={S.card}>
              <Field label={t.theme}>
                <div style={{ display: "flex", gap: 6 }}>
                  {["dark", "light"].map((v) => (
                    <button
                      key={v}
                      style={{
                        ...S.btn(
                          uiCfg.theme === v ? accent : th.bg3,
                          uiCfg.theme === v ? "#fff" : th.text,
                        ),
                        flex: 1,
                        textAlign: "center",
                        padding: "8px 0",
                      }}
                      onClick={() => setUiCfg((p) => ({ ...p, theme: v }))}
                    >
                      {t[v]}
                    </button>
                  ))}
                </div>
              </Field>
              <Field label={`${t.fontSize}: ${uiCfg.fontSize}px`}>
                <input
                  type="range"
                  min="10"
                  max="18"
                  value={uiCfg.fontSize}
                  onChange={(e) =>
                    setUiCfg((p) => ({ ...p, fontSize: +e.target.value }))
                  }
                  style={{ width: "100%" }}
                />
              </Field>
              <Field label={t.accentColor}>
                <div style={{ display: "flex", gap: 6 }}>
                  {[
                    "#3B82F6",
                    "#10B981",
                    "#EF4444",
                    "#F59E0B",
                    "#8B5CF6",
                    "#EC4899",
                    "#06B6D4",
                  ].map((c) => (
                    <div
                      key={c}
                      onClick={() => setUiCfg((p) => ({ ...p, accent: c }))}
                      style={{
                        width: 28,
                        height: 28,
                        borderRadius: 6,
                        background: c,
                        cursor: "pointer",
                        border:
                          uiCfg.accent === c
                            ? "2px solid #fff"
                            : "2px solid transparent",
                      }}
                    />
                  ))}
                </div>
              </Field>
              <Field label={t.bgImage}>
                <input
                  style={S.input}
                  placeholder="https://..."
                  value={uiCfg.bgImage}
                  onChange={(e) =>
                    setUiCfg((p) => ({ ...p, bgImage: e.target.value }))
                  }
                />
              </Field>
              {uiCfg.bgImage && (
                <Field
                  label={`${t.bgOpacity}: ${(uiCfg.bgOpacity * 100).toFixed(0)}%`}
                >
                  <input
                    type="range"
                    min="0.01"
                    max="0.4"
                    step="0.01"
                    value={uiCfg.bgOpacity}
                    onChange={(e) =>
                      setUiCfg((p) => ({ ...p, bgOpacity: +e.target.value }))
                    }
                    style={{ width: "100%" }}
                  />
                </Field>
              )}
              <button
                style={{
                  ...S.btn(th.bg3),
                  width: "100%",
                  marginTop: 10,
                  textAlign: "center",
                  padding: "8px 0",
                }}
                onClick={() =>
                  setUiCfg({
                    theme: "dark",
                    fontSize: 13,
                    bgImage:
                      "https://invoker-pray.github.io/img/cover/toysandtools-cover.jpg",
                    bgOpacity: 0.08,
                    accent: "#3B82F6",
                  })
                }
              >
                {t.resetUI}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
