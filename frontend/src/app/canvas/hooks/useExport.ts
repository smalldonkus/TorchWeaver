import { Ephesis } from "next/font/google";
import { lazy } from "react";

export default function useExport(nodes: any[], edges: any[]) {
  return () => {
    const inputMap: Record<string, string[]> = {};
    edges.forEach((edge) => {
      if (!inputMap[edge.target]) inputMap[edge.target] = [];
      inputMap[edge.target].push(edge.source);
    });

    const layers = nodes.map((node) => {
      const [typeRaw, ...labelParts] = (node.data.label || "").split(":");
      const type = typeRaw.trim().toLowerCase();
      const id = labelParts.join(":").trim() || node.id;
      const params = node.data.parameters || {};

      let parameters = {};
      if (layerParamMap[type]) {
        parameters = layerParamMap[type](params);
      }

      return {
        id,
        type,
        inputs: (inputMap[node.id] || []).map((src) => `${src}_out`),
        outputs: [`${node.id}_out`],
        parameters,
      };
    });

    const json = JSON.stringify({ layers }, null, 2);
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "diagram.json";
    link.click();
    URL.revokeObjectURL(url);
  };
}

const layerParamMap: Record<string, (params: any) => any> = {

  // linear layers
  linear: (p) => linearParams(p),

  // convolutional layers
  conv1d: (p) => convParams,
  conv2d: (p) => convParams,
  conv3d: (p) => convParams,
  convtransposed1d: (p) => convParams,
  convtransposed2d: (p) => convParams,
  convtransposed3d: (p) => convParams,

  // pooling layers
  maxpool1d: (p) => poolParams(p),
  maxpool2d: (p) => poolParams(p),
  maxpool3d: (p) => poolParams(p),
  maxunpool1d: (p) => poolParams(p),
  maxunpool2d: (p) => poolParams(p),
  maxunpool3d: (p) => poolParams(p),
  avgpool1d: (p) => poolParams(p),
  avgpool2d: (p) => poolParams(p),
  avgpool3d: (p) => poolParams(p),
  adaptivemaxpool1d : (p) => adaptivePool(p),
  adaptivemaxpool2d : (p) => adaptivePool(p),
  adaptivemaxpool3d : (p) => adaptivePool(p),
  adaptiveavgpool1d : (p) => adaptivePool(p),
  adaptiveavgpool2d : (p) => adaptivePool(p),
  adaptiveavgpool3d : (p) => adaptivePool(p),

  // padding layers
  pad: (p) => paddingParams(p),
  constantpad1d: (p) => paddingParams(p),

  // normalization layers
  localresponsenorm: (p) => localresponsenorm(p),
  batchnorm1d: (p) => normalizationParams(p),
  batchnorm2d: (p) => normalizationParams(p),
  batchnorm3d: (p) => normalizationParams(p),
  instancenorm1d: (p) => normalizationParams(p),
  instancenorm2d: (p) => normalizationParams(p),
  instancenorm3d: (p) => normalizationParams(p),
  lazybatchnorm1d: (p) => normalizationParams(p),
  lazybatchnorm2d: (p) => normalizationParams(p),
  lazybatchnorm3d: (p) => normalizationParams(p),
  syncbatchnorm: (p) => normalizationParams(p),

  // dropout layers
  dropout1d: (p) => ({ p: Number(p.p) || 0.5 }),
  dropout2d: (p) => ({ p: Number(p.p) || 0.5 }),
  dropout3d: (p) => ({ p: Number(p.p) || 0.5 }),



};

const linearParams = (p: any) => ({
  in_features: Number(p.in_features) || null,
  out_features: Number(p.out_features) || null,
});

const convParams = (p: any) => ({
  in_channels: Number(p.in_channels) || null,
  out_channels: Number(p.out_channels) || null,
  kernel_size: Number(p.kernel_size) || null,
  stride: Number(p.stride) || 1,
  padding: Number(p.padding) || 0,
  output_padding: Number(p.output_padding) || 0,
  dilation: Number(p.dilation) || 1,
  groups: Number(p.groups) || 1,
  bias: Boolean(p.bias) || true,
  padding_mode: p.padding_mode || "zeros",
});

const poolParams = (p: any) => ({
  kernel_size: Number(p.kernel_size) || null,
  stride: Number(p.stride) || null,
  padding: Number(p.padding) || 0,
  dilation: Number(p.dilation) || 1,
  return_indices: Boolean(p.return_indices) || false,
  ceil_mode: Boolean(p.ceil_mode) || false,
}); 

const adaptivePool = (p: any) => ({
  output_size: Number(p.output_size) || null,
  return_indices: Boolean(p.return_indices) || false,
});

const paddingParams = (p: any) => ({
  in_channels: Number(p.in_channels) || null,
  out_channels: Number(p.out_channels) || null,
  padding: p.padding || null,
});

const normalizationParams = (p: any) => ({
  num_features: Number(p.num_features) || null,
  eps: Number(p.eps) || 1e-5,
  momentum: Number(p.momentum) || 0.1,
  affine: Boolean(p.affine) || true,
  track_running_stats: Boolean(p.track_running_stats) || true,
  process_group: p.process_group || null,
});

const localresponsenorm = (p: any) => ({
  size: Number(p.size) || null,
  alpha: Number(p.alpha) || 0.0001,
  beta: Number(p.beta) || 0.75,
  k: Number(p.k) || 1.0,
});


/* 

all layers types and their parameters --- torch nn module

Linear Layers:
- linear: in_features, out_features
// - bilinear: in1_features, in2_features, out_features # 2 inputs not doing for now

Convolutional Layers:
- conv1d: in_channels, out_channels, kernel_size, stride, padding
- conv2d: in_channels, out_channels, kernel_size, stride, padding
- conv3d: in_channels, out_channels, kernel_size, stride, padding
- convtransposed1d: in_channels, out_channels, kernel_size, stride, padding, output_padding
- convtransposed2d: in_channels, out_channels, kernel_size, stride, padding, output_padding
- convtransposed3d: in_channels, out_channels, kernel_size, stride, padding, output_padding

#not doing as there is no in_channels param in the node
// -lazyconv1d: out_channels, kernel_size, stride, padding
// -lazyconv2d: out_channels, kernel_size, stride, padding
// -lazyconv3d: out_channels, kernel_size, stride, padding
// -lazyconvtransposed1d: out_channels, kernel_size, stride, padding, output_padding
// -lazyconvtransposed2d: out_channels, kernel_size, stride, padding, output_padding
// -lazyconvtransposed3d: out_channels, kernel_size, stride, padding, output_padding


Pooling Layers:
- maxpool1d: kernel_size, stride, padding
- maxpool2d: kernel_size, stride, padding
- maxpool3d: kernel_size, stride, padding
- maxunpool1d: kernel_size, stride, padding
- maxunpool2d: kernel_size, stride, padding
- maxunpool3d: kernel_size, stride, padding
- avgpool1d: kernel_size, stride, padding
- avgpool2d: kernel_size, stride, padding
- avgpool3d: kernel_size, stride, padding
- adaptiveavgpool1d: output_size
- adaptiveavgpool2d: output_size
- adaptiveavgpool3d: output_size
- adaptivemaxpool1d: output_size
- adaptivemaxpool2d: output_size
- adaptivemaxpool3d: output_size

Padding Layers:
- pad: padding
- constantpad1d: padding, value
- constantpad2d: padding, value
- constantpad3d: padding, value
- reflectionpad1d: padding
- reflectionpad2d: padding
- reflectionpad3d: padding
- replicationpad1d: padding
- replicationpad2d: padding
- replicationpad3d: padding
- zeropad1d: padding
- zeropad2d: padding
- zeropad3d: padding

Normalization Layers:
- batchnorm1d: num_features
- batchnorm2d: num_features
- batchnorm3d: num_features
- lazybatchnorm1d: num_features
- lazybatchnorm2d: num_features
- lazybatchnorm3d: num_features
- syncbatchnorm: num_features
- layernorm: normalized_shape
- groupnorm: num_groups, num_channels
- instancenorm1d: num_features
- instancenorm2d: num_features
- instancenorm3d: num_features
- localresponsenorm: size, alpha, beta, k
- RMSNorm: normalized_shape
- 


Recurrent Layers:
- rnn: input_size, hidden_size, num_layers, bidirectional
- lstm: input_size, hidden_size, num_layers, bidirectional
- gru: input_size, hidden_size, num_layers, bidirectional
- rnncell: input_size, hidden_size
- lstmcell: input_size, hidden_size
- grucell: input_size, hidden_size

Activation Layers: not sure of activation layers params
// - relu
// - leakyrelu: negative_slope
// - prelu: num_parameters, init
// - gelu
// - silu
// - elu: alpha
// - selu
// - celu: alpha
// - hardswish
// - hardsigmoid
// - softmax: dim
// - logsoftmax: dim
// - tanh
// - sigmoid
// - softplus
// - softsign
// - threshold: threshold, value
// - hardtanh: min_val, max_val
// - softshrink: lambda
// - hardshrink: lambd
// - rrelu: lower, upper
// - celu: alpha
// - glu: dim
// - mish
// - swish
// - tanhshrink
// - threshold: threshold, value
// - logsigmoid
// - multiheadattention: embed_dim, num_heads
// - multiheadattention: embed_dim, num_heads, kdim, vdim

Transofrmation Layers:
// - transformer
- transformerencoder
- transformerencoderlayer
// - transformerdecoder
// - transformerdecoderlayer

Dropout Layers:
- dropout: p
- dropout2d: p
- dropout3d: p
- alphaDropout: p
- featureAlphaDropout: p

Sparse Layers:
- embedding: num_embeddings, embedding_dim
- embeddingbag: num_embeddings, embedding_dim

Vision Layers:
- pixelshuffle: upscale_factor
- unpixelshuffle: downscale_factor
- unfold: kernel_size, dilation, padding, stride
- fold: output_size, kernel_size, dilation, padding, stride

Shuffle Layers:
- channelshuffle: groups

DataParallel Layers:
- dataparallel
- distributeddataparallel

*/