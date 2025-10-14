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

Activation Layers:
- relu
- leakyrelu: negative_slope
- prelu: num_parameters, init
- gelu
- silu
- elu: alpha
- selu
- celu: alpha
- hardswish
- hardsigmoid
- softmax: dim
- logsoftmax: dim
- tanh
- sigmoid
- softplus
- softsign
- threshold: threshold, value
- hardtanh: min_val, max_val
- softshrink: lambda
- hardshrink: lambd
- rrelu: lower, upper
- celu: alpha
- glu: dim
- mish
- swish
- tanhshrink
- threshold: threshold, value
- logsigmoid
- multiheadattention: embed_dim, num_heads
- multiheadattention: embed_dim, num_heads, kdim, vdim

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

  // padding layers
  pad: (p) => paddingParams(p),
  constantpad1d: (p) => paddingParams(p),



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
});

const poolParams = (p: any) => ({
  kernel_size: Number(p.kernel_size) || null,
  stride: Number(p.stride) || 1,
  padding: Number(p.padding) || 0,
}); 

const rnnParams = (p: any) => ({
  input_size: Number(p.input_size) || null,
  hidden_size: Number(p.hidden_size) || null,
  num_layers: Number(p.num_layers) || 1,
  bidirectional: Boolean(p.bidirectional) || false,
});

const paddingParams = (p: any) => ({
  padding: p.padding || null,
});