require 'torch'
require 'nn'
require 'nngraph'

local ModelBuilder = torch.class('ModelBuilder')

function ModelBuilder:make_net(w2v, opts)
  if opts.cudnn == 1 then
    require 'cudnn'
    require 'cunn'
  end

  local input = nn.Identity()()

  local lookup
  if opts.model_type == 'multichannel' then
    local channels = {}
    for i = 1, 2 do
      local chan = nn.LookupTable(opts.vocab_size, opts.vec_size)
      chan.weight:copy(w2v)
      chan.weight[1]:zero()
      chan.name = 'channel' .. i
      table.insert(channels, chan(input))
    end
    lookup = channels
  else
    lookup = nn.LookupTable(opts.vocab_size, opts.vec_size)
    if opts.model_type == 'static' or opts.model_type == 'nonstatic' then
      lookup.weight:copy(w2v)
    else
      -- rand
      lookup.weight:uniform(-0.25, 0.25)
    end
    -- padding should always be 0
    lookup.weight[1]:zero()

    lookup = lookup(input)
  end

  -- kernels is an array of kernel sizes
  local kernels = {opts.kernel1, opts.kernel2, opts.kernel3}
  local layer1 = {}
  for i = 1, #kernels do
    local conv
    local conv_layer
    local max_time
    if opts.cudnn == 1 then
      conv = cudnn.SpatialConvolution(1, opts.num_feat_maps, opts.vec_size, kernels[i])
      if opts.model_type == 'multichannel' then
        local lookup_conv = {}
        for chan = 1, 2 do
          table.insert(lookup_conv, nn.Reshape(opts.num_feat_maps, opts.max_sent-kernels[i]+1, true)(
            conv(
            nn.Reshape(1, opts.max_sent, opts.vec_size, true)(
            lookup[chan]))))
        end
        conv_layer = nn.CAddTable()(lookup_conv)
        max_time = nn.Max(3)(cudnn.ReLU()(conv_layer))
      else
        if opts.highway_conv_layers > 0 then
          -- Highway conv layers
          local highway_conv = HighwayConv.conv(opts.vec_size, opts.max_sent, kernels[i], opts.highway_conv_layers)
          conv_layer = nn.Reshape(opts.num_feat_maps, opts.max_sent-kernels[i]+1, true)(
            conv(nn.Reshape(1, opts.max_sent, opts.vec_size, true)(
            highway_conv(lookup))))
          max_time = nn.Max(3)(cudnn.ReLU()(conv_layer))
        else
          conv_layer = nn.Reshape(opts.num_feat_maps, opts.max_sent-kernels[i]+1, true)(
            conv(
            nn.Reshape(1, opts.max_sent, opts.vec_size, true)(
            lookup)))
          max_time = nn.Max(3)(cudnn.ReLU()(conv_layer))
        end
      end
    else
      conv = nn.TemporalConvolution(opts.vec_size, opts.num_feat_maps, kernels[i])
      if opts.model_type == 'multichannel' then
        local lookup_conv = {}
        for chan = 1,2 do
          table.insert(lookup_conv, conv(lookup[chan]))
        end
        conv_layer = nn.CAddTable()(lookup_conv)
        max_time = nn.Max(2)(nn.ReLU()(conv_layer)) -- max over time
      else
        conv = nn.TemporalConvolution(opts.vec_size, opts.num_feat_maps, kernels[i])
        conv_layer = conv(lookup)
        max_time = nn.Max(2)(nn.ReLU()(conv_layer)) -- max over time
      end
    end

    conv.weight:uniform(-0.01, 0.01)
    conv.bias:zero()
    table.insert(layer1, max_time)
  end

  local conv_layer_concat
  if #layer1 > 1 then
    conv_layer_concat = nn.JoinTable(2)(layer1)
  else
    conv_layer_concat = layer1[1]
  end

  local last_layer = conv_layer_concat
  if opts.highway_mlp > 0 then
    -- use highway layers
    local highway = HighwayMLP.mlp((#layer1) * opts.num_feat_maps, opts.highway_layers)
    last_layer = highway(conv_layer_concat)
  end

  -- last layer is a vector of size opts.num_feat_maps
  model = nn.gModule({input}, {last_layer})
  return model
end

return ModelBuilder
