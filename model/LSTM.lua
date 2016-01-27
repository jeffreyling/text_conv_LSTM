require 'rnn'
require 'nn'

local LSTMBuilder = {}

-- Builds an LSTM upon d-dimensional input sequence
function LSTMBuilder:make_net(d, opts)
  local input = nn.Identity()()

  local seq = nn.Sequencer()(nn.SplitTable(1)(input))
  local lstm = nn.FastLSTM(d, d)
  local h = lstm(seq)

  -- need to get last element
  local last_hidden = h[#h]

  -- simple MLP layer
  local linear = nn.Linear(, opts.num_classes)
  linear.weight:normal():mul(0.01)
  linear.bias:zero()

  local softmax
  if opts.cudnn == 1 then
    softmax = cudnn.LogSoftMax()
  else
    softmax = nn.LogSoftMax()
  end

  local output = softmax(linear(nn.Dropout(opts.dropout_p)(h[#h]))) 

  local model = nn.gModule({input}, {output})
  return model
end

return LSTMBuilder
