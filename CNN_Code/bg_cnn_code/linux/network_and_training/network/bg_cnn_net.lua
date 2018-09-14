require 'nn'
require 'cunn'
dofile 'custom_layers.lua'

-- conv layers
subnet = nn.Sequential()
subnet:add(nn.Reshape(6,37,37))
subnet:add(nn.CustomSpatialConvolution(6, 24, 5, 5, 0, 0.01, 0.1, 1, 1, 3, 3))
subnet:add(nn.SpatialBatchNormalization(24))
subnet:add(nn.ReLU())
subnet:add(nn.SpatialMaxPooling(2, 2))
subnet:add(nn.CustomSpatialConvolution(24, 48, 5, 5, 0, 0.01, 0.1, 1, 1, 3, 3))
subnet:add(nn.SpatialBatchNormalization(48))
subnet:add(nn.ReLU())
subnet:add(nn.SpatialMaxPooling(2, 2))
subnet:add(nn.CustomSpatialConvolution(48, 96, 5, 5, 0, 0.01, 0.1, 1, 1, 3, 3))
subnet:add(nn.SpatialBatchNormalization(96))
subnet:add(nn.ReLU())
subnet:add(nn.SpatialMaxPooling(2, 2))

-- mlp classifier
subnet:add(nn.Reshape(96*6*6))
subnet:add(nn.CustomLinear(96*6*6,2048))
subnet:add(nn.BatchNormalization(2048))
subnet:add(nn.ReLU())
subnet:add(nn.CustomLinear(2048,37*37))
subnet:add(nn.Sigmoid())

--test
--test_input = torch.rand(1,2,3,37,37)
--print(#subnet:forward(test_input))
return subnet


