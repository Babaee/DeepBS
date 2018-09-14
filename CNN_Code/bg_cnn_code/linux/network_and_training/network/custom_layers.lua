require 'nn'

do
    local Linear, parent = torch.class('nn.CustomLinear', 'nn.Linear')
   
    -- override the constructor to have the additional range of initialization

    function Linear:__init(inputSize, outputSize, mean, std, bias)
        parent.__init(self,inputSize,outputSize)
        self:reset(mean,std,bias,outputSize)
    end
   
    -- override the :reset method to use custom weight initialization.        
    function Linear:reset(mean,stdv,bias,outputSize,absBound)	
		local mu, sig, b
		local outputSize = outputSize or 1	
		local absBound = absBound or 0.2
 	
        if (mean ~= nil and stdv ~= nil  and bias ~= nil )then
            mu = mean
			sig = stdv
			b = bias
        else
			mu = 0
			sig = 0.01
			b = 0.1
        end

        self.weight:normal(mu,sig)
        self.bias = torch.ones(outputSize) * b

		repeat
			-- check elements with abs value > 0.2
			gt_mask = self.weight:abs():gt(absBound)
			-- reassign those elements
			self.weight[gt_mask]:normal(mu,sig)

		until torch.all(gt_mask:eq(0))
    end

end


do
    local SpatialConvolution, parent = torch.class('nn.CustomSpatialConvolution', 'nn.SpatialConvolution')
   
    -- override the constructor to have the additional range of initialization

    function SpatialConvolution:__init(nInputPlane, nOutputPlane, kW, kH, mean, std, bias, dW, dH, padW, padH)
        parent.__init(self, nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
        self:reset(mean,std,bias,nOutputPlane)
    end
   
    -- override the :reset method to use custom weight initialization.        
    function SpatialConvolution:reset(mean,stdv,bias, nOutputPlane ,absBound)
		local mu, sig, b
		local absBound = absBound or 0.2
		local nOutputPlane = nOutputPlane or 1

        if (mean ~= nil and stdv ~= nil  and bias ~= nil ) then
            mu = mean
			sig = stdv
			b = bias
        else
			mu = 0
			sig = 0.01
			b = 0.1
        end

        self.weight:normal(mu,sig)
        self.bias = torch.ones(nOutputPlane) * b
		local gt_mask
		repeat
			-- check elements with abs value > 0.2
			gt_mask = self.weight:abs():gt(absBound)
			-- reassign those elements
			self.weight[gt_mask]:normal(mu,sig)
		until torch.all(gt_mask:eq(0))
    end

end



