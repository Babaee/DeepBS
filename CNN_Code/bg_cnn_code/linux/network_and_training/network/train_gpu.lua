require 'DataLoader'
require 'optim'
require 'gnuplot'
require 'cunn'
require 'cutorch'
-- threads
torch.setnumthreads(1)
print('<torch> set nb of threads to ' .. torch.getnumthreads())

net = require 'bg_cnn_net'
--net = torch.load('/usr/home/dit/Desktop/conv_bg_subtractor_color/model/trained/all_categories/subsense_big/big_net_006.net') -- resume training
--criterion = nn.BCECriterion()
--criterion = nn.MSECriterion()
--criterion = nn.AbsCriterion()
--criterion = nn.ClassNLLCriterion()
criterion = nn.CrossEntropyCriterion()

local dl_param = {
	h5_sequence = '/usr/home/rez/ZM/CNN/Up_Proj/', 
	sequence_stat = '/usr/home/rez/ZM/CNN/Up_Proj/stats/sequence_stat.h5' ,
	bg_stat = '/usr/home/rez/ZM/CNN/Up_Proj/stats/bg_stat.h5',
	start_train = 0, 
	stop_train = 149, 
	sequence_length = 170,
	batch_size = 150,
	gray_scale = false
}

local dl = DataLoader(dl_param)

rmsprop_prop_params = {
   --learningRate = 2.5e-3,
   learningRate = 5e-3
}

x, dl_dx = net:getParameters()


-- enable gpu computation
net = net:cuda()
criterion = criterion:cuda()



-- define step function
step = function()
	net:training()
	local current_loss = 0
	local mini_batch
	local iteration = 0
	local lambda = 0 -- used for L2 regularizatuion
	local max_iter = dl.max_iter.train
	local accLoss = 0
	local pSize = 37
	local label_vector

	while true do
		xlua.progress(iteration,max_iter)
		mini_batch = dl:getNextBatch('train')
		if not mini_batch then break end

		mini_batch.data = mini_batch.data:cuda()
		mini_batch.label = mini_batch.label:cuda()

		--break
	    local feval = function(x_new)
	        -- reset data
	        if x ~= x_new then x:copy(x_new) end
	        dl_dx:zero()

	        -- perform mini-batch gradient descent
			local output = net:forward(mini_batch.data)
			
			-- ignore labels procedure
			label_vector = mini_batch.label:view((#mini_batch.label)[1]*1369, 1):clone()
			output = output:view((#mini_batch.label)[1]*1369, 1)
			--print(output)
			output[label_vector:eq(2)] = 1
			label_vector[label_vector:eq(1)] = 2
			label_vector[label_vector:eq(0)] = 1
			local output_0 = torch.CudaTensor((#mini_batch.label)[1]*1369,1):uniform(1)-output
			output = torch.cat(output_0,output)
			--print(output)
			--print(label_vector)
	        local loss = criterion:forward(output, label_vector)
			local gradients = criterion:backward(output, label_vector)
			gradients = gradients:select(2,2)
	        net:backward(mini_batch.data,gradients:resize((#mini_batch.label)[1],37*37))

	        -- locals:
	        --local norm,sign= torch.norm,torch.sign

	        -- Loss:
	        --loss = loss + lambda *  norm(x,2)^2/2

	        -- Gradients:
	        --dl_dx:add(x:clone():mul(lambda))

	       -- normalize gradients and f(X)
	       --dl_dx:div(mini_batch.size)
	       --loss = loss/mini_batch.size
		
	        return loss, dl_dx
	    end

	    _, fs = optim.rmsprop(feval, x, rmsprop_prop_params)
	    -- fs is a table containing value of the loss function
	    -- (just 1 value for the SGD optimization)
	    current_loss = current_loss + fs[1]
		iteration = iteration + 1
		accLoss = accLoss + fs[1]
		if iteration % 100 == 0 then
			print(string.format('Iteration: %d Current loss: %f', iteration, accLoss/100))
			accLoss = 0		
		end
		if iteration == max_iter then break end
	end

	return current_loss/iteration
end

eval = function()
	net:evaluate()
	local max_iter = dl.max_iter.val
	local loss = 0
	local iteration = 0
	local output
	local pSize = 37

	while true do
		xlua.progress(iteration,max_iter)
		mini_batch = dl:getNextBatch('val')
		--if not mini_batch then break end
		if not mini_batch or iteration == max_iter then break end

		mini_batch.data = mini_batch.data:cuda()
		mini_batch.label = mini_batch.label:cuda()

		output = net:forward(mini_batch.data)

		-- ignore labels procedure
		label_vector = mini_batch.label:view((#mini_batch.label)[1], pSize*pSize):clone()
		output[label_vector:eq(2)] = 1
		label_vector[label_vector:eq(2)] = 1

		output = output:view((#mini_batch.label)[1]*1369, 1)
			--print(output)
		output[label_vector:eq(2)] = 1
		label_vector[label_vector:eq(1)] = 2
		label_vector[label_vector:eq(0)] = 1
		local output_0 = torch.CudaTensor((#mini_batch.label)[1]*1369,1):uniform(1)-output
		output = torch.cat(output_0,output)

		loss = loss + criterion:forward(output, label_vector)
		iteration = iteration + 1
	end

	return loss/iteration
end

-- do training and screenshot every epoch
local output_folder = '/usr/home/rez/ZM/CNN/Up_Proj/trained_gpu/'

os.execute('mkdir -p ' .. sys.dirname(output_folder))

local epochs = 10
local train_offset = 0
train_loss = torch.linspace(1,epochs,epochs)
train_x = torch.linspace(1,epochs,epochs)
val_x = torch.linspace(1,epochs,epochs)
val_loss = torch.linspace(1,epochs,epochs)


local saveEvery = 1

for i=1,epochs do
	local last_loss = 0
	print('epoch Nr. ',i)
	train_loss[i] = step()
	print(string.format('Loss on the training set: %4f', train_loss[i]))
	val_loss[i]  = eval()
	print(string.format('Loss on the validation set: %4f', val_loss[i]))

	--if ((i + train_offset)%saveEvery==0) then
	output_filename = output_folder .. string.format('/big_net_%03d.net', i + train_offset)
	torch.save(output_filename, net)
	--end
	dl = nil
	dl = DataLoader(dl_param)
	torch.save(output_folder .. '/val_loss.t7', val_loss)
	torch.save(output_folder .. '/train_loss.t7', train_loss)

	-- save train vs val loss figure

	gnuplot.epsfigure(output_folder .. '/val_loss.eps')
	gnuplot.plot({'Val loss',val_x,val_loss})
	gnuplot.xlabel('Epochs')
	gnuplot.ylabel('Validation Loss')
	gnuplot.plotflush()

	gnuplot.epsfigure(output_folder .. '/train_loss.eps')
	gnuplot.plot({'Train loss',train_x,train_loss})
	gnuplot.xlabel('Epochs')
	gnuplot.ylabel('Train Loss')
	gnuplot.plotflush()

	gnuplot.epsfigure(output_folder .. '/train_val_loss.eps')
	gnuplot.plot({'Train loss',train_x,train_loss},{'Val loss',val_x,val_loss})
	gnuplot.xlabel('Epochs')
	gnuplot.ylabel('Loss')
	gnuplot.plotflush()
end


torch.save(output_folder .. '/val_loss.t7', val_loss)
torch.save(output_folder .. '/train_loss.t7', train_loss)

-- save train vs val loss figure

gnuplot.epsfigure(output_folder .. '/val_loss.eps')
gnuplot.plot({'Val loss',val_x,val_loss})
gnuplot.xlabel('Epochs')
gnuplot.ylabel('Validation Loss')
gnuplot.plotflush()

gnuplot.epsfigure(output_folder .. '/train_loss.eps')
gnuplot.plot({'Train loss',train_x,train_loss})
gnuplot.xlabel('Epochs')
gnuplot.ylabel('Train Loss')
gnuplot.plotflush()

gnuplot.epsfigure(output_folder .. '/train_val_loss.eps')
gnuplot.plot({'Train loss',train_x,train_loss},{'Val loss',val_x,val_loss})
gnuplot.xlabel('Epochs')
gnuplot.ylabel('Loss')
gnuplot.plotflush()



