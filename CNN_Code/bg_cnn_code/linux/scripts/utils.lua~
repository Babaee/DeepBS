require 'hdf5'
require 'nn'
require 'image'
require 'xlua'
require 'cunn'
-- require 'caffe'

F = {}


local function imToBgMap(in_im, file_out, bg_im, model, bg_mean, sequence_mean, pSize, save)
	local im = image.load(in_im, 3, 'float')
	
	local height, width = 240, 320
	local original_height, original_width = (#im)[2], (#im)[3]
	im = image.scale(im, width, height, 'bilinear')
	bg_im = image.scale(bg_im , width, height, 'bilinear')

	local pSize = pSize or 25
	local save = save or false

	local padY = (math.floor(height/pSize) * pSize + pSize - height) % pSize
	local padX = (math.floor(width/pSize) * pSize + pSize - width) % pSize

	local pad_height = height + padY
	local pad_width = width + padX

	-- initialize model
	model:evaluate()

	local containers = {	
		torch.zeros(pad_height/pSize * pad_width/pSize,2,3, pSize, pSize),
		torch.zeros(pad_height/pSize * pad_width/pSize,2,3, pSize, pSize),
		torch.zeros(pad_height/pSize * pad_width/pSize,2,3, pSize, pSize),
		torch.zeros(pad_height/pSize * pad_width/pSize,2,3, pSize, pSize)
	}

	-- load and resize images images
	imTensors  = {	
		torch.zeros(6,pad_height, pad_width),
		torch.zeros(6,pad_height, pad_width),
		torch.zeros(6,pad_height, pad_width),
		torch.zeros(6,pad_height, pad_width)
	}
	
	-- set images for different corners
	imTensors[1][{{1,3}, {1,height}, {1,width}}] = im
	imTensors[1][{{4,6}, {1,height}, {1,width}}] = bg_im

	imTensors[2][{{1,3}, {padY + 1, padY+height}, {padX+1, padX+width}}] = im
	imTensors[2][{{4,6}, {padY + 1, padY+height}, {padX+1, padX+width}}]  = bg_im

	imTensors[3][{{1,3}, {1,height}, {padX+1, padX  +width}}] = im
	imTensors[3][{{4,6}, {1,height}, {padX+1, padX  +width}}] = bg_im

	imTensors[4][{{1,3}, {padY + 1, padY+height}, {1,width}}] = im
	imTensors[4][{{4,6}, {padY + 1, padY+height}, {1,width}}]  = bg_im

	i = 0

	for y = 1,height + padY, pSize do
		for x = 1,width + padX, pSize do
			i = i + 1
			for j = 1, 4 do			
			containers[j][{i,1,{},{},{}}] = imTensors[j][{{1,3},{y,y+pSize-1},{x,x+pSize-1}}]
			containers[j][{i,1,{},{},{}}]:add(-sequence_mean)

			containers[j][{i,2,{},{},{}}] = imTensors[j][{{4,6},{y,y+pSize-1},{x,x+pSize-1}}]
			containers[j][{i,2,{},{},{}}]:add(-bg_mean)

			end
		end
	end

	local pred = {}
	local out_ims = {}

	local predictions = model:forward(torch.cat(containers, 1):cuda()):float():clone()
	--local predictions = model:forward({torch.cat(containers, 1):cuda(),torch.cat(containers, 1):cuda()}):float():clone() -- for hierarchichal net

	for j = 1, 4 do
		out_ims[j] = torch.zeros(pad_height,pad_width)
		pred[j] = predictions[{{(j-1)*(#containers[j])[1] + 1, j*(#containers[j])[1]},{}}]
		pred[j] = pred[j]:reshape((#containers[j])[1], pSize, pSize)
	end

	
	i = 0
	for y = 1,height + padY, pSize do
		for x = 1,width + padX, pSize do
			i = i + 1
			for j = 1, 4 do
				out_ims[j][{{y,y+pSize-1},{x,x+pSize-1}}] = pred[j][{i,{},{}}]
			end
		end
	end

	out_ims[1] = out_ims[1][{{1,height}, {1,width}}]
	out_ims[2] = out_ims[2][{{padY + 1, padY+height}, {padX+1, padX+width}}]
	out_ims[3] = out_ims[3][{{1,height}, {padX+1, padX+width}}]
	out_ims[4] = out_ims[4][{{padY+1, padY+height}, {1,width}}]


	out_sum = torch.zeros(height,width)
	for j = 1,4 do
		out_sum = out_sum + out_ims[j]
	end
	out = image.scale(out_sum:mul(0.25), original_width, original_height, 'bilinear')

	if save then
		--image.save(file_out, image.y2jet(out *255))
		image.save(file_out, out)
		--image.save(file_out,avgOverSuperpx(out, in_im))
	end
	out = image.scale(out, 320, 240, 'bilinear')

	--input = torch.zeros(1,5,240,320)
	--input[{1,{1,3},{},{}}] = im*255.0
	--input[{1,5,{},{}}] = out
	--input[{1,4,{},{}}] = 1-out
	
	--out = caffe_model:forward(input:float())
	return out--out[{1,2,{},{}}]

	--test for analyzing combined maps
	--[[
	local myFile = hdf5.open(string.format('predictions.h5'), 'w')
	for j = 1,4 do
		myFile:write(string.format('/highway_%d', j), out_ims[j])
	end
	myFile:close()
	]]
end


local function getfeatureMaps(in_im, file_out, bg_im, model, bg_mean, sequence_mean, noFeatureMaps, featSize, save)
    local im = image.load(in_im, 3, 'float')
    
    local height, width = 240, 320
    local original_height, original_width = (#im)[2], (#im)[3]
    im = image.scale(im, width, height, 'bilinear')
    bg_im = image.scale(bg_im , width, height, 'bilinear')

    local pSize = pSize or 37
    local save = save or false

    local padY = (math.floor(height/pSize) * pSize + pSize - height) % pSize
    local padX = (math.floor(width/pSize) * pSize + pSize - width) % pSize

    local pad_height = height + padY
    local pad_width = width + padX

    -- initialize model
    model:evaluate()

    local containers = {torch.zeros(pad_height/pSize * pad_width/pSize,2,3, pSize, pSize)}

    -- load and resize images images
    imTensors  = {torch.zeros(6,pad_height, pad_width)}
    
    -- set images for different corners
    imTensors[1][{{1,3}, {1,height}, {1,width}}] = im
    imTensors[1][{{4,6}, {1,height}, {1,width}}] = bg_im

    i = 0

    local  xRange= torch.range(1,width + padX, pSize)
    local  yRange= torch.range(1,height + padY, pSize)
    local patchHeight, patchWidth = (#yRange)[1], (#xRange)[1]

    for y = 1,height + padY, pSize do
        for x = 1,width + padX, pSize do
            i = i + 1    
            containers[1][{i,1,{},{},{}}] = imTensors[1][{{1,3},{y,y+pSize-1},{x,x+pSize-1}}]
            containers[1][{i,1,{},{},{}}]:add(-sequence_mean)

            containers[1][{i,2,{},{},{}}] = imTensors[1][{{4,6},{y,y+pSize-1},{x,x+pSize-1}}]
            containers[1][{i,2,{},{},{}}]:add(-bg_mean)
        end
    end

    local pred = {}
    local out_ims = {}

    local predictions = model:forward(torch.cat(containers, 1)):float():clone()
    --print('OK!')
    --print(#predictions)    
    
    -- reassemble patches
    out_size = featSize
    feat_map_no = noFeatureMaps

    out_ims[1] = torch.zeros(feat_map_no, patchHeight * out_size, patchWidth * out_size)
    --pred[1] = predictions[{{},feat_map_no,{},{}}]
    pred[1] = predictions[{{},{},{},{}}]
    --pred[1] = pred[1]:reshape((#containers[1])[1], out_size, out_size)

    --print('OK!')
    --print(#pred[1])    


    i = 0
    
    for y = 1,patchHeight*out_size,out_size do
        for x = 1,patchWidth*out_size,out_size do
            i = i + 1
            out_ims[1][{{},{y,y+out_size-1},{x,x+out_size-1}}] = pred[1][{i,{},{},{}}]
        end
    end
    out_ims[1] = image.toDisplayTensor({input=out_ims[1],padding=5})
    --out = image.scale(out_ims[1] , original_width, original_height, 'bilinear')

    if save then
        --image.save(file_out, image.y2jet(out *255))
        image.save(file_out, out_ims[1])
        --image.save(file_out,avgOverSuperpx(out, in_im))
    end

    return out_ims[1]


end

    F.imToBgMap = imToBgMap
    F.getfeatureMaps = getfeatureMaps

return F






