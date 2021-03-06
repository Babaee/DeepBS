require 'image'
require 'nn'
require 'cunn'
require 'xlua'
--require 'hdf5'
-- dofile '/usr/home/dit/Desktop/deep_bg_segmenter/model/custom_layers.lua'
dofile '/usr/home/rez/ZM/CNN/bg_cnn_code/bg_cnn_code/linux/network_and_training/network/custom_layers.lua'

-- F = require '/usr/home/rez/ZM/CNN/bg_cnn_code/bg_cnn_code/linux/utils'
F = require 'utils'

-- Lua implementation of PHP scandir function
function scandir(directory)
    local i, t, popen = 0, {}, io.popen
    local pfile = popen('ls -a "'..directory..'"')
    for filename in pfile:lines() do
        i = i + 1
		if i > 2 then
        	t[i-2] = filename
		end
	end
    pfile:close()
    return t
end

-- threads
torch.setnumthreads(4)
print('<torch> set nb of threads to ' .. torch.getnumthreads())


root_dir = '/usr/home/rez/ZM/CDNet_Dataset/dataset'


local categories = scandir(root_dir)
local videos = {}

for i=1, #categories do
	videos[categories[i]] = scandir(root_dir .. '/' .. categories[i])
end

-- initialize paths and other parameters
local default_model_file = '/usr/home/rez/ZM/CNN/trained/big_net_010.net'

local model
local out_root = '/usr/home/rez/ZM/CNN/ZM_Result/cdnet2014'
local bg_root = '/usr/home/rez/ZM/CNN/bg_cnn_code/bg_cnn_code/linux/data/bg_data'
local bg_stat_file = '/usr/home/rez/ZM/CNN/ZM_Result/stats/bg_stat.h5'
local fg_stat_file = '/usr/home/rez/ZM/CNN/ZM_Result/stats/sequence_stat.h5'


-- local bg_stat
-- local sequence_stat
-- local bg_mean
-- local sequence_mean

-- open h5 files and get statistics
-- bg_stat = hdf5.open(bg_stat_file)
-- sequence_stat = hdf5.open(fg_stat_file)
-- bg_mean = bg_stat:read('patch_mean_color'):all()
-- sequence_mean = sequence_stat:read('patch_mean_color'):all()
-- bg_stat:close()
-- sequence_stat:close()


-- load model
model = torch.load(default_model_file)
-- set for gpu mode
model = model:cuda()
model:evaluate()

local bg_im

for i=1, #categories do
	print(string.format('processing category: %s', categories[i]))
	xlua.progress(i, #categories)

	cat_path = root_dir .. '/' .. categories[i]
	for v=1, #videos[categories[i]] do
		video_path = cat_path .. '/' .. videos[categories[i]][v] .. '/input'
		file_names = scandir(video_path)

		for f = 1,#file_names do
			-- bg_im = image.load(bg_root .. '/' .. categories[i] .. '/' .. videos[categories[i]][v] ..'/' .. string.gsub(string.gsub(file_names[f],"in", "bin"), "jpg", "png"), 3, 'float')
			bg_im = image.load(bg_root .. '/' .. categories[i] .. '/' .. videos[categories[i]][v] ..'/' .. 'background.jpg', 3, 'float')
			file_path = video_path .. '/' .. file_names[f]
			out_path = out_root .. '/' .. categories[i] .. '/' .. videos[categories[i]][v] ..'/' .. string.gsub(string.gsub(file_names[f],"in", "bin"), "jpg", "png")
			beliefs = F.imToBgMap(file_path, out_path , bg_im, model, bg_mean, sequence_mean, 37, true)
		end

	end
end


--[[

]]
