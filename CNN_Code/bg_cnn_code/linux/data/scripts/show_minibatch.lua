require 'DataLoader' 
require 'image'


--[[
	- Displays mini batch of size
	- adapt paths and indices
	- sequence_length: number of files (train + val files)
	- keep gray_scale = false
	- Parameter settings e.g. in "train" folder : data000000.h5 - data000010.h5
			  					 "val" folder: data000011.h5 - data000016.h5
		=> start_train = 0, stop_train = 10, sequence_length = 17

     important: start with qlua -> $ qlua show_minibatch.lua
]]

-- local category = 'all_categories'
local category = ''

local dl_param = {
	h5_sequence = '/usr/home/rez/ZM/CNN/ZM_Result/', 
	sequence_stat = '/usr/home/rez/ZM/CNN/ZM_Result/stats/sequence_stat.h5' ,
	bg_stat = '/usr/home/rez/ZM/CNN/ZM_Result/stats/bg_stat.h5',
	start_train = 0, 
	stop_train = 149, 
	sequence_length = 3,
	batch_size = 48,
	gray_scale = false
}

l = DataLoader(dl_param)
test = l:getNextBatch('train')


image.display(image.toDisplayTensor(test.data[{{},1,{},{},{}}]))
image.display(image.toDisplayTensor(test.data[{{},2,{},{},{}}]))
image.display(image.toDisplayTensor(test.label[{{},{},{}}]))

