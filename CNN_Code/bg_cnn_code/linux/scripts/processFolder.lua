require 'nn'
require 'cunn'
require 'xlua'
require 'hdf5'
dofile '/usr/home/dit/Desktop/deep_bg_segmenter/model/custom_layers.lua'
F = require 'utils' 
require 'image'

-- threads
torch.setnumthreads(4)
print('<torch> set nb of threads to ' .. torch.getnumthreads())


-- initialize paths and other parameters
local default_model_file = '/usr/home/dit/Desktop/conv_bg_subtractor_color/model/trained/all_categories/subsense_big/big_net_010.net'

local model
local out_root = '/media/dit/data/Datasets/beliefs/subsense_big_net_10/cdnet2014'
local bg_root = '/media/dit/data/Datasets/cdnet2014_subsense_bg_model_corrected'
local bg_stat_file = '/media/dit/data/Datasets/subsense_block_data_color/all_categories/stats/bg_stat.h5'
local fg_stat_file = '/media/dit/data/Datasets/subsense_block_data_color/all_categories/stats/sequence_stat.h5'


local bg_stat
local sequence_stat
local bg_mean
local sequence_mean

-- open h5 files and get statistics
bg_stat = hdf5.open(bg_stat_file)
sequence_stat = hdf5.open(fg_stat_file)
bg_mean = bg_stat:read('patch_mean_color'):all()
sequence_mean = sequence_stat:read('patch_mean_color'):all()
bg_stat:close()
sequence_stat:close()


-- load model
model = torch.load(default_model_file)
-- set for gpu mode
model = model:cuda()
model:evaluate()

local bg_im


folder_root = '/media/dit/data/Datasets/wallflower/TimeOfDay'
bg_root = '/media/dit/data/Datasets/bgs_segmentation/subsense/wallflower/TimeOfDay/bg_model'
belief_out_root = '/media/dit/data/Datasets/bgs_segmentation/beliefs/wallflower/TimeOfDay'

i = 1851



--for i=1, 10000 do
	img = folder_root .. string.format('/b%05d.png', i-1)
	bg_im = image.load(bg_root .. string.format('/bin%06d.png', i), 3, 'float')
	out_img = belief_out_root  .. string.format('/bin%06d.png', i)
	beliefs = F.imToBgMap(img , out_img  , bg_im, model, bg_mean, sequence_mean, 37, true)
--end

--[[

]]
