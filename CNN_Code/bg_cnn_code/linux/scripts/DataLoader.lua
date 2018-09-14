require 'torch'
require 'hdf5'
require 'image'

local DataLoader = torch.class('DataLoader')
-- dataloader class for iid training (i.e. no sequence training)

function DataLoader:__init(dl_param)

	self.h5_info = {}
	self.h5_info.train = {root = dl_param.h5_sequence .. '/train', start = dl_param.start_train, stop = dl_param.stop_train}
	self.h5_info.val = {root = dl_param.h5_sequence .. '/val', start = dl_param.stop_train + 1, stop = dl_param.sequence_length + dl_param.start_train - 1}
	self.batch_size = dl_param.batch_size
	self.data = {}
	self.stat = {}

	-- load statistics h5 files (generated from matlab)
	local bg_stat = hdf5.open(dl_param.bg_stat)
	local sequence_stat = hdf5.open(dl_param.sequence_stat)

	if dl_param.gray_scale then
		self.stat.bg_std =  bg_stat:read('patch_std_gray'):all()
		self.stat.bg_mean= bg_stat:read('patch_mean_gray'):all()
		self.stat.sequence_std =  sequence_stat:read('patch_std_gray'):all()
		self.stat.sequence_mean = sequence_stat:read('patch_mean_gray'):all()
		--self.stat.sequence_mean  = self.stat.sequence_mean:permute(2,1)
		--self.stat.sequence_std = self.stat.sequence_std:permute(2,1)
		--self.stat.bg_mean = self.stat.bg_mean:permute(2,1)
		--self.stat.bg_std = self.stat.bg_std:permute(2,1)

	else
		self.stat.bg_std =  bg_stat:read('patch_std_color'):all():permute(3,1,2)
		self.stat.bg_mean = bg_stat:read('patch_mean_color'):all():permute(3,1,2)
		self.stat.sequence_std =  sequence_stat:read('patch_std_color'):all():permute(3,1,2)
		self.stat.sequence_mean = sequence_stat:read('patch_mean_color'):all():permute(3,1,2)

		--self.stat.sequence_mean  = self.stat.sequence_mean:permute(1,3,2)
		--self.stat.sequence_std = self.stat.sequence_std:permute(1,3,2)
		--self.stat.bg_mean = self.stat.bg_mean:permute(1,3,2)
		--self.stat.bg_std = self.stat.bg_std:permute(1,3,2)
	end    

	bg_stat:close()
	sequence_stat:close()

	-- load first h5 files

	self.data.train = loadH5PatchFile(generateFilename(self.h5_info.train.root, self.h5_info.train.start,'data','h5'), true, self.stat.bg_mean ,self.stat.sequence_mean, self.stat.bg_std, self.stat.sequence_std)
	self.data.val = loadH5PatchFile(generateFilename(self.h5_info.val.root, self.h5_info.val.start,'data','h5'), false, self.stat.bg_mean ,self.stat.sequence_mean, self.stat.bg_std, self.stat.sequence_std)
	self.pSize = (#self.data['train'].data)[4]

	-- store idx for batch/file loading
	self.idx_batch = {train=1, val=1}
	self.idx_h5 = {train=self.h5_info.train.start , val=self.h5_info.val.start}

	-- set max iterations
	self.max_iter = {}
	local train_length = (dl_param.stop_train -dl_param.start_train +1)
	self.max_iter.train = math.ceil((#self.data.train.data)[1]/self.batch_size) *  train_length
	self.max_iter.val = math.ceil((#self.data.val.data)[1]/self.batch_size) * (dl_param.sequence_length - train_length)

end


function DataLoader:getNextBatch(id)
	-- returns next batch, opens new file if necessary, returns false if all files and batches have been processed
	assert(id =="val" or id=="train", "id must be either train or val")

	local batch = {}
	local start = (self.idx_batch[id]-1) * self.batch_size + 1
	local stop = math.min(self.data[id].size, self.idx_batch[id] * self.batch_size)

	-- routine if all batches or files are processed	
	if start > (self.data[id].size) then
		if self.idx_h5[id] == self.h5_info[id].stop then
			return false
		end
		self.data[id] = nil
		self.idx_h5[id] = self.idx_h5[id]  + 1

		-- shuffle if loading training data
		local shuffle
		if (id=='train') then
		  shuffle = true
		else
		  shuffle = false
		end
		-- recalculate start and stop indices
		self.data[id] = loadH5PatchFile(generateFilename(self.h5_info[id].root, self.idx_h5[id], 'data', 'h5'), shuffle, self.stat.bg_mean ,self.stat.sequence_mean, self.stat.bg_std, self.stat.sequence_std)
			
		self.idx_batch[id] = 1
		start = (self.idx_batch[id]-1) * self.batch_size + 1
		stop = math.min(self.data[id].size, self.idx_batch[id] * self.batch_size)
	end
	-- data is now [N, 6, pSize, pSize]
	batch.data = self.data[id].data[{{start,stop},{},{},{},{}}]:clone()
	batch.size = stop - start + 1
	batch.label = self.data[id].label[{{start,stop},{}}]:clone()
	
	self.idx_batch[id] = self.idx_batch[id] + 1

	return batch
end

-- Non-member functions
function loadH5PatchFile(h5_file, shuffle, bg_mean, fg_mean, bg_std, fg_std)
	-- loads h5 patch file and returns table with data label and size
	local shuffle = shuffle or false
	local f = hdf5.open(h5_file, 'r')
	local patches = f:read('/patches'):all()
	local labels = f:read('/labels'):all()
	patches = patches:double()
	labels = labels:double()

	-- get patches image in right format [N, 2, 1, psize, psize]
	patches = patches:permute(1,2,5,3,4)
	
	for i = 1, (#patches)[1] do
		-- rescaling and zero mean subtraction
		patches[{i,1,{},{},{}}] = patches[{i,1,{},{},{}}]:div(255.0):add(-fg_mean)
		patches[{i,2,{},{},{}}] = patches[{i,2,{},{},{}}]:div(255.0):add(-bg_mean)

	end
				
	-- shuffle
	if shuffle then
    	local rand_idx = torch.randperm((#patches)[1])
		local rand_patch = torch.zeros(patches:size())
		local rand_label = torch.zeros(labels:size())

    	for t = 1,(#rand_idx)[1] do
			rand_patch[{t,1,{},{},{}}] = patches[{rand_idx[t],1,{},{},{}}]
			rand_patch[{t,2,{},{},{}}] = patches[{rand_idx[t],2,{},{},{}}]
			rand_label[t] = labels[rand_idx[t]]
		end
		patches = rand_patch:clone()
		labels = rand_label:clone()
	end
	local pSize = (#patches)[4]
	

	local data = {
    size = (#patches)[1],
    data = patches:clone(),
    label = labels:clone()
	}
	f:close()
	return data
end


function generateFilename(root, idx, prefix, extension)
	-- appends components to zero filled filename 
	return root .. '/' .. prefix .. string.format('%06d.',idx) .. extension
end


