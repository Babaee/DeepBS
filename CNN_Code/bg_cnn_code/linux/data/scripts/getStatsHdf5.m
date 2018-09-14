sequence = '/usr/home/rez/ZM/CNN/Up_Proj/train';
output_folder = '/usr/home/rez/ZM/CNN/Up_Proj/stats';
start = 0;
end_id = 149;

N = end_id - start + 1;

patchSize = 37;
gray = false;

if gray
sum_input = zeros(patchSize);
sum_bg = zeros(patchSize);
else
sum_input = zeros(3,patchSize,patchSize);
sum_bg = zeros(3,patchSize,patchSize);
end

h = waitbar(0,'Initializing waitbar...');
%% calculate mean
for i=start:end_id
   waitbar((i-start)/N,h,sprintf('%.2f%% along...',(i-start)/N * 100))
   % open file and extract bg, input patches
   imPatches = h5read([sequence sprintf('/data%06d.h5', i)], '/patches');
   
   in_patches = imPatches(:,:,:,1,:);
   bg_patches = imPatches(:,:,:,2,:);

   in_patches = permute(in_patches, [5 1 3 2 4]);
   bg_patches = permute(bg_patches, [5 1 3 2 4]);
   
   if gray
       sum_input = sum_input +reshape(sum(in_patches,1), patchSize, patchSize);
       sum_bg = sum_bg +reshape(sum(bg_patches,1), patchSize, patchSize);
   else
       sum_input = sum_input +reshape(sum(in_patches,1), 3, patchSize, patchSize);
       sum_bg = sum_bg +reshape(sum(bg_patches,1), 3, patchSize, patchSize);   
   end
end
close(h);

sz = size(in_patches);
patches_per_file = sz(1);

mean_input = sum_input/(N * patches_per_file);
mean_bg = sum_bg/(N * patches_per_file);

%% calc std
if gray
sum_input = zeros(patchSize);
sum_bg = zeros(patchSize);
m_in = permute(repmat(mean_input,1,1,patches_per_file),[3 1 2]); 
m_bg = permute(repmat(mean_bg,1,1,patches_per_file),[3 1 2]);
else
sum_input = zeros(3,patchSize,patchSize);
sum_bg = zeros(3,patchSize,patchSize);
m_in = permute(repmat(mean_input,1,1,1,patches_per_file),[4 1 2 3]); 
m_bg = permute(repmat(mean_bg,1,1,1,patches_per_file),[4 1 2 3]);
end
h = waitbar(0,'Initializing waitbar...');



% calculate mean
for i=start:end_id
   waitbar((i-start)/N,h,sprintf('%.2f%% along...',(i-start)/N * 100))
   
   % open file and extract bg, input patches
   imPatches = h5read([sequence sprintf('/data%06d.h5', i)], '/patches');
   
   in_patches = imPatches(:,:,:,1,:);
   bg_patches = imPatches(:,:,:,2,:);

   in_patches = permute(in_patches, [5 1 3 2 4]);
   bg_patches = permute(bg_patches, [5 1 3 2 4]);
   
  if gray
   sum_input = sum_input + reshape(sum((double(in_patches) - m_in).^2,1), patchSize, patchSize);
   sum_bg = sum_bg + reshape(sum((double(in_patches) - m_bg).^2,1),patchSize, patchSize);
  else
   sum_input = sum_input + reshape(sum((double(in_patches) - m_in).^2,1), 3, patchSize, patchSize);
   sum_bg = sum_bg + reshape(sum((double(in_patches) - m_bg).^2,1), 3, patchSize, patchSize);
  end
end
close(h);
std_input = sqrt(sum_input/(N * patches_per_file));
std_bg = sqrt(sum_bg/(N * patches_per_file));


%% save stats
bg_stat = [output_folder '/bg_stat.h5'];
in_stat = [output_folder '/sequence_stat.h5'];
if gray
    hdf5write(bg_stat, '/patch_mean_gray', mean_bg/255, '/patch_std_gray', std_bg/255);
    hdf5write(in_stat, '/patch_mean_gray', mean_input/255, '/patch_std_gray', std_input/255);
else
    hdf5write(bg_stat, '/patch_mean_color', mean_bg/255, '/patch_std_color', std_bg/255);
    hdf5write(in_stat, '/patch_mean_color', mean_input/255, '/patch_std_color', std_input/255);
end

%%
% bg_stat = '/media/dit/data/Datasets/patch_data/highway/stats/bg_stat.h5';
% in_stat = '/media/dit/data/Datasets/patch_data/highway/stats/bg_stat.h5';
% bg_mean_stat = h5read(bg_stat, '/patch_mean_gray');
% in_mean_stat = h5read(in_stat, '/patch_mean_gray');
% 
% disp(bg_mean_stat - mean_bg/255);
% disp(in_mean_stat - mean_input/255);
%%

%imshow(permute(mean_input,[3 2 1]))