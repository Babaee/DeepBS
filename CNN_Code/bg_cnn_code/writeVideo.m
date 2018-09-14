inDir = 'E:\Datasets\tracking_data\Crowd_PETS09S1L3\S1\L3\Time_14-17';
outDir = 'C:\Users\Dinh\Desktop\video_data\pets2009\S1\L3\Time_14-17';
category = 'View_002';


imageNames = dir(fullfile(inDir,category,'*.jpg'));
imageNames = {imageNames.name}';

outputVideo = VideoWriter(fullfile(outDir,category,sprintf('%s.avi',category)));
outputVideo.FrameRate = 20;
open(outputVideo)

for ii = 1:length(imageNames)-1
   img = imread(fullfile(inDir,category,imageNames{ii}));
   writeVideo(outputVideo,img)
end

close(outputVideo);