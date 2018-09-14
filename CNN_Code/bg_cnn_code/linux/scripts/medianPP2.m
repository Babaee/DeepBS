% spatial median filtering of the dataset output


categories = {};


belief_root = '/usr/home/rez/ZM/CNN/Up_Proj/result_newmodel';
out_root = '/usr/home/rez/ZM/CNN/Up_Proj/result_newmodel_post_04';
% belief_root = '/media/dit/data/Datasets/cdnet2014_subsense_bg_model';
% out_root = '/media/dit/data/Datasets/cdnet2014_subsense_bg_model_corrected';
R = 0.4;
% Get a list of all files and folders in this folder.
files = dir(belief_root);
% Get a logical vector that tells which is a directory.
dirFlags = [files.isdir];
% Extract only those that are directories.
subFolders = files(dirFlags);
% Print folder names to command window.
for k = 1 : length(subFolders)

    if k > 2
        categories{k-2} =  subFolders(k).name;
       
        
        path_ = [belief_root '/'  categories{k-2}];
        
%         try
%             if all(categories{k-2} == 'PTZ')
%                 continue;
%             end
%         catch
%             
%         end

        files_2 = dir(path_);
        
        dirFlags_2 = [files_2.isdir];
        subFolders_2 = files(dirFlags_2);
        videos = {};
        
        for l = 1 : length(files_2)
             if(l>2)
                videos{l-2} =  files_2(l).name;
                fprintf('video name #%d = %s\n', l, videos{l-2});
                try
                
                    for i = 1:10000
                        
                       file_path =  [path_ '/' videos{l-2} '/' sprintf('bin%06d.png', i)];
                       im = im2double(imread(file_path));
                       im = medfilt2(im, [9 9]);
                       im = (im > R);
                         %im = padBg(file_path);  
                       out_path = [out_root '/'  categories{k-2} '/' videos{l-2} '/' sprintf('bin%06d.png', i)];
                       if exist(out_path,'file')
                           break;
                       end
                       imwrite(im, out_path);
                       
                    end
                    
                catch
                    disp(file_path);
                    continue;
            
                end
             end
    
        end
        
    end 
    
end
