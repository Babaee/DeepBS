% function im = padBg(image)
%     im = imread(image);
% 
%     % remove zero boundaries
%     im2 = im(1+2:end-2,1+2:end-2,:);
%     % pad image with repetition
%     im(1+2:end-2,1:2,:) = im2(:,1:2,:);
%     im(1+2:end-2,end-2:end,:) = im2(:,end-2:end,:);
% 
%     im2 = im(1+2:end-2,:,:);
% 
%     im(1:2,:,:) = im2(1:2,:,:);
%     im(end-2:end,:,:) = im2(end-2:end,:,:);
% end

                             
in_root = '/media/dit/data/Datasets/bgs_segmentation/subsense/wallflower/TimeOfDay/bg_model'; 
out_root = '/media/dit/data/Datasets/bgs_segmentation/subsense/wallflower/TimeOfDay/bg_model';

for i = 1:10000
   image = [in_root sprintf('/bin%06d.png', i)]; 
   out_im = [out_root sprintf('/bin%06d.png', i)];
   

    im = imread(image);

    % remove zero boundaries
    im2 = im(1+2:end-2,1+2:end-2,:);
    % pad image with repetition
    im(1+2:end-2,1:2,:) = im2(:,1:2,:);
    im(1+2:end-2,end-2:end,:) = im2(:,end-2:end,:);

    im2 = im(1+2:end-2,:,:);

    im(1:2,:,:) = im2(1:2,:,:);
    im(end-2:end,:,:) = im2(end-2:end,:,:);

    imwrite(im,out_im);
end



                                                                                                                                                                                                                                                                                                                    