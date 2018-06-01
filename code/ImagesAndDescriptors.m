function [images] = ImagesAndDescriptors(path)
imagefiles = dir([path,'\*.jpg']);      
images = []
nfiles = length(imagefiles);    % Number of files found
for ii=1:nfiles
   currentfilename = [path,'\',imagefiles(ii).name];
   currentimage = rgb2gray(imread(currentfilename));
   points = detectSURFFeatures(currentimage);
   [features, valid_points] = extractFeatures(currentimage, points);
   images = [images ; features];
end