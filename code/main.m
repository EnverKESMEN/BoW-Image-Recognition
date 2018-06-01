%Extract all SURF descriptor from all training images
features = ImagesAndDescriptors('airplanes/train');
features = [features ; ImagesAndDescriptors('faces/train')];
features = [features ; ImagesAndDescriptors('leopards/train')];
features = [features ; ImagesAndDescriptors('motorbikes/train')];
features = [features ; ImagesAndDescriptors('watch/train')];

%Find center points whith K-means algorithm for vocabulary
vocabularySize=1000;
[~,vocabulary] = kmeans(features,vocabularySize);

clear features;


%label id's for SVM
%1 for airplanes
%2 for faces
%3 for leopards
%4 for motorbikes
%5 for watch
label=[];
svmFeatures = [ ];


%Find BoW for all aiplanes train image and add svmFeatures matrix
%For each images add '1' to label matrix
imagefiles = dir(['airplanes/train','\*.jpg']);
nfiles = length(imagefiles);    % Number of files found
for ii=1:nfiles
  
   %Read image from folder
   currentfilename = ['airplanes/train','\',imagefiles(ii).name];
   currentimage = rgb2gray(imread(currentfilename));
   
   %extract SURF descriptors
   points = detectSURFFeatures(currentimage);
   [features, valid_points] = extractFeatures(currentimage, points);
   
   %Find minimum euclidian SSD between descriptors and vocabularry
   dist=pdist2(vocabulary,features);
   [firstmin,index]=min(dist);
  
    %Write 0 to all columns
    for i=1 : vocabularySize
        BoW(i)=0;
    end
    
    %Convert features to BoW
    a = size(index,2); 
    for i=1 : a
        BoW(1,index(i)) = BoW(1,index(i)) + 1;
    end
    
    %Add BoW to svmFeatures
    svmFeatures = [svmFeatures;BoW]
    
    %add label '1' for each image
    label = [label;1]
    
end


%Find BoW for all faces train image and add svmFeatures matrix
%For each images add '2' to label matrix
imagefiles = dir(['faces/train','\*.jpg']);
nfiles = length(imagefiles);    % Number of files found
for ii=1:nfiles
  
   %Read image from folder
   currentfilename = ['faces/train','\',imagefiles(ii).name];
   currentimage = rgb2gray(imread(currentfilename));
   
   %extract SURF descriptors
   points = detectSURFFeatures(currentimage);
   [features, valid_points] = extractFeatures(currentimage, points);
   
   %Find minimum euclidian SSD between descriptors and vocabularry
   dist=pdist2(vocabulary,features);
   [firstmin,index]=min(dist);
  
    %Write 0 to all columns
    for i=1 : vocabularySize
        BoW(i)=0;
    end
    
    %Convert features to BoW
    a = size(index,2); 
    for i=1 : a
        BoW(1,index(i)) = BoW(1,index(i)) + 1;
    end
    
    %Add BoW to svmFeatures
    svmFeatures = [svmFeatures;BoW]
    
    %add label '2' for each image
    label = [label;2]
    
end

%Find BoW for all leopards train image and add svmFeatures matrix
%For each images add '3' to label matrix
imagefiles = dir(['leopards/train','\*.jpg']);
nfiles = length(imagefiles);    % Number of files found
for ii=1:nfiles
  
   %Read image from folder
   currentfilename = ['leopards/train','\',imagefiles(ii).name];
   currentimage = rgb2gray(imread(currentfilename));
   
   %extract SURF descriptors
   points = detectSURFFeatures(currentimage);
   [features, valid_points] = extractFeatures(currentimage, points);
   
   %Find minimum euclidian SSD between descriptors and vocabularry
   dist=pdist2(vocabulary,features);
   [firstmin,index]=min(dist);
  
    %Write 0 to all columns
    for i=1 : vocabularySize
        BoW(i)=0;
    end
    
    %Convert features to BoW
    a = size(index,2); 
    for i=1 : a
        BoW(1,index(i)) = BoW(1,index(i)) + 1;
    end
    
    %Add BoW to svmFeatures
    svmFeatures = [svmFeatures;BoW]
    
    %add label '3' for each image
    label = [label;3]
    
end



%Find BoW for all motorbikes train image and add svmFeatures matrix
%For each images add '4' to label matrix
imagefiles = dir(['motorbikes/train','\*.jpg']);
nfiles = length(imagefiles);    % Number of files found
for ii=1:nfiles
  
   %Read image from folder
   currentfilename = ['motorbikes/train','\',imagefiles(ii).name];
   currentimage = rgb2gray(imread(currentfilename));
   
   %extract SURF descriptors
   points = detectSURFFeatures(currentimage);
   [features, valid_points] = extractFeatures(currentimage, points);
   
   %Find minimum euclidian SSD between descriptors and vocabularry
   dist=pdist2(vocabulary,features);
   [firstmin,index]=min(dist);
  
    %Write 0 to all columns
    for i=1 : vocabularySize
        BoW(i)=0;
    end
    
    %Convert features to BoW
    a = size(index,2); 
    for i=1 : a
        BoW(1,index(i)) = BoW(1,index(i)) + 1;
    end
    
    %Add BoW to svmFeatures
    svmFeatures = [svmFeatures;BoW]
    
    %add label '4' for each image
    label = [label;4]
    
end


%Find BoW for all motorbikes train image and add svmFeatures matrix
%For each images add '5' to label matrix
imagefiles = dir(['watch/train','\*.jpg']);
nfiles = length(imagefiles);    % Number of files found
for ii=1:nfiles
  
   %Read image from folder
   currentfilename = ['watch/train','\',imagefiles(ii).name];
   currentimage = rgb2gray(imread(currentfilename));
   
   %extract SURF descriptors
   points = detectSURFFeatures(currentimage);
   [features, valid_points] = extractFeatures(currentimage, points);
   
   %Find minimum euclidian SSD between descriptors and vocabularry
   dist=pdist2(vocabulary,features);
   [firstmin,index]=min(dist);
  
    %Write 0 to all columns
    for i=1 : vocabularySize
        BoW(i)=0;
    end
    
    %Convert features to BoW
    a = size(index,2); 
    for i=1 : a
        BoW(1,index(i)) = BoW(1,index(i)) + 1;
    end
    
    %Add BoW to svmFeatures
    svmFeatures = [svmFeatures;BoW]
    
    %add label '5' for each image
    label = [label;5]
    
end

%train struct with svmtrain
trainingStruct = svmtrain(label,svmFeatures, '-c 100');

clear BoW currentfilename currentimage dist features firstmin i ii imagefiles index  nfiles points  valid_points;


testlabel= [ ];
testFeature=[];


%Find BoW for all airplanes train image and add svmFeatures matrix
%For each images add '1' to label matrix
imagefiles = dir(['airplanes/test','\*.jpg']);
nfiles = length(imagefiles);    % Number of files found
for ii=1:nfiles
   %read image from file
   currentfilename = ['airplanes/test','\',imagefiles(ii).name];
   currentimage = rgb2gray(imread(currentfilename));
   
   %extract SURF descriptors
   points = detectSURFFeatures(currentimage);
   [features, valid_points] = extractFeatures(currentimage, points);
   
   %Find minimum euclidian SSD between descriptors and vocabularry
   dist=pdist2(vocabulary,features);
   [firstmin,index]=min(dist);
  
   
    a = size(index,2);

    %Write 0 to all columns
    for i=1 : vocabularySize
        BoW(i)=0;
    end
    
    %Convert features to BoW
    for i=1 : a
        BoW(1,index(i)) = BoW(1,index(i)) + 1;
    end
    
    %Add BoW to svmFeatures
    testFeature = [testFeature;BoW];
    
    %add label '1' for each image
    testlabel = [testlabel;1];
    
end


%Find BoW for all faces train image and add svmFeatures matrix
%For each images add '2' to label matrix
imagefiles = dir(['faces/test','\*.jpg']);
nfiles = length(imagefiles);    % Number of files found
for ii=1:nfiles
   %read image from file
   currentfilename = ['faces/test','\',imagefiles(ii).name];
   currentimage = rgb2gray(imread(currentfilename));
   
   %extract SURF descriptors
   points = detectSURFFeatures(currentimage);
   [features, valid_points] = extractFeatures(currentimage, points);
   
   %Find minimum euclidian SSD between descriptors and vocabularry
   dist=pdist2(vocabulary,features);
   [firstmin,index]=min(dist);
  
   
    a = size(index,2);

    %Write 0 to all columns
    for i=1 : vocabularySize
        BoW(i)=0;
    end
    
    %Convert features to BoW
    for i=1 : a
        BoW(1,index(i)) = BoW(1,index(i)) + 1;
    end
    
    %Add BoW to svmFeatures
    testFeature = [testFeature; BoW];
    
    %add label '2' for each image
    testlabel = [testlabel;2];
    
end

%Find BoW for all leopards train image and add svmFeatures matrix
%For each images add '3' to label matrix
imagefiles = dir(['leopards/test','\*.jpg']);
nfiles = length(imagefiles);    % Number of files found
for ii=1:nfiles
   %read image from file
   currentfilename = ['leopards/test','\',imagefiles(ii).name];
   currentimage = rgb2gray(imread(currentfilename));
   
   %extract SURF descriptors
   points = detectSURFFeatures(currentimage);
   [features, valid_points] = extractFeatures(currentimage, points);
   
   %Find minimum euclidian SSD between descriptors and vocabularry
   dist=pdist2(vocabulary,features);
   [firstmin,index]=min(dist);
  
   
    a = size(index,2);

    %Write 0 to all columns
    for i=1 : vocabularySize
        BoW(i)=0;
    end
    
    %Convert features to BoW
    for i=1 : a
        BoW(1,index(i)) = BoW(1,index(i)) + 1;
    end
    
    %Add BoW to svmFeatures
    testFeature = [testFeature; BoW];
    
    %add label '3' for each image
    testlabel = [testlabel;3];
    
end

%Find BoW for all motorbikes train image and add svmFeatures matrix
%For each images add '4' to label matrix
imagefiles = dir(['motorbikes/test','\*.jpg']);
nfiles = length(imagefiles);    % Number of files found
for ii=1:nfiles
   %read image from file
   currentfilename = ['motorbikes/test','\',imagefiles(ii).name];
   currentimage = rgb2gray(imread(currentfilename));
   
   %extract SURF descriptors
   points = detectSURFFeatures(currentimage);
   [features, valid_points] = extractFeatures(currentimage, points);
   
   %Find minimum euclidian SSD between descriptors and vocabularry
   dist=pdist2(vocabulary,features);
   [firstmin,index]=min(dist);
  
   
    a = size(index,2);

    %Write 0 to all columns
    for i=1 : vocabularySize
        BoW(i)=0;
    end
    
    %Convert features to BoW
    for i=1 : a
        BoW(1,index(i)) = BoW(1,index(i)) + 1;
    end
    
    %Add BoW to svmFeatures
    testFeature = [testFeature; BoW];
    
    %add label '4' for each image
    testlabel = [testlabel;4];
    
end


%Find BoW for all watch train image and add svmFeatures matrix
%For each images add '5' to label matrix
imagefiles = dir(['watch/test','\*.jpg']);
nfiles = length(imagefiles);    % Number of files found
for ii=1:nfiles
   %read image from file
   currentfilename = ['watch/test','\',imagefiles(ii).name];
   currentimage = rgb2gray(imread(currentfilename));
   
   %extract SURF descriptors
   points = detectSURFFeatures(currentimage);
   [features, valid_points] = extractFeatures(currentimage, points);
   
   %Find minimum euclidian SSD between descriptors and vocabularry
   dist=pdist2(vocabulary,features);
   [firstmin,index]=min(dist);
  
   
    a = size(index,2);

    %Write 0 to all columns
    for i=1 : vocabularySize
        BoW(i)=0;
    end
    
    %Convert features to BoW
    for i=1 : a
        BoW(1,index(i)) = BoW(1,index(i)) + 1;
    end
    
    %Add BoW to svmFeatures
    testFeature = [testFeature; BoW];
    
    %add label '5' for each image
    testlabel = [testlabel;5];
    
end

 [predicted_label,accuracy,asd] = svmpredict(testlabel, testFeature, trainingStruct);
 
 %calculate confusion matris
 [cMatrix,cOrder] = confusionmat(testlabel,predicted_label);
 clear cOrder a Bow currentfilename currentimage dist features firstmin i ii imagefiles index nfiles points vocabularySize valid_points;
 
