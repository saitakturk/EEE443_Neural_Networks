%THIS IS THE SOLUTION FOR QUESTION 2.YOU SHOULD COPY THIS CODE AND PASTE PROPER .M FILE AND RUN IN MATLAB( COPY ONLY CODES IN MULTI LINE COMMENT)
%===============================Q2 MATLAB======================================================================
function sait_akturk_21501734_hw3(question)
clc
close all

switch question
    case '2'
	disp('2');
        %% question 2 code goes here
	q2();
           
end

end




function q2()
%Main function to run q2 for assignment 3 

%load image patches
[d, invX, x] = load_data();

%normalized image patches
normalized_data2 = normalize_patches(d);
%show all patches and normalized patches
show_images(d, normalized_data2);


%q2-2
%optimal hyper parameters
P.Lin = 256; %input layer dimension 
P.Lhid = 64; %hidden layer dimension
options.MaxIter = 200;
P.lambda = 5 * 1e-4;
P.rho= 0.010;
P.beta = 2.5; 



%weight and bias initialization
W_input = weight_and_bias_init(P.Lin, P.Lhid); 

%cost and gradients
J  = @(W_input) aeCost(W_input, normalized_data2, P);

%get updated weights
[Weights, fx, i]= fmincgrad(J, W_input,options);

%reshape w1 to ideal 
W1 = reshape(Weights(1 : P.Lhid*P.Lin), P.Lhid, P.Lin);

%show weights
show_weights(W1');


end 

function [ normalized ] = normalize_patches(d)
%This function normalize patches according to assignment 
%Params:
%   d : all image patches
%Return:
%   normalized : normalized image patches

%rgb to gray patches
post_d = gray_data(d);

%flatten
flat_post_d = flatten(post_d);

%remove the mean of all patches from image patches
normalized_d = find_mean_insensity(flat_post_d);



%divide std
stdx = 3 * std(normalized_d(:));
mean = max(min(normalized_d, stdx), -stdx) / stdx;

%make range [0.1, 0.9]
normalized = (mean + 1) * 0.4 + 0.1;

end 
function [weights ] = weight_and_bias_init(in, hid )
%This function initialize weights and biases
%Params:
%   in : input size 
%   hid : hidden size 
%Return:
%   weights : weights and biases 1D

%initialize weights 
w0  = sqrt(6) / sqrt(in+hid+1);   
w1 = rand(hid, in) * 2 * w0 - w0;
w2 = rand(in, hid) * 2 * w0 - w0;
%initialize biases
b1 =  rand(hid, 1) * 2 * w0 - w0;
b2 =  rand(in, 1) * 2 * w0 - w0;

weights = [ w1(:); w2(:); b1(:); b2(:)];

end 

function  show_images(d, normalized_data2)
%This function show patches and normalized patches
%Params:
%   d : all image patches
%   normalize_data2 : normalized image patches
%Return:
%   

%reshape data to plot
reshaped_data2 = reshape(normalized_data2, [ 16 , 16 ,10240 ] );

%empty matrix to show image patches 
rgb_images = zeros( 10 * 16, 20 * 16 ,3 );
%empty matrix to show normalized image patches 
normalized_images = zeros( 10 * 16, 20 * 16 ) ; 

%random image patches 
for i = 1:10
    for j = 1:20  
       rgb_images(((i-1) * 16 )+1 :i * 16, ((j-1) * 16 )+1 :j * 16, :) = d(:,:,:,((i-1)*20 + j)+200 );
       normalized_images(((i-1) * 16 )+1 :i * 16, ((j-1) * 16 )+1 :j * 16) = reshaped_data2(:,:,((i-1)*20 + j)+200);
    end
end

%plot random image patches
figure;
subplot(2,1,1)
imshow(rgb_images);
title('rgb patches');
subplot(2,1,2)
imshow(normalized_images);
title('normalized patches');



end

function mean_insensities = find_mean_insensity(flat_post_d)
%This function remove mean from image patches
%Params:
%   flat_post_d : all flatten gray image patches
%Return:
%   mean_insensities : removed mean image patches

%remove image mean
mean_insensities = bsxfun(@minus, flat_post_d, mean(flat_post_d));

end
function flat_post_d = flatten(post_d)
%This function flatten data format
%Params:
%   post_d  : image patches
%Return:
%   flat_post_d : flatten image patches

%size of array
flat_post_d = reshape(post_d,[256, 10240] );

end

function post_d = gray_data(d)
%This function conver rgb to gray
%Params:
%   d : all image patches
%Return:
%   post_d : all gray image patches
post_d =  0.2126 * d(:,:,1,:) + 0.7152 * d(:,:,2,:) + 0.0722 * d(:,:,2,:);


end 

function [d, invXForm, xForm ] = load_data()
%read the data to struct
data2 = load('assign3_data2.mat');
%convert data types 
data2_data = struct2cell( data2(1) );

%size 16x16x3x10240
d = cell2mat( data2_data(1) );
invXForm = cell2mat(data2_data(2));
xForm = cell2mat( data2_data(3));
end

function [h, array] = show_weights(weights)





%  rescaling for constrast adjusting
weights = weights - mean(weights(:));

% find rows and columns
[L number_of_filter] = size(weights);

%weight height default 16 
weight_height = sqrt(L);

%divisor of number of filters 
div = divisors(number_of_filter);

%find size of divisor number
[temp, no_of_div] = size(div);

%middle divisor to shape overall image
image_height = div(round(no_of_div/2));

%image width to show weight images
image_width = number_of_filter / image_height;

%empty aray to show all weigths
patches = zeros(image_height*(weight_height), image_height*(weight_height));



%count weights number
count = 1;
%set weights
for j = 1 : image_height
    for i = 1 : image_width
        if count > number_of_filter
            continue; 
        end
        %normalize weights to adjust constrast
        patches((i-1)*(weight_height)+(1:weight_height), (j-1)*(weight_height)+(1:weight_height)) = reshape(weights(:, count), weight_height,weight_height) / max(abs(weights(:, count)));
      
        count = count + 1;
    end
end
%plotweights
figure;
w = imagesc(patches, [-1 1]);
colormap(gray);
axis image off

drawnow;

end