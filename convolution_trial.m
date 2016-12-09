clc;
clear;
close all;

mexcuda -lc convolve.cu

H = fspecial('gaussian',30,15);

I = imread('lion.jpg');
Img = double(I(:,:,1));
[my_conv,mat_conv] = myConvolution(Img,H);

figure;
subplot(1,2,1);
imshow(uint8(my_conv));
title('GPU convolution');
subplot(1,2,2);
imshow(uint8(mat_conv));
title('MATLAB convolution');

fprintf('\nMax difference between the two = %f\n',max(max(my_conv-mat_conv)));