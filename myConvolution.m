function [A,B] = myConvolution(Matrix,kernel)
    
    tic;
    B = conv2(Matrix,kernel,'same');
    conv2_time = toc;
    fprintf('Matlab conv2 convolution takes %f\n',conv2_time);
    
    tic;
    A = convolve(Matrix,flipud(fliplr(kernel)));
    my_time = toc;
    fprintf('GPU convolution takes %f\n',my_time);
    
return