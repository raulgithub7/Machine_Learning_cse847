% Principle Component Analysis: Experiment
clc;
clear;
% load the strcture data & call A
dataset=load('USPS.mat') ;
D=dataset.A; % D marix of size mxd

fig1 = reshape(D(1,:), 16, 16);
imshow(fig1');

% Normalize matrix D to make X matrix
Dbar=mean(D,1);
X=[];
for i=1:size(D,1) % rows
    for j=1:size(D,2) % columns
        X(i,j)=D(i,j)-Dbar(1,j);
    end
end

% compute PCA using SVD
[U,S,V] = svd(X);
score=U*S;
c=0;
% loop for d values
p=[10, 50, 100, 250];
for i=1:size(p,2)
    X_compressed=(score(:,1:p(1,i)))*(V(:,1:p(1,i)))';

    % total error is the Frobenius norm error
    fprintf('For p= %i ', p(i))
    total_error = norm(X - X_compressed, 'fro')

    % save images
    filename1 = ['first_image_reconstructed','_','for_', num2str(p(i)), '.png'];
    A1 = reshape(X_compressed(1,:), 16, 16);
    imwrite(A1',filename1);
    filename2 = ['second_image_reconstructed','_','for_', num2str(p(i)), '.png'];
    A2 = reshape(X_compressed(2,:), 16, 16);
    imwrite(A2',filename2);

end


