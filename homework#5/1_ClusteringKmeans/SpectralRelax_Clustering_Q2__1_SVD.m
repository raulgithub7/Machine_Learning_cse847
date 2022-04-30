% Question 1_part2: Spectral Relaxation k-mean clustering algorithm
clc
clear;
% import txt file
data = importdata('Cluster_Data.txt');

% =========== Spectral Relaxation k-means clustering Algorithm ===================
% adjust data to be matrix mxn (m-components column vectors)
X=data';
% SVD
[U,S,V] = svd(X);

% clusters
k=4;
% generate an arbitrary orthogonal matrix: I_k (identity matrix)
sigma =ones(1,k);
Q=diag(sigma);

% eigenvector of X'X is V
Y_star=(V(:,1:k))*Q;

% verify Y'*Y=I_k
verify=(Y_star')*Y_star;

% generate random centroids from Y_star
dim_mat=size(Y_star,2);
mean_mat=mean(Y_star, 1);
BigSigma_mat=cov(Y_star);
% Every row-vector in centroid-matrix is the centroid of a specific cluster
centroids = mvnrnd(mean_mat,BigSigma_mat, dim_mat);

% =========== k-means clustering Algorithm ===================
sz=size(Y_star, 1);
count_current=2*ones(1,k); % initialize this parameter
change=3; % initial value to control the end of "while" loop
iter=0; % initialize this parameter

while change>0
    iter=iter+1 % display the current iteration
    m_ik=zeros(size(Y_star,1),k);

    % fix c_j and determine m_ik 
    [m_ik]=EucledianDist(Y_star, centroids, k, m_ik, sz);

    % re-calculate centroids
    [centroids]=calc_centroids(Y_star, k, m_ik);

    % count re-clustering changes
    count_new=sum(m_ik,1);
    change=max(abs(count_new-count_current));
    count_current=count_new;

    % Get cluster assignment back
    class_list=[];
    for i=1:size(m_ik,1)
        for j=1:size(m_ik,2)
            if m_ik(i,j)==1
                class_list(i,1)=j;
            end
        end
    end

    % plot
    for j=1:k
        subset=[];
        c=0;
        for i=1:size(data,1)
            if class_list(i,1)==j
                c=c+1;
                subset(c,:)=data(i,:);
            end
        end
        
        plot(subset(:,1), subset(:,2),'o')
        xlabel('x coordinate')
        ylabel('y coordinate')
        filename = ['iter','_', num2str(iter), '_', 'plot', '.png'];
        saveas(gcf,filename)
        hold on
    end 
    hold off
end

% =========== Funtions ===================
% Eucledian distance function
function [m_ik]=EucledianDist(x, centroids, k, m_ik, sz)
    
    for i=1:sz
        dist=[];
        for j=1:k
            dist(j)=dot((x(i,:)-centroids(j,:)), (x(i,:)-centroids(j,:)));
        end
        min_val=min(dist);
        index=find(dist==min_val);
        m_ik(i,index)=1;
    end
end

% recalculate centroids function
function [centroids]=calc_centroids(x, k, m_ik)
    centroids=[];
    for j=1:k
        sum_m_ik=sum(m_ik(:,j));
        for pos=1:size(x,2)
            centroids(j,pos)=(sum(m_ik(:,j).*x(:,pos)))/sum_m_ik;
        end
    end
end











    
    