% Q21_part1: k-mean clustering algorithm
clc
clear;
% import txt file
data = importdata('Cluster_Data.txt');

% plot original data
plot(data(:,1),data(:,2), 'o')
hold on

% ==================== Alternating procedure, k-means clustering algorithm ======================
% choose K
k=4; %<-- INPUT VALUE

% initial centroids
x_min=min(data(:,2));
x_max=max(data(:,2));
delta_x=(x_max-x_min);
y_min=min(data(:,1));
y_max=max(data(:,1));
delta_y=(y_max-y_min);

miu=[x_min;y_max];
BigSigma=[delta_x,0;0,delta_y];
centroids = mvnrnd(miu,BigSigma,k);
% plots
plot(centroids(:,1),centroids(:,2), '+', 'MarkerSize',15, 'LineWidth',4)
xlabel('x coordinate')
ylabel('y coordinate')
hold off
saveas(gcf,'original_data.png')

% Algorithm
x=data;
sz=size(x, 1);
count_current=2*ones(1,k);
change=3;
iter=0
while change>1
    iter=iter+1
    m_ik=zeros(size(data,1),k);

    % fix c_j and determine m_ik
    [m_ik]=EucledianDist(x, centroids, k, m_ik, sz);

    % re-calculate centroids
    [centroids]=calc_centroids(x, k, m_ik);

    % count re-clustering changes
    count_new=sum(m_ik,1);
    change=max(abs(count_new-count_current));
    count_current=count_new;

    % plot & save plot
    for i=1:k
        rmv_x=x;
        rmv_x(m_ik(:,i)==0,:)=[];
        plot(rmv_x(:,1), rmv_x(:,2),'o')
        hold on
    end
    plot(centroids(:,1),centroids(:,2), '+', 'MarkerSize',15, 'LineWidth',4)
    xlabel('x coordinate')
    ylabel('y coordinate')
    hold off
    filename = ['iter','_', num2str(iter), '_', 'plot', '.png'];
    saveas(gcf,filename)


end

% ============================ Functions ===================================
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
        centroids(j,1)=(sum(m_ik(:,j).*x(:,1)))/sum_m_ik;
        centroids(j,2)=(sum(m_ik(:,j).*x(:,2)))/sum_m_ik;           
    end
end










    
    