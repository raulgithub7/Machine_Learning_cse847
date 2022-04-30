% Clustering Q2: generate 1,000 bivariate random patterns from each of the three densities
clc
clear;

miu1=[0;3];
BigSigma1=[1,0;0,1];
n1=1000;
R_class1 = mvnrnd(miu1,BigSigma1,n1);
Val_class1=transpose(R_class1);

miu2=[6;2];
BigSigma2=[1,0;0,1];
n2=1000;
R_class2 = mvnrnd(miu2,BigSigma2,n2);
Val_class2=transpose(R_class2);

miu3=[5;9];
BigSigma3=[1,0;0,2];
n3=1000;
R_class3 = mvnrnd(miu3,BigSigma3,n3);
Val_class3=transpose(R_class3);

miu4=[0;8];
BigSigma4=[1,0;0,1];
n4=1000;
R_class4 = mvnrnd(miu4,BigSigma4,n4);
Val_class4=transpose(R_class4);

% plots
plot(R_class1(:,1),R_class1(:,2), 'o')
hold on
plot(R_class2(:,1),R_class2(:,2), 'o')
hold on
plot(R_class3(:,1),R_class3(:,2), 'o')
hold on
plot(R_class4(:,1),R_class4(:,2), 'o')
hold off

legend('omega1','omega2', 'omega3', 'omega4')

kmeanCluster_data=[R_class1;R_class2;R_class3;R_class4];
writematrix(kmeanCluster_data,'Cluster_Data.txt','Delimiter',' ') 



    
    