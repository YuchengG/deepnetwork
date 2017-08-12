%% 特征图像绘制
% 微调前和微调后每层稀疏编码机提取的特征图像如下

%% 载入数据
clc
clear
load('all_theta.mat');

num = 6;

%% 微调前第一层可视化图像
data = pre_theta{1}.w;
display_network(data(1:num^2, :)', true, true, num, true); 

%% 微调前第二层可视化图像
data = pre_theta{2}.w * pre_theta{1}.w;
display_network(data(1:num^2, :)', true, true, num, true); 

%% 微调后第一层可视化图像
data = post_theta{1}.w;
display_network(data(1:num^2, :)', true, true, num, true); 

data = post_theta{2}.w * post_theta{1}.w;
%% 微调后第二层可视化图像
display_network(data(1:num^2, :)', true, true, num, true); 

