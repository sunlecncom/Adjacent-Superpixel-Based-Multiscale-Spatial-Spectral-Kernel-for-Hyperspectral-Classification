clear
close all
clc
%kernel_typer =  'RBF_kernel'
% number of classes
k = 16;
addpath('ers-master');
% number of Monte Carlo runs
MCruns = 10;
%-------------------------------------------------------------------------
%--------------------------------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%  do not modify from this point on
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% monte carlo runs
MMiter = MCruns;%mc  10 times

% ground truth image
load AVIRIS_Indiana_16class
% AVIRIS Indian Pines data set
load imgreal;
im = img;
sz = size(im);
clear img;

im = reshape(im,sz(1)*sz(2),sz(3));
im = im';
im([104:108 150:163 220],:) =[];
sz(3) = size(im,1);
%nomalize the image
im = im./repmat(sqrt(sum(im.^2)),sz(3),1);
%acquire first pc
[~,x]=pca(im');
p=1; % choose the first PC for superpixel segmentation
x=x(:,1:p);%return
img1=reshape(x, sz(1),sz(2), p);
img1=mat2gray(img1);
img1=im2uint8(img1);
trainall = trainall';
% start Monte Carlo runs
CA_SVM = [];
fprintf('\nStarting Monte Carlo runs \n\n');
for iter = 1:MMiter
    fprintf('MC run %d \n', iter);
    % randomly select the initial training set from the ground truth image
    indexes = train_test_random_newvector(trainall(2,:),[2 44 26 8 15 23 2 15 2 30 75 19 7 39 12 3] );
    %indexes = train_test_random_newvector(trainall(2,:), [2 40 24 7 14 24 2 13 2 14 70 15 8 36 11 4]);
    %training samples and index of training samples and ground truth for error calculate
    train1 = trainall(:,indexes);
    test_true = trainall;
    test_true(:,indexes) = [];
    test1 = zeros(2,sz(1)*sz(2));
    test1(1,:) = 1:sz(1)*sz(2);
    test1(2,trainall(1,:)) = trainall(2,:);
    train = im(:,train1(1,:));
    test = im;
    y = train1(2,:);
    y_test = test1(2,:);
    % parameter setting for kernels
    sigma=1/(2^7);sigma_r=0.1; %optimal parameter
    K=0;K_test=0;
    multiscale=[1 2 4 8 16 32];%superpixel scales
    for j=multiscale
        num=j*100;                           %superpixel number
        [labels] = mex_ers(double(img1),num);%do the segment
        allcases=cal_allwithnsp(im,labels,sigma,sigma_r,sz);% calculate kernel matrix(contain all cases that needed) between two ASs
        [K2,a1] = calculateTrainK2(labels,train1,allcases,sz);%indexing out the training kernel matrix
        K_t=calculateTestK2(labels,test1,a1,sz);%indexing out the testing kernel matrix
        K=K+K2;K_test=K_test+K_t;%kernel combination
    end
    K=K./numel(multiscale);K_test=K_test./numel(multiscale);
    n=size(train1,2);
    %-------------------------libsvm for train------------------
    %libSVM kernel
    libK = [(1:n)' K];
    train_label = y';
    model_precomputed = svmtrain(train_label, libK, '-t 4 -g 9.1896 -c 1097.496 ');
    
    %libSVM kernel for test
    %xlib_test = test;
    %----------------------libsvm for testing------------------
    p_svm = [];
    K_test = [(1:sz(1)*sz(2))' K_test];
    [predict_label_P, accuracy_P] = svmpredict((1:sz(1)*sz(2))', K_test, model_precomputed);
    % calculate classification accuracy
    class1=predict_label_P';
    [a.OA,a.kappa,a.AA,a.CA] = calcError( test_true(2,:)-1, class1(test_true(1,:))-1, 1: k);
    AA_SVM(iter) = a.AA;
    Kappa_SVM(iter) = a.kappa;
    CA_SVM= [CA_SVM a.CA];
    OA_SVM(iter)=a.OA
end

%-------------------------------------------------------------------------%
%  evaluation of the algorithm performance
%-------------------------------------------------------------------------%
%%
% compute the mean OAs over MMiter MC runs
mean_SVM_classification = mean(OA_SVM).*100
STD_OA_SVM = std(OA_SVM).*100
mean_AA = mean(AA_SVM).*100
STD_AA_SVM = std(AA_SVM).*100
mean_kappa = mean(Kappa_SVM)
STD_kappa = std(Kappa_SVM)
mean_CA = mean(CA_SVM,2).*100
%classification map
load('IP_gt.mat')
Nonzero_map = zeros(sz(1),sz(2));
Nonzero_index =  find(f ~= 0);
Nonzero_map(Nonzero_index)=1;%
classmap=reshape(class1, sz(1),sz(2));

wholeim=label2color(classmap,'india');
figure,imshow(wholeim,'border','tight');set (gcf,'Position',[700,100,350,350]);

resultmap=classmap.*Nonzero_map;
partofim=label2color(resultmap,'india');
figure,imshow(partofim,'border','tight');set (gcf,'Position',[700,100,350,350]);

gt=label2color(f,'india');
figure,imshow(gt,'border','tight');set (gcf,'Position',[700,100,350,350]);
