clear all
close all
clc
%kernel_typer = 'RBF_kernel'
addpath('ers-master');
% number of classes
k = 9;

% number of Monte Carlo runs
MCruns = 10;
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%  do not modify from this point on
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% total initial training sample
n_train =360;
% number of initial traing samples per class
train_perclass = round(n_train/k);
% monte carlo runs
MMiter = MCruns;

% ground truth image
load Pavia_University_9class;
% AVIRIS Indian Pines data set
load PaviaU;
im = paviaU;
sz = size(im);
clear paviaU;

im = reshape(im,sz(1)*sz(2),sz(3));
im = im';
trainall = Pavia_University_9class';
clear Pavia_University_9class;
%nomalize the image
im = im./repmat(sqrt(sum(im.^2)),sz(3),1);
%acquire first pc
[~,x]=pca(im');
p=1;x=x(:,1:p);
img1=reshape(x, sz(1),sz(2), p);
img1=mat2gray(img1);
img1=im2uint8(img1);

%load the training samples
CA_SVM = [];
fprintf('\nStarting Monte Carlo runs \n\n');
Choose_Type = 'Random';
%Choose_Type = 'Fixed';
for iter = 1:MMiter
    tic;
    fprintf('MC run %d \n', iter);
    switch Choose_Type
        case 'Random'
            indexes = train_test_random_newvector(trainall(2,:),[15 15 15 15 15 15 15 15 15] );
            train1 = trainall(:,indexes);
            test1 = trainall;
            test1(:,indexes)=[];
        case 'Fixed'
            indexes = train_test_random_newvector(PaviaU_train(2,:),[100 100 100 100 100 100 100 100 100]);
            train1 = PaviaU_train(:,indexes);
            test1 = trainall;
    end
    % start active section iterations
    train = im(:,train1(1,:));
    test = im(:,test1(1,:));
    y = train1(2,:);
    % parameter setting for kernels
    sigma=1/(2^5);sigma_r=0.1;     %optimal sigma
    K=0;K_test=0;
    multiscale=[ 2 4 8 16 32 64];
    for j=multiscale
        num=j*100;% superpixel number
        [labels] = mex_ers(double(img1),num);%do the segment
        allcases=cal_allwithnsp(im,labels,sigma,sigma_r,sz);% calculate kernel matrix(contain all cases that needed) between two ASs
        [K2,a1] = calculateTrainK2(labels,train1,allcases,sz);%indexing out the training kernel matrix
        K_t=calculateTestK2(labels,1:sz(1)*sz(2),a1,sz);%indexing out the testing kernel matrix
        K=K+K2;K_test=K_test+K_t;%kernel combination
    end
    K=K/(numel(multiscale));K_test=K_test/(numel(multiscale));
    
    n=size(train1,2);
    %-------------------------libsvm for train------------------
    %libSVM kernel
    libK_newkernel = [(1:n)' K];
    train_label = y';
    
    model_precomputed2 = svmtrain(train_label, libK_newkernel, '-t 4 -g 1.8661 -c 630.3459');
    %libSVM kernel for test
    %xlib_test = test;
    %----------------------libsvm for testing------------------
    K_test=[(1:sz(1)*sz(2))' K_test];
    [predict_label_P_newkernel, accuracy_P_newkernel] = svmpredict((1:sz(1)*sz(2))', K_test, model_precomputed2);
    class1=predict_label_P_newkernel';
    [a.OA,a.kappa,a.AA,a.CA] = calcError( test1(2,:)-1, class1(test1(1,:))-1, 1: k);
    
    % calculate classification accuracy
    AA_SVM(iter) = a.AA;
    Kappa_SVM(iter) = a.kappa;
    CA_SVM = [CA_SVM a.CA];
    OA_SVM(iter)=a.OA
    toc;
end
%%
%-------------------------------------------------------------------------%
%  evaluation of the algorithm performance
%-------------------------------------------------------------------------%
mean_SVM_classification = mean(OA_SVM).*100
STD_SVM_OA = std(OA_SVM).*100
mean_AA = mean(AA_SVM).*100
STD_AA = std(AA_SVM).*100
mean_kappa = mean(Kappa_SVM)
STD_kappa = std(Kappa_SVM)
mean_CA = mean(CA_SVM,2).*100
%
%
%%clssification map
load('PaviaU_gt.mat');f=paviaU_gt;
Nonzero_map = zeros(sz(1),sz(2));
Nonzero_index =  find(f ~= 0);
Nonzero_map(Nonzero_index)=1;%
classmap=reshape(class1, sz(1),sz(2));

wholeim=label2color(classmap,'uni');
figure,imshow(wholeim,'border','tight');set (gcf,'Position',[700,100,sz(2),sz(1)]);

resultmap=classmap.*Nonzero_map;
partofim=label2color(resultmap,'uni');
figure,imshow(partofim,'border','tight');set (gcf,'Position',[700,100,sz(2),sz(1)]);

gt=label2color(f,'uni');
figure,imshow(gt,'border','tight');set (gcf,'Position',[700,100,sz(2),sz(1)]);