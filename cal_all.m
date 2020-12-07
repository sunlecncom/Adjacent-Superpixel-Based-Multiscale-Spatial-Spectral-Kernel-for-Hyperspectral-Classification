function [allcases,dist,feature_labels]=cal_all(im,labels,sigma)
labels=labels+1;
sz=size(labels);
labels=reshape(labels,1,sz(1)*sz(2));
maxsegment=max(labels(:));
feature_labels=zeros(size(im,1),maxsegment);%mean feature of different labeled superpixel
%mean filter within each superpixel
for i=1:maxsegment
    hp_i= labels==i;
    feature_labels(:,i)=mean(im(:,hp_i),2);
end
%kernel computation
 nx_s = sum(feature_labels.^2);
 [X,Y] = meshgrid(nx_s);
 dist=X+Y-2*(feature_labels'*feature_labels);        
 allcases=exp(-dist/2/sigma^2);
     
     
     
    