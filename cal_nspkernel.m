function [A] = cal_nspkernel( dist,labels, sz,sigma_r)%neighbor superpixel
labels=labels+1;
MaxSegments=max(labels(:));
no_lines=sz(1);no_rows=sz(2);
s=[no_lines, no_rows];
A=zeros(MaxSegments,MaxSegments);

for i=1:MaxSegments
supind=find(labels==i);

    [a,b]=ind2sub(s,supind); 
    %find index of neighbor superpixel
    n1=diag(labels(max(1,a-1),b));
    n2=diag(labels(min(no_lines,a+1),b));
    n3=diag(labels(a,max(1,b-1)));
    n4=diag(labels(a,min(no_rows,b+1)));
    ind=unique([n1;n2;n3;n4]);
    ind(ind==i)=[];
    c=[ind;i];
%assign different for different neighbors    
d=dist(i,c);
weight=exp(-d/(2*sigma_r^2));
A(i,c)=weight./sum(weight(:));
end