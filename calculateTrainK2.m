function [K2,a1] = calculateTrainK2(labels,train1,allcases,sz) %
Samplepos=train1(1,:);   labels=labels+1;
[p,q]=meshgrid(1:sz(2),1:sz(1));
snum=size(Samplepos,2); 
A=zeros(max(labels(:)),snum);
%find the index of training samples
for j=1:snum
q1=q(Samplepos(j));
p1=p(Samplepos(j));
    label=labels(q1,p1);
    A(label,j)=1;
end
%index the training kernel out 
a1=allcases*A;
K2=A'*a1;



