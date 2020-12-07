function K2test = calculateTestK2(labels,test1,a1,sz) %
Samplepos=test1(1,:);   labels=labels+1;
[p,q]=meshgrid(1:sz(2),1:sz(1));
snum=size(Samplepos,2); 
M=max(labels(:));

n0=1;K2test=[];
%make a loop incase the short of computer memory 
nsplits = floor(snum/100000)+1;
for i=1:nsplits
    if i==nsplits                                  
    n1=snum-n0+1;                            
    else
    n1=floor(snum/nsplits);         
    end 
A2=zeros(M,n1);
%find the index of testing samples
for j=n0:n0+n1-1
q1=q(Samplepos(j));
p1=p(Samplepos(j));
    label=labels(q1,p1);
    A2(label,j-n0+1)=1;
end
% index the testing kernel out
n0=n0+n1;
K2test_split=A2'*a1;
K2test=[K2test;K2test_split];
end
end