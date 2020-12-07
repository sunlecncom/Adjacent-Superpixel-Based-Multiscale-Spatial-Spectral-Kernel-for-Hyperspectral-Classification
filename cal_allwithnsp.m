%calculate all kernel between two ASs
function allcaseswithnsp=cal_allwithnsp(im,labels,sigma,sigma_r,sz)
[allcases,dist]=cal_all(im,labels,sigma);%calculate all kernel between two superpixels
A=cal_nspkernel( dist,labels, sz,sigma_r);%acquire index and weight of neighbor superpixel
allcaseswithnsp=A*allcases*A';%all kernel between ASs
