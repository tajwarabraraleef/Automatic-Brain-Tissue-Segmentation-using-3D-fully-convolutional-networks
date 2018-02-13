%checking the quantitative results
%Tajwar, Julia

clear all 
clc
close all

path = 'E:\Girona\MISA\Final_project\Code\iSeg2017-nic_vicorob-master\comp\Testing\IBSR_0';
path2 = 'E:\Girona\MISA\Final_project\Code\iSeg2017-nic_vicorob-master\datasets\Testing\IBSR_0';

i=1;
for j=11:15
%Reading ground truth images
if(j<16)
gt = load_untouch_nii([path num2str(j) '_seg.nii.gz']);
gt=gt.img;
end

%reading original images
org = load_untouch_nii([path2 num2str(j) '.nii.gz']);
org=org.img;
orgindx = find(org==0);

%loading corresponding predicted image
predicted_image2 = load_untouch_nii([path num2str(j) '_seg_6_final.nii.gz']);
predicted_image= predicted_image2.img;

%Removing background
predicted_image(orgindx) = 0;
%Performing connected component analysis
predicted_image = connnected(predicted_image);
%Saving post processed image
predicted_image2.img = predicted_image;
save_untouch_nii(predicted_image2,['IBSR_0' num2str(j) '.nii.gz']) 

%Finding Quantitative results
if(j<16)
tempp = find(predicted_image~=0);
predicted_image= predicted_image(find(predicted_image~=0));
gt= gt(tempp);
predicted =reshape(predicted_image,length(predicted_image),1);
gt =reshape(gt,length(gt),1);


ground_truth = gt;
ground_truth(ground_truth==1)=1;
ground_truth(ground_truth==2)=0;
ground_truth(ground_truth==3)=0;

predicted_image = predicted;
predicted_image(predicted_image==1)=1;
predicted_image(predicted_image==2)=0;
predicted_image(predicted_image==3)=0;


P= length(predicted_image(predicted_image==1));%P
T_Positive= predicted_image & ground_truth; %%TP
TP= length(T_Positive(T_Positive==1));
F_positive=( predicted_image & (~ground_truth));%FP
FP= length(F_positive(F_positive==1));
F_negitive=( (~predicted_image) & ground_truth);%FN
FN= length(F_negitive(F_negitive==1));
T_negitive=( (~predicted_image) & (~ground_truth));%TN
TN= length(T_negitive(T_negitive==1));
bothFalse = xor(predicted_image, ground_truth);

AC(i,1)=(TP+TN)/(TP+FP+FN+TN);
DI (i,1)= 2*TP/((2*TP)+FP+FN);
SE(i,1)=TP/(TP+FN);
SP(i,1)=TN/(TN+FP);
JA(i,1)=TP/(TP+FN+FP);

ground_truth = gt;
ground_truth(ground_truth==1)=0;
ground_truth(ground_truth==2)=1;
ground_truth(ground_truth==3)=0;

predicted_image = predicted;
predicted_image(predicted_image==1)=0;
predicted_image(predicted_image==2)=1;
predicted_image(predicted_image==3)=0;


P= length(predicted_image(predicted_image==1));%P
T_Positive= predicted_image & ground_truth; %%TP
TP= length(T_Positive(T_Positive==1));
F_positive=( predicted_image & (~ground_truth));%FP
FP= length(F_positive(F_positive==1));
F_negitive=( (~predicted_image) & ground_truth);%FN
FN= length(F_negitive(F_negitive==1));
T_negitive=( (~predicted_image) & (~ground_truth));%TN
TN= length(T_negitive(T_negitive==1));
bothFalse = xor(predicted_image, ground_truth);


AC(i,2)=(TP+TN)/(TP+FP+FN+TN);
DI (i,2)= 2*TP/((2*TP)+FP+FN);
SE(i,2)=TP/(TP+FN);
SP(i,2)=TN/(TN+FP);
JA(i,2)=TP/(TP+FN+FP);

ground_truth = gt;
ground_truth(ground_truth==1)=0;
ground_truth(ground_truth==2)=0;
ground_truth(ground_truth==3)=1;

predicted_image = predicted;
predicted_image(predicted_image==1)=0;
predicted_image(predicted_image==2)=0;
predicted_image(predicted_image==3)=1;


P= length(predicted_image(predicted_image==1));%P
T_Positive= predicted_image & ground_truth; %%TP
TP= length(T_Positive(T_Positive==1));
F_positive=( predicted_image & (~ground_truth));%FP
FP= length(F_positive(F_positive==1));
F_negitive=( (~predicted_image) & ground_truth);%FN
FN= length(F_negitive(F_negitive==1));
T_negitive=( (~predicted_image) & (~ground_truth));%TN
TN= length(T_negitive(T_negitive==1));
bothFalse = xor(predicted_image, ground_truth);

AC(i,3)=(TP+TN)/(TP+FP+FN+TN);
DI (i,3)= 2*TP/((2*TP)+FP+FN);
SE(i,3)=TP/(TP+FN);
SP(i,3)=TN/(TN+FP);
JA(i,3)=TP/(TP+FN+FP);

i=i+1;
end
end

disp('The average accuracy is:')
mean(AC)
disp('The average dice coefficient is:')
mean(DI)
disp('The average jaccard index is:')
mean(JA)
disp('The average sensitivity is:')
mean(SE)
disp('The average specificity is:')
mean(SP)
mean(DI)
std(DI)