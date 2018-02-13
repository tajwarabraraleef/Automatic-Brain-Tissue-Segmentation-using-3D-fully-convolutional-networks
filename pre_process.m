clear all; close all;

path = 'datasets\Testing\IBSR_0'; %path of preprocessing images
ref = load_untouch_nii(['datasets\Training\IBSR_0' '6.nii.gz']); %Reference image
ref = ref.img;
idxref = find(ref~=0);
%Intensity normalization to 0-255
ref = double(ref);
ref =(ref/double(max(max(max(ref)))));
ref = round(255*ref);
ref = uint8(ref);

for i=11:18
   current_org = load_untouch_nii([path num2str(i) '.nii.gz']);
   current = current_org.img;
   idxcur = find(current~=0);
   %Intensity normalization to 0-255
   current = double(current);   
   current = current/double(max(max(max(current))));
   current = round(current*255); 
   current = uint8(current);
   %Performing histogram matching only on regions except background
   match = imhistmatchn(current(idxcur), ref(idxref),512); 
   new = current;
   new(idxcur) = match;
   current_org.img = new;
   save_untouch_nii(current_org,['IBSR_0' num2str(i) '.nii.gz']) %saving preprocessed volumes
end