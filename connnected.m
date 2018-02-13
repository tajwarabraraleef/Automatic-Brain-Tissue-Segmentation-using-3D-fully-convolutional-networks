function out = connnected(image)
%performing connected component analysis on each separate class and onlly
%keeping the biggest component
new =zeros(size(image));
class1 = new;
class1(find(image==1))=1;
class2 = new;
class2(find(image==2))=2;
cc = bwconncomp(class2);
numPixels = cellfun(@numel,cc.PixelIdxList);
[biggest,idx] = max(numPixels);
class2(find(class2==2)) = 3;
class2(cc.PixelIdxList{idx}) = 2;
class3 = new;
class3(find(image==3))=3;
cc = bwconncomp(class3);
numPixels = cellfun(@numel,cc.PixelIdxList);
[biggest,idx] = max(numPixels);
class3 = new;
class3(cc.PixelIdxList{idx}) = 3;

out  = class1 + class2 + class3;

end
