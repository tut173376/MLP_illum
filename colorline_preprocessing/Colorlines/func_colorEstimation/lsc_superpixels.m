%% �摜���Z�O�����g������(LSC)
function seg = lsc_superpixels(img, superpixelNum, ratio)

if nargin == 2
    ratio=0.075; % dafault(thesis recommends)
end

gaus = fspecial('gaussian',3);
I = imfilter(uint8(img*255),gaus);

seg=LSC_mex(I, superpixelNum, ratio);

mask = boundarymask(seg);
% perim8bit = uint8(repmat(~mask,[1,1,3])); % perimeter=0

seg(mask==true) = 0; % �g��index��0�Ƃ���

% imshow( imoverlay( double(img)./255 ,mask ,'black') ) % ���摜
% imshow( imoverlay( double(label)./double(max(max(label))) ,mask ,'black') ) % ���x��
