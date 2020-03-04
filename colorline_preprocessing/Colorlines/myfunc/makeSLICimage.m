%% superpixel�̓��A���O�����Z�O�����g�𔒂ŕ\������

im = im2single(imread('F:\�l�t�H���_\M1\Uchimi\testImgs\Img_160928\IMG_0011.jpg'));

% =================================== %
% �摜�����T�C�Y %
% Ex.) max(size( resizedImg )) = 1024;
% =================================== %
rSize = [NaN, NaN];
[ M, I ] = max(size(im));
if M > 1024 + 100
    rSize(I) = 1024;
    im = imresize( im, rSize );
end
im = max(0,min(1,im));

% =================================== %
% calc segments by SLIC (super pixel) %
% --
% "param"
% divNum : super pixel �ɂ�镪�����i�Z�ӂɑ΂��āj
% =================================== %
tic
if ~exist('divNum','var')
    divNum = 20;
end
bsize = min( size(im(:,:,1)) ) / divNum;
seg = vl_slic( im, bsize,  0.1);
fprintf('slic: ');
toc

% saturation map : ���ׂ�, �O�a���Ă����f�̈ʒu
if ~exist('lowLv','var')
    lowLv = 0.3;
end
if ~exist('highLv','var')
    highLv = 0.99;
end
satMap = sum( lowLv < im & im < highLv, 3 ) == 0;

% =================================== %
% �e�Z�O�����g�ŋ��E���v�Z %
% =================================== %
perim = true(size(im(:,:,1)));
for k = 0 : max(seg(:))
    perimK = bwperim( seg == k, 4 );
    perim(perimK) = false;
end
% perim8bit = uint8(cat(3,perim,perim,perim));
perim8bit = uint8(repmat(perim,[1,1,3]));

seg = seg + 1; % seg�̍ŏ���1��
seg(perim(:,:,1)==false) = 0; % �g��index��0�Ƃ���

% show all valid super pixels %
figure(3);
seg2 = seg > 0;
for n = find(invalidSegIdx==true)' % invalidSegIdx�͏c�x�N�g��
    seg2(seg == n) = false;
end

img = im.*repmat(seg2,[1 1 3]);
for i = 1:size(img,2)
    for j = 1:size(img,1)
        if img(j,i,1) == 0
            img(j,i,:) = 255;
        end
    end
end
img = img.*single(perim8bit);

figure;
subplot(1,2,1);imshow( im.*single(perim8bit) )
subplot(1,2,2);imshow(img)


% output
imwrite( im.*single(perim8bit),  './outImgs/orgImg.jpg' )
imwrite( img , './outImgs/superpixel(SLIC).jpg')