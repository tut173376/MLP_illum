%% superpixelの内、除外したセグメントを白で表示する

im = im2single(imread('F:\個人フォルダ\M1\Uchimi\testImgs\Img_160928\IMG_0011.jpg'));

% =================================== %
% 画像をリサイズ %
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
% divNum : super pixel による分割数（短辺に対して）
% =================================== %
tic
if ~exist('divNum','var')
    divNum = 20;
end
bsize = min( size(im(:,:,1)) ) / divNum;
seg = vl_slic( im, bsize,  0.1);
fprintf('slic: ');
toc

% saturation map : 黒潰れ, 飽和している画素の位置
if ~exist('lowLv','var')
    lowLv = 0.3;
end
if ~exist('highLv','var')
    highLv = 0.99;
end
satMap = sum( lowLv < im & im < highLv, 3 ) == 0;

% =================================== %
% 各セグメントで境界を計算 %
% =================================== %
perim = true(size(im(:,:,1)));
for k = 0 : max(seg(:))
    perimK = bwperim( seg == k, 4 );
    perim(perimK) = false;
end
% perim8bit = uint8(cat(3,perim,perim,perim));
perim8bit = uint8(repmat(perim,[1,1,3]));

seg = seg + 1; % segの最小を1に
seg(perim(:,:,1)==false) = 0; % 枠のindexを0とする

% show all valid super pixels %
figure(3);
seg2 = seg > 0;
for n = find(invalidSegIdx==true)' % invalidSegIdxは縦ベクトル
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