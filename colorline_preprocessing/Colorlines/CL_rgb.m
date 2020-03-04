clear
clc
close all
warning off

% =================================== %
% �p�����[�^��ݒ�
% =================================== %
divNum = 20;

scale = 200; %100;%200;

lowLv = 0.3;%0.3;
highLv = 0.95;
th_prc = 99;

th_error = 300;%3500; %2500 %3000 %5000
th_pixNum = 100;%50;

% 2��������pi/32, 3��������pi/4�������݂����c
th_angle = pi/32; %pi/8; %pi/12; %pi/16;

% gmm:4, kmeans:3
%clusteringMode = 'gmm'; K = 4;
 clusteringMode = 'kmeans'; %K = 3;
% clusteringMode = 'clusterdata';

CLcut = 'true';
%CLcut = 'false';

th_clusterSize_alpha = 0.0; % 0.25
th_clusterMax_alpha = 0.0; % 0.25

% =================================== %
% read image %
% =================================== %

% fName = 'C:\Users\jinno\Documents\MATLAB\_myPrograms\_IntrinsicImage\Multi-IntrinsicImage\samples\';
% % im = im2single(imread([fName '16000K_2.JPG']));
% light_xy = [ 0.26, 0.26 ];
% % im = im2single(imread([fName 'blue_2.JPG']));
% im = im2single(imread([fName 'pink_2.JPG']));
% % im = im2single(imread([fName 'green_2.JPG']));

% im = im2single(imread('imgs\IMG_0768.JPG'));
% im = im2single(imread('imgs\IMG_4453.JPG'));

%  im = im2single(imread('sampleDatas\20160210\_MG_5694.JPG'));
% light_xy = [ 0.25, 0.5; 0.2, 0.2 ]; % left[0.25,0.5], right[0.2,0.2]
%   im = im2single(imread('sampleDatas\20160210\_MG_5696.JPG'));
%  light_xy = [ 0.6, 0.3; 0.25, 0.5 ]; % left[0.6,0.3], right[0.25,0.5]
   im = im2single(imread('X:\testImgs\colorChart\IMG_0002.jpg'));
   light_xy = [ 0.2, 0.2; 0.6, 0.3 ]; % left[0.2,0.2], right[0.6,0.3]

% im = im2single(imread('sampleDatas\20160304\_MG_5715.JPG'));
% im = im2single(imread('sampleDatas\20160304\_MG_5712.JPG'));
% light_xy = [ 0.3, 0.45; 0.25, 0.25; 0.6, 0.3 ]; % left-green[0.3,0.45], right-blue[0.25,0.25], up-red[0.6,0.3]

%im = im2single(imread('sampleDatas\20160304\_MG_5710.JPG'));
% im = im2single(imread('sampleDatas\20160304\_MG_5705.JPG'));
% light_xy = [ 0.25, 0.5; 0.2, 0.2; 0.6, 0.3 ]; % left-green[0.25,0.5], right-blue[0.2,0.2], up-red[0.6,0.3]


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


% =================================== %
% change colorSpace %
% --
% lab : [0,100], L : [0, 1], uv : [0, 0.7]
% func "rgb2xyz" is designed for nonlinear image
% =================================== %
ycbcr = rgb2ycbcr( cast(im, 'uint8') );
lab = rgb2lab( im, 'WhitePoint', 'd65', 'ColorSpace', 'adobe-rgb-1998' );
XYZ = rgb2xyz( im, 'WhitePoint', 'd65', 'ColorSpace', 'adobe-rgb-1998' );

L = lab(:,:,1)/100;
u = 4*XYZ(:,:,1) ./ ( XYZ(:,:,1) + 15*XYZ(:,:,2) + 3*XYZ(:,:,3) );
v = 9*XYZ(:,:,2) ./ ( XYZ(:,:,1) + 15*XYZ(:,:,2) + 3*XYZ(:,:,3) );

r = im(:,:,1); g = im(:,:,2); b = im(:,:,3);

% =================================== %
% make map %
% --
% + saturation map
% + edge(laplacian) map
% --
% "param"
% lowLv : ����ȉ��̒��f�lRGB���܂މ�f�����ׂ��f�Ƃ���
% highLv : ����ȉ��̍���f�lRGB���܂މ�f��O�a��f�Ƃ���
% th_prc : ��ʁi100 - th_prc�j���̋����G�b�W���܂ޗ̈������
% =================================== %

% saturation map : ���ׂ�, �O�a���Ă����f�̈ʒu
if ~exist('lowLv','var')
    lowLv = 0.3;
end
if ~exist('highLv','var')
    highLv = 0.99;
end
satMap = sum( lowLv < im & im < highLv, 3 ) == 0;

% edge(laplacian) map : �G�b�W�͒e��
if ~exist('th_prc','var')
    th_prc = 99;
end
lapMap = abs( vl_imgrad(L) ); % sqrt( vl_imgrad(L).^2 + vl_imgrad(u).^2 + vl_imgrad(v).^2 );
lapMap = lapMap > prctile(lapMap(:),th_prc);
% lapMap = zeros( size(L) );


% =================================== %
% decide param for u'v' chromaticity diagram %
% --
% ���� ���j, "�F�x�}�̒��F", �����H�|��w�H�w���I�v Vol.36, No.1, 2013.
% fileName: �F�x�}�̕\���@_vol36-1-09.pdf
% --
% "param"
% scale : uv�F�x�}��̕������D�傫������Ɛ��x�����シ�邪�C��_���d�Ȃ炸�ɃG���[�ƂȂ�m����
% =================================== %
if ~exist('scale','var')
    scale = 200;
end
vrange = [ 0, 1, 0, 1 ]; % [ 0, 1, 0, 1 ]
range = vrange * scale;
plotX = round( range(1) : range(2) );

uu = vrange(1) : 1/scale : vrange(2);
vv = vrange(3) : 1/scale : vrange(4);

uLeng = length(uu); % 201;
vLeng = length(vv); % 201;

bg_u = repmat(uu,[vLeng,1]);
bg_v = repmat(vv',[1,uLeng]);

bg_x = 9*bg_u ./ ( 6*bg_u - 16*bg_v + 12 );
bg_y = 4*bg_v ./ ( 6*bg_u - 16*bg_v + 12 );
bg_z = 1 - bg_x - bg_y;

bg_rgb = xyz2rgb( cat(3,bg_x,bg_y,bg_z), 'WhitePoint', 'd65', 'ColorSpace', 'adobe-rgb-1998' );

bg = round( 255*bg_rgb );
outerMap = bg<0 | bg>255;
outerMap = outerMap(:,:,1) | outerMap(:,:,2) | outerMap(:,:,3);
bg( repmat(outerMap,[1 1 3]) ) = 128;
clear bg_*

% =================================== %
% �L����colorlines���v�Z %
% --
% "param"
% th_error : ���e���钼���ߎ��덷
% =================================== %
tic
if ~exist('th_error','var')
    th_error = 50; %150; %3500; % 2500 % 3000 % 5000
end
if ~exist('th_pixNum','var')
    th_pixNum = 50;
end
segNum = max(seg(:));
allCoef = -ones(segNum,7); % [ a1, b1, error1;... ] = [ -1, -1, -1;... ]

cube = 1;
if cube
    skip = 0.2;
    for j = 0:skip:1
        for i = 0:skip:1
            
            plot3([j j], [0 1], [i, i], 'k-')
            hold on
        end
    end
    
    for j = 0:skip:1
        for i = 0:skip:1
            
            plot3([0 1], [j j], [i, i], 'k-')
            hold on
        end
    end
    
    for j = 0:skip:1
        for i = 0:skip:1
            
            plot3([i i], [j j], [0 1], 'k-')
            hold on
        end
    end
end

% superpixel���ꂼ��ɑ΂��Ē�������
for n = 1:segNum % �g��0�͔�΂�
    segIdx = seg == n;
    idx = segIdx & ~satMap;
    
    % edge��1��f�ł�����Δ�΂��C�܂��́C�L����f��50�ȉ��Ȃ��΂�
    if sum(sum(lapMap( segIdx ))) == 0 && sum(idx(:)) > th_pixNum
        seg_r = r( idx );
        seg_g = g( idx ); % u : [0,0.7] -> [0,200]
        seg_b = b( idx ); % v : [0,0.7] -> [0,200]
        seg_rgb = [seg_r(:), seg_g(:), seg_b(:)];

%         [line3d, inlier] = ransacfitline(seg_rgb', 5);
%         plot3(line3d(1,:), line3d(2,:), line3d(3,:));
        
        
        
        
        % --- pca
        [coeff,score,roots] = pca(seg_rgb);
        [~,p] = size(seg_rgb);
        meanX = mean(seg_rgb,1);
        
        dirVect = coeff(:,1);
        t = [min(score(:,1)), max(score(:,1))*50];
        endpts = [meanX + t(1)*dirVect'; meanX + t(2)*dirVect'];

        plot3(endpts(:,1),endpts(:,2),endpts(:,3),'o-');
        
        ax = gca;
        axis([0 1 0 1 0 1]);
        axis square, grid on, box on
        ax.BoxStyle = 'full';
        ax.XLabel.String = 'R';
        ax.YLabel.String = 'G';
        ax.ZLabel.String = 'B';
        ax.Color = [0.8, 0.8, 0.8];
        hold on
        keyboard
        
        coef = real( polyfit( seg_g(:), seg_b(:), 1) ); % polyfit( x, y, n ) ���f�����
        sError = meanSqrError( coef(1), coef(2), [seg_g(:)'; seg_b(:)']); % ������
        
        % latent:�听�����U, explained:��^��
        [~,~,latent, ~, explained] = pca([seg_g(:),seg_b(:)]);
        coefRatio = latent(1)/latent(2);
        
        allCoef( n, 1:5 ) = [ coef, sError, coefRatio, explained(1) ];
        
%         if explained(1) < 80 || coefRatio < 10
%             padding = 0.5;
%             hold on        
%             plot(seg_u, seg_v, '.');
%             seg_range = min(seg_u):0.01:max(seg_u);
%             seg_cl = coef(1)*seg_range + coef(2);
%             plot(seg_range, seg_cl);
%             hold off
%             axis([min(seg_u)-padding, max(seg_u)+padding, min(seg_v)-padding, max(seg_v)+padding]);
%             xlabel('u (Luv)','FontSize',20); ylabel('v (Luv)','FontSize',20);
%             pause, clf;
%         end
% ----------------------------------------------------------------------------------------------------------------------

        % superpixel�̃v���b�g
%         padding = 0.5;
%         hold on        
%         plot(seg_u, seg_v, '.');
%         seg_range = min(seg_u):0.01:max(seg_u);
%         seg_cl = coef(1)*seg_range + coef(2);
%         plot(seg_range, seg_cl);
% km = [seg_u(:), seg_v(:)]';
% iterNum = 300; thDist = 5; thInlrRatio = 0.1;
% [t,r] = ransac(km,iterNum,thDist,thInlrRatio);
% k1 = -tan(t);
% b1 = r/cos(t);
% plot(seg_range, seg_range*k1+b1);
% 
%         hold off
%         axis([min(seg_u)-padding, max(seg_u)+padding, min(seg_v)-padding, max(seg_v)+padding]);
%         xlabel('u (Luv)','FontSize',20); ylabel('v (Luv)','FontSize',20);
%         pause, clf;
        
        % ���E����
        %seg_uAve = (min(seg_u)+max(seg_u))/2.0; % ColorLine�̒��S(u�̕���)
        seg_uAve = mean(seg_g); % ColorLine�̒��S(�d�S)
        
        LowIdx  = find(seg_g < seg_uAve); % (u�����ϖ���)
        HighIdx = find(seg_g >= seg_uAve); % (u�����ψȏ�)
        seg_yL = mean(seg_r(LowIdx));  % colorLine�̋P�x�l(��)
        seg_yR = mean(seg_r(HighIdx)); % colorLine�̋P�x�l(�E)
        if seg_yL > seg_yR
            dir = -1; %'left';
        else
            dir = 1; %'right';
        end
       allCoef(n,6) = dir;
       allCoef(n,7) = seg_uAve;
       
%        coef = real( polyfit( seg_u(:), seg_L(:), 1) );
%        allCoef(n,6) = coef(1);
       %fprintf('poly: %f, ave: %d\n', coef(1), dir);
       
        % Luminance
%         padding = 0.05;
%         t1 = seg_u; t2 = seg_L;
%         plot(t1, t2, '.');
%         xlabel('seg_u'); ylabel('seg_L');
%         axis([min(t1)-padding, max(t1)+padding, min(t2)-padding, max(t2)+padding]);
%         xlabel('u (Luv)','FontSize',20); ylabel('L (Luv)','FontSize',20);
%         lsline;
%         pause, clf;
% ----------------------------------------------------------------------------------------------------------------------
        
    end
end


% allCoef(n:4): ��1�听���Ƒ�2�听���̊���, allCoef(n,5): ��1�听���̊�^��
invalidSegIdx = allCoef(:,3) < 0 | allCoef(:,3) > th_error | allCoef(:,4) < 10 | allCoef(:,5) < 80;
allCoef( invalidSegIdx, : ) = [];
allCoef = unique( allCoef, 'rows' );
coefNum = size(allCoef,1);
fprintf('calColorline: ')
toc
