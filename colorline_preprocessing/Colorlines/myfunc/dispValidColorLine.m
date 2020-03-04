% n: 現在見ているセグメントID
% seg: セグメントマップ
% dir: 明るい方向(-1:左, 1:右)
% range: uv色度図の分割数*vのレンジ(？)
% coef: uvプロットの線形近似の結果
% seg_u: セグメントのu値
% seg_v: セグメントのv値
% im: 入力画像
% bg: uv色度図

function dispValidColorLine(n, seg, dir, range, coef, seg_u, seg_v, im, bg)

nseg = seg~=n; % 対象ではないセグメント(暗くする)
tseg = seg==n; % 対象セグメント

tmat = repmat(tseg,[1 1 3]);
nmat = repmat(nseg,[1 1 3])*0.3;
mat = nmat + tmat;
cspixels = im.*mat; % 画像生成

% 左に元画像, 右にsuperpixelを表示
subplot(1,2,1), imshow(cspixels);
subplot(1,2,2), imshow(double(bg)/255);

hold on
plot(seg_u, seg_v, 'wx'); % 画素プロット

% 輝度の高い方向に伸ばす
% if dir == -1
%     seg_range = range(1):0.05:max(seg_u);
% else
%     seg_range = min(seg_u):0.05:range(2);
% end
% 輝度情報を無視する
seg_range = range(1):0.3:range(2);

seg_cl = coef(1)*seg_range + coef(2); % セグメントのcolorLine
plot(seg_range, seg_cl, 'w-');
hold off

xlabel('u (Luv)','FontSize',20); ylabel('v (Luv)','FontSize',20);
axis on
end