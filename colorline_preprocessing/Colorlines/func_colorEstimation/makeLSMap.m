% make map %
% --
% + saturation map
% + edge(laplacian) map
% --
% "param"
% lowLv : これ以下の低画素値RGBを含む画素を黒潰れ画素とする
% highLv : これ以下の高画素値RGBを含む画素を飽和画素とする
% th_prc : 上位（100 - th_prc）％の強いエッジを含む領域を除く

function [lapMap, satMap] = makeLSMap(L, im, lowLv, highLv, th_prc)

if ~exist('lowLv','var')
    lowLv = 0.1;
end
if ~exist('highLv','var')
    highLv = 0.99;
end
satMap = sum( lowLv < im & im < highLv, 3 ) == 0;
% csatMap = sum( lowLv < im & im < highLv, 3 ) == 0;
% lsatMap = L > 0.99 | L < 0.2;
% satMap = csatMap | lsatMap;


% edge(laplacian) map : エッジは弾く
if ~exist('th_prc','var')
    th_prc = 99;
end
lapMap = abs( vl_imgrad(L) );
lapMap = lapMap > prctile(lapMap(:),th_prc);

end