% make map %
% --
% + saturation map
% + edge(laplacian) map
% --
% "param"
% lowLv : ����ȉ��̒��f�lRGB���܂މ�f�����ׂ��f�Ƃ���
% highLv : ����ȉ��̍���f�lRGB���܂މ�f��O�a��f�Ƃ���
% th_prc : ��ʁi100 - th_prc�j���̋����G�b�W���܂ޗ̈������

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


% edge(laplacian) map : �G�b�W�͒e��
if ~exist('th_prc','var')
    th_prc = 99;
end
lapMap = abs( vl_imgrad(L) );
lapMap = lapMap > prctile(lapMap(:),th_prc);

end