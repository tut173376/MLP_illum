% n: ���݌��Ă���Z�O�����gID
% seg: �Z�O�����g�}�b�v
% dir: ���邢����(-1:��, 1:�E)
% range: uv�F�x�}�̕�����*v�̃����W(�H)
% coef: uv�v���b�g�̐��`�ߎ��̌���
% seg_u: �Z�O�����g��u�l
% seg_v: �Z�O�����g��v�l
% im: ���͉摜
% bg: uv�F�x�}

function dispValidColorLine(n, seg, dir, range, coef, seg_u, seg_v, im, bg)

nseg = seg~=n; % �Ώۂł͂Ȃ��Z�O�����g(�Â�����)
tseg = seg==n; % �ΏۃZ�O�����g

tmat = repmat(tseg,[1 1 3]);
nmat = repmat(nseg,[1 1 3])*0.3;
mat = nmat + tmat;
cspixels = im.*mat; % �摜����

% ���Ɍ��摜, �E��superpixel��\��
subplot(1,2,1), imshow(cspixels);
subplot(1,2,2), imshow(double(bg)/255);

hold on
plot(seg_u, seg_v, 'wx'); % ��f�v���b�g

% �P�x�̍��������ɐL�΂�
% if dir == -1
%     seg_range = range(1):0.05:max(seg_u);
% else
%     seg_range = min(seg_u):0.05:range(2);
% end
% �P�x���𖳎�����
seg_range = range(1):0.3:range(2);

seg_cl = coef(1)*seg_range + coef(2); % �Z�O�����g��colorLine
plot(seg_range, seg_cl, 'w-');
hold off

xlabel('u (Luv)','FontSize',20); ylabel('v (Luv)','FontSize',20);
axis on
end