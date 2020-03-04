clear;

scale = 10000;

vrange = [ 0, 0.7, 0, 0.7 ];
range = vrange * scale;

uu = vrange(1) : 1/scale : vrange(2);
vv = vrange(3) : 1/scale : vrange(4);

uLeng = length(uu); vLeng = length(vv);
bg_u = repmat(uu,[vLeng,1]); bg_v = repmat(vv',[1,uLeng]);

bg_x = 9*bg_u ./ ( 6*bg_u - 16*bg_v + 12 );
bg_y = 4*bg_v ./ ( 6*bg_u - 16*bg_v + 12 );
bg_z = 1 - bg_x - bg_y;

bg_rgb = xyz2rgb( cat(3,bg_x,bg_y,bg_z), 'WhitePoint', 'd65', 'ColorSpace', 'adobe-rgb-1998' );
outerMap = bg_rgb<0.0 | bg_rgb>1.0;

outerMap = outerMap(:,:,1) | outerMap(:,:,2) | outerMap(:,:,3);
bg_rgb( repmat(outerMap,[1 1 3]) ) = 0.5; % set gray

bg = round( 255*bg_rgb );

save('outerMap(scale=10000).mat', 'outerMap');
