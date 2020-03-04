function rgb = lupvp2rgb( lupvp )
tmp(:,:,3) = lupvp(:,:,1); % L(Lu'v')
tmp(:,:,1) = lupvp(:,:,2); % u'(Lu'v')
tmp(:,:,2) = lupvp(:,:,3); % v'(Lu'v')

cform_upvpl2xyz = makecform('upvpl2xyz'); % u'v'l -> xyz
xyz = applycform(tmp, cform_upvpl2xyz);

rgb = xyz2rgb(xyz, 'WhitePoint', 'd65', 'ColorSpace', 'adobe-rgb-1998');