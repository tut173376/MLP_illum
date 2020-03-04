function lupvp = rgb2lupvp( rgb )
xyz = rgb2xyz(rgb, 'WhitePoint', 'd65', 'ColorSpace', 'adobe-rgb-1998');

cform_xyz2upvpl = makecform('xyz2upvpl'); % xyz->u'v'l
tmp = applycform(xyz, cform_xyz2upvpl);

lupvp(:,:,1) = tmp(:,:,3); % L(Lu'v')
lupvp(:,:,2) = tmp(:,:,1); % u'(Lu'v')
lupvp(:,:,3) = tmp(:,:,2); % v'(Lu'v')