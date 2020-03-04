function upvpl = rgb2upvpl( rgb )
xyz2upvpl = makecform('xyz2upvpl'); % xyz->u'v'l converter
xyz = rgb2xyz(rgb, 'WhitePoint', 'd65', 'ColorSpace', 'adobe-rgb-1998');
upvpl = applycform(xyz, xyz2upvpl);