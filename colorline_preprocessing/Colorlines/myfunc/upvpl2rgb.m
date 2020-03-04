function rgb = upvpl2rgb( upvpl )

cform_upvpl2xyz = makecform('upvpl2xyz'); % u'v'l -> xyz
xyz = applycform(upvpl, cform_upvpl2xyz);

rgb = xyz2rgb(xyz, 'WhitePoint', 'd65', 'ColorSpace', 'adobe-rgb-1998');