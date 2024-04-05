ref_image='your_image_path'
load(ref_image)

% coordinate sof reference shape
ref_top_left=[426,399]
ref_bottom_left=[426,466]
ref_bottom_right=[485,466]
ref_top_right=[485,399]

x_cm=8.69
y_cm=5.43
width_cm=3.91
height_cm=4.04
x_pixels=tl(1)
y_pixels=tl(2)
width_pixels=abs(tl(1)-tr(1))
height_pixels=abs(tl(2)-bl(2))

% coordinate so newly drawn shappe, in this case a square of same size as
% reference but in different location
tl_new=[586,349]
bl_new=[586,349+height_pixels]
br_new=[586+width_pixels,349+height_pixels]
tr_new=[586+width_pixels,349]

% plot the images
imagesc(snap)
hold on
plot(ref_top_left(1),ref_top_left(2),'marker','x','MarkerSize',5,'Color','white')
plotref_bottom_leftbl(1),ref_bottom_left(2),'marker','x','MarkerSize',5,'Color','white')
plot(ref_bottom_right(1),ref_bottom_right(2),'marker','x','MarkerSize',5,'Color','white')
plot(ref_top_right(1),ref_top_right(2),'marker','x','MarkerSize',5,'Color','white')
plot(tl_new(1),tl_new(2),'marker','x','MarkerSize',5,'Color','white')
plot(bl_new(1),bl_new(2),'marker','x','MarkerSize',5,'Color','white')
plot(br_new(1),br_new(2),'marker','x','MarkerSize',5,'Color','white')
plot(tr_new(1),tr_new(2),'marker','x','MarkerSize',5,'Color','white')



% consider the reolution
pixel_resolution_x = 1280/33.867; % example pixel resolution in x-direction (pixels per cm)
pixel_resolution_y = 720/19.05; % example pixel resolution in y-direction (pixels per cm)

% Define the transformation matrix
scale_x = width_cm * pixel_resolution_x / width_pixels;
scale_y = height_cm * pixel_resolution_y / height_pixels;
translation_x = (x_cm * pixel_resolution_x) - x_pixels * scale_x;
translation_y = (y_cm * pixel_resolution_y) - y_pixels * scale_y;
T = [scale_x 0 0; 0 scale_y 0; translation_x translation_y 1];

tform = affine2d(T);


% Transform the coordinates of the two points to centimeter coordinates

% reference point to double chek the tranform works as expected
point1_cm = transformPointsForward(tform, tl);
point2_cm = transformPointsForward(tform, bl);
point3_cm = transformPointsForward(tform, br);
point4_cm = transformPointsForward(tform, tr);
disp(['Point Top Left coordinates in cm image: ', num2str(point1_cm./[pixel_resolution_x,pixel_resolution_y])]);
disp(['Point Bottom Left coordinates in cm image: ', num2str(point2_cm./[pixel_resolution_x,pixel_resolution_y])]);
disp(['Point Bottom Right coordinates in cm image: ', num2str(point3_cm./[pixel_resolution_x,pixel_resolution_y])]);
disp(['Point Top Right coordinates in cm image: ', num2str(point4_cm./[pixel_resolution_x,pixel_resolution_y])]);


% new shape powerpoint coordinates
tl_new_cm = transformPointsForward(tform, tl_new);
bl_new_cm = transformPointsForward(tform, bl_new);
br_new_cm = transformPointsForward(tform, br_new);
tr_new_cm = transformPointsForward(tform, tr_new);

% use tl_new_cm coordintaes for powerpoint and width and height. When
% doping it porperly we will always work on pixels with psychtoolbox
width_cm=abs(tl_new_cm(1)./pixel_resolution_x-tr_new_cm(1)./pixel_resolution_x)
height_cm=abs(tl_new_cm(2)./pixel_resolution_y-bl_new_cm(2)./pixel_resolution_y)

% Display the coordinates of the two points in centimeter image
disp(['Point Top Left coordinates in cm image: ', num2str(tl_new_cm./[pixel_resolution_x,pixel_resolution_y])]);
disp(['Point Bottom Left coordinates in cm image: ', num2str(bl_new_cm./[pixel_resolution_x,pixel_resolution_y])]);
disp(['Point Bottom Right coordinates in cm image: ', num2str(br_new_cm./[pixel_resolution_x,pixel_resolution_y])]);
disp(['Point Top Right coordinates in cm image: ', num2str(tr_new_cm./[pixel_resolution_x,pixel_resolution_y])]);
