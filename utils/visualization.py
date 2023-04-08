import numpy as np 
import cv2 
import matplotlib.pyplot as plt


# Credit: https://gist.github.com/woolpeeker/d7e1821e1b5c556b32aafe10b7a1b7e8
# drawMatches numpy version
def draw_matches(img1, kp1, img2, kp2, matches, color=None): 
    """Draws lines between matching keypoints of two images.  
    Keypoints not in a matching pair are not drawn.
    Places the images side by side in a new image and draws circles 
    around each keypoint, with line segments connecting matching pairs.
    You can tweak the r, thickness, and figsize values as needed.
    Args:
        img1: An openCV image ndarray in a grayscale or color format.
        kp1: ndarray [n1, 2]
        img2: An openCV image ndarray of the same format and with the same 
        element type as img1.
        kp2: ndarray [n2, 2]
        matches: ndarray [n_match, 2]
        img1 keypoints and whose queryIdx attribute refers to img2 keypoints.
        color: The color of the circles and connecting lines drawn on the images.  
        A 3-tuple for color images, a scalar for grayscale images.  If None, these
        values are randomly generated.  
    """
    # We're drawing them side by side.  Get dimensions accordingly.
    # Handle both color and grayscale images.
    if len(img1.shape) == 3:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], img1.shape[2])
    elif len(img1.shape) == 2:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1])
    new_img = np.zeros(new_shape, type(img1.flat[0]))  
    # Place images onto the new image.
    new_img[0:img1.shape[0],0:img1.shape[1]] = img1
    new_img[0:img2.shape[0],img1.shape[1]:img1.shape[1]+img2.shape[1]] = img2
    
    # Draw lines between matches.  Make sure to offset kp coords in second image appropriately.
    r = 15
    thickness = 2
    if color:
        c = color
    for m in matches:
        # Generate random color for RGB/BGR and grayscale images as needed.
        if not color: 
            c = np.random.randint(0,256,3) if len(img1.shape) == 3 else np.random.randint(0,256)
        if callable(color):  c = color()
        # So the keypoint locs are stored as a tuple of floats.  cv2.line(), like most other things,
        # wants locs as a tuple of ints.
        end1 = tuple(np.round(kp1[m[0]]).astype(int))
        end2 = tuple(np.round(kp2[m[1]]).astype(int) + np.array([img1.shape[1], 0]))
        cv2.line(new_img, end1, end2, c, thickness)
        cv2.circle(new_img, end1, r, c, thickness)
        cv2.circle(new_img, end2, r, c, thickness)
    return new_img



def get_image_with_matches(imgA, kpA, imgB, kpB, matches=None):
    kpA, kpB = kpA[-1:-3:-1].T, kpB[-1:-3:-1].T
    if matches is None: matches = [(i, i) for i in range(len(kpA))]
    matched_image = draw_matches(imgA, kpA, imgB, kpB, matches, ColorGenerator(3))
    return cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB)



class ColorGenerator:
    def __init__(self, step):
        self.step = step
        self._step_size = int(255 / (step-1))
        self.count = [0, 0, 0]
    def __call__(self):
        self.count[0] = (self.count[0] + 1) % self.step
        if self.count[0] == (self.step-1):
            self.count[1] = (self.count[1] + 1) % self.step
        if self.count[1] == (self.step-1):
            self.count[2] = (self.count[2] + 1) % self.step
        return [self.count[i] * self._step_size for i in range(3)]


def display_image_with_keypoints(ij_set, image, name, save_dir=None):
    image = np.copy(image)
    get_color = ColorGenerator(3)
    for ij in ij_set.T.astype(int):
        cv2.circle(image , (ij[1], ij[0]), 5, get_color(), 1)
    if save_dir: cv2.imwrite(f'{save_dir}/{name}.png', image)
    cv2.imshow(name, image)
    cv2.waitKey(20)


def get_image_with_keypoints(ij_set, image):
    image = np.copy(image)
    get_color = ColorGenerator(3)
    for ij in ij_set.T.astype(int):
        cv2.circle(image , (ij[1], ij[0]), 5, get_color(), 1)
    return image


from utils.geometric.se3_utils import Se3PixelHandler

class FrameVisualizer:
    def __init__(self, ij_samples, depth_samples, intrinsic, thickness=4, length=0.15):
        self.K = intrinsic

        self.thickness = thickness  # 2
        axis_length = length  #0.075
        
        # Create vectors representing frame
        I = np.eye(4)*axis_length; I[:, 3] = 1
        self.axis3d = (I[i] for i in range(4)) # x, y, z, o

        # Push frame 
        origin = self._get_3d_origin(ij_samples, depth_samples)
        T_init = np.eye(4); T_init[0:3, 3] = origin
        
        # Position axis at start
        self.axis3d = [T_init @ v for v in self.axis3d]

    def _get_3d_origin(self, ijs, ds):
        handler = Se3PixelHandler(ijs, ds, self.K)
        xyz = handler.get_3d()
        return xyz.mean(dim=1)
    
    def get_frame_3d(self, se3=np.eye(4)):
        # Returns [x, y, z, origin] 3d points.
        return [(se3@v)[0:3] for v in self.axis3d]

    def __call__(self, image, se3=np.eye(4)):
        '''Get image with overlayed transformed frame'''
        im = np.copy(image)

        # transform axis
        T_axis = self.get_frame_3d(se3)

        # project on image
        frame = [v/v[2] for v in T_axis]
        projected = [self.K @ v for v in frame]
        x, y, z, o = [p[0:2].astype(int) for p in projected]

        # Draw
        r, g, b = (255, 0, 0), (0, 255, 0), (0, 0, 255) 
        for end, color in zip([x, y, z], [r, g, b]):
            cv2.line(im, o, end, color, thickness=self.thickness)

        return im

    
def draw_frame(ijs, depth, intrinsic, drawing_canvas, se3=np.eye(4)):
    '''
    - centers camera frame at ijs
    - if se3 given, transforms the frame
    - draws frame on image and returns
    ijs and depth need be correlated, drawing_canvas can be any image
    '''
    im = np.copy(drawing_canvas)
    # Get 3d center of ijs
    sample_depths = depth[ijs[0], ijs[1]]   
    handler = Se3PixelHandler(ijs, sample_depths, intrinsic)
    xyz = handler.get_3d()
    origin = xyz.mean(dim=1)
    
    # Make frame transform matrix
    frame = np.eye(4)
    frame[0:3, 3] = origin
    frame = se3@frame
    
    # Create camera frame axis
    axis_length = 0.075
    I = np.eye(4)*axis_length; I[:, 3] = 1
    axis = (I[i] for i in range(4)) # x, y, z, o

    # transform axis
    T_v = ((frame@v)[0:3] for v in axis)

    # project on image
    T_v = (v/v[2] for v in T_v)
    T_p = (intrinsic @ v for v in T_v)
    x, y, z, o = (p[0:2].astype(int) for p in T_p)

    # Draw
    r, g, b = (255, 0, 0), (0, 255, 0), (0, 0, 255) 
    for end, color in zip([x, y, z], [r, g, b]):
        cv2.line(im, o, end, color, thickness=2)

    return im




from PIL import Image
def make_gif(im_file_list, out_file_name):
    # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
    imgs = (Image.open(f) for f in im_file_list)
    img = next(imgs)  # extract first image from iterator
    img.save(fp=out_file_name, format='GIF', append_images=imgs,
            save_all=True, duration=1500, loop=0)