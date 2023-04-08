import torch 
import numpy as np
from torch import meshgrid, arange, stack, concatenate, Tensor



def mask2bbox(mask)->Tensor:
    '''
    Takes in a mask of shape [B, C, H, W] and returns a tensor of shape [B, C, 4] 
    where each element is a bounding box of the form [x, y, w, h] in the format
    required by torchvision.ops.box_iou
    '''
    _, _, H, W = mask.shape
    M = max(H, W)
    mgrid = stack(meshgrid(arange(H), arange(W)), dim=-1)  # [H, W, 2]
    mgrid = mgrid[None, None]                              # [1, 1, H, W, 2]
    mask = mask[:, :, :, :, None]                          # [B, C, H, W, 1]

    imask = mgrid*mask                                     # [B, C, H, W, 2]
    imask = imask.flatten(2, 3)                            # [B, C, HxW,  2]
    max_ij = imask.max(2).values                           # [B, C, 2]
    imask[imask == 0] = M
    min_ij = imask.min(2).values                           # [B, C, 2]
    
    boxes = concatenate([min_ij[:, :, 1], min_ij[:, :, 0], 
                         max_ij[:, :, 1], max_ij[:, :, 0],], dim=-1)          # [B, C, 4]
    return boxes


def labels2masks(labels, background=0):
    '''Creates a mask for each label in labels'''
    masks = []
    for label in np.unique(labels):
        if label == background: continue
        masks.append(labels == label)
    return np.stack(masks, axis=0)


def get_image_pose_from_episode(episode):
    T = list(episode['objects'].values())[0]['world_T_rgb']
    return torch.as_tensor(T)

# Load data    
def collate_fn(batch):
    def img2tensor(key):
        im = np.array([v[0][key]/255.0 for v in batch])
        return torch.as_tensor(im).permute(0, 3, 1, 2).float()
    # rgb = np.array([v[0]['image']/255.0 for v in batch])
    # rgb = torch.as_tensor(rgb).permute(0, 3, 1, 2).float()
    rgb = img2tensor('image')
    nocs = img2tensor('nocs')
    targets = []
    for i, data in enumerate(batch):
        images, meta = data[0], data[1]
        depth = torch.as_tensor(images['depth'])
        masks = torch.as_tensor(labels2masks(images['semantics']))
        boxes = mask2bbox(masks[None]).type(torch.float64).reshape([-1, 4])
        labels = torch.ones(boxes.size(0)).type(torch.int64)  # NOTE: currently only one class exists
        # semantics = torch.as_tensor(images['semantics'].astype(np.int64)).unsqueeze(0)
        # semantic_ids = torch.as_tensor([v['semantic_id'] for v in meta['objects'].values()])
        targets.append({
            'depth': depth, 'masks': masks, 'nocs': nocs[i],
            'labels': labels, 'boxes': boxes, 
            # 'semantic_ids': semantic_ids,
            'camera_pose': get_image_pose_from_episode(meta),
            'intrinsics': torch.as_tensor(meta['intrinsics']),
            })

    return rgb, targets

class CollateFunctor:
    def __call__(self, batch):
        return collate_fn(batch)

from torchvision import transforms as T
class TransformImagesAndCollate:
    def __init__(self, transforms=None, device=None):
        
        self.collate = collate_fn
        
        if transforms is not None: self.transform = transforms
        else:
            self.transform = torch.nn.Sequential(
                    T.RandomHorizontalFlip(p=0.5),
                    T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                    # T.Normalize(mean=[0.485, 0.456, 0.406],  # NOTE: already done in model
                    #             std=[0.229, 0.224, 0.225],),
                )
            
        if device is not None:          self.device = device  
        elif torch.cuda.is_available(): self.device = 'cuda'
        else:                           self.device = 'cpu'

    def __call__(self, batch):
        rgb, targets = self.collate(batch)

        rgb = rgb.to(self.device)
        rgb = self.transform(rgb)
        return rgb, targets