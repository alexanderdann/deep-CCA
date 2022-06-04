import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch

class EarlyStoppingCallback:
    def check_last_epochs(self, loss_list, idx, condition):
        losses = torch.tensor(loss_list)
        
        if (torch.std(losses[-idx:]) < condition) and (idx < len(losses)):
            return True
        
        else:
            return False
        

def visualize_brain_maps(brain_map_1, brain_map_2, title='Brain Maps', tensorboard=False):
    assert brain_map_1.shape[1] == brain_map_2.shape[1]
    _, samples = brain_map_1.shape
    fmri_map_size = int(np.sqrt(samples))
    
    fig, ax = plt.subplots(1, 2, figsize=(15, 30))
    images = list()
    for idx, brain_map in enumerate([brain_map_1, brain_map_2]):
        fmri = np.zeros(shape=(fmri_map_size, fmri_map_size))
        
        for bm_idx, raw_fmri in enumerate(brain_map):
            fmri += np.reshape(raw_fmri, (fmri_map_size, fmri_map_size))
        
        cv2.circle(fmri, (fmri_map_size//2, fmri_map_size//2), fmri_map_size//2, np.amax(fmri))
        ax[idx].imshow(fmri, cmap='CMRmap')
        ax[idx].get_xaxis().set_visible(False)
        ax[idx].get_yaxis().set_visible(False)
        ax[idx].set_title(f'{title} for view {idx+1}', fontsize=15)
        
        images.append(fmri)
    
    if tensorboard:
        plt.close()
        return images
    
    else:
        plt.show()
    
    