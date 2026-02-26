import torch 

from src.data.dataloading import EnavippH5Dataset
import src.models.eGoNavi as eGoNavi
import matplotlib.pyplot as plt


# model = eGoNavi()

ds  = EnavippH5Dataset("/home/saitama/Documents/Event_based_Navigation/h5_test/data_collect_20260219_170113.h5")

print(ds[0].keys())

fig, ax = plt.subplots(1, 2)


# ax[0].imshow(ds[10]["rgb_images"].permute(1, 2, 0).cpu().numpy())
# ax[0].set_title("RGB Left Image")
# ax[0].axis('off')
print(ds[10]["voxel"][0].shape)

ax[1].imshow(ds[20]["voxel"][0].sum(dim=0).cpu().numpy(), cmap='gray')
ax[1].set_title("Event Voxel (Summed)")
ax[1].axis('off')

print(ds[10]["action"])



plt.show()



# try:
#     # Get the first batch (since shuffle=False, these are indices 0-31)
#     batch = next(iter(train_loader))
    
#     # Load the raw dataset for comparison
#     raw_ds = EnavippH5Dataset(h5_files[0], load_rgb=False)
    
#     # Show original (720x1280) at index 20
#     idx_to_compare = 20
#     raw_voxel = raw_ds[idx_to_compare]['voxel'].sum(dim=0).cpu().numpy()
#     ax[0].imshow(raw_voxel, cmap='gray')
#     ax[0].set_title(f"Original Voxel (idx {idx_to_compare})\nShape: {raw_ds[idx_to_compare]['voxel'].shape}")
#     ax[0].axis('off')

#     # Show processed from batch (at the same index 20)
#     processed_voxel = batch['voxel'][idx_to_compare].sum(dim=0).cpu().numpy()
#     ax[1].imshow(processed_voxel, cmap='gray')
#     ax[1].set_title(f"Processed Batch Voxel (idx {idx_to_compare})\nShape: {batch['voxel'][idx_to_compare].shape}")
#     ax[1].axis('off')

#     print(f"Action sample from batch [idx {idx_to_compare}]: {batch['action'][idx_to_compare]}")
#     plt.show()

# except Exception as e:
#     print(f"\nTest failed: {e}")
#     import traceback
#     traceback.print_exc()


