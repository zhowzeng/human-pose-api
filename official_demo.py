from mmpose.apis import (init_pose_model, inference_bottom_up_pose_model, vis_pose_result)

config_file = 'associative_embedding_hrnet_w32_coco_512x512.py'
checkpoint_file = 'hrnet_w32_coco_512x512-bcb8c247_20200816.pth'
pose_model = init_pose_model(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'

image_name = './demo/blue_in.jpg'
# test a single image
pose_results, _ = inference_bottom_up_pose_model(pose_model, image_name)

# show the results
print(pose_results)
vis_pose_result(pose_model, image_name, pose_results, out_file='./demo/blue_out.jpg')