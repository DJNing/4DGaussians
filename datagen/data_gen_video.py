# %%
import sapien
from data_utils import *
import numpy as np
import math
from pathlib import Path as P
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from tqdm import tqdm
# scene setup

def scene_setup_v3(urdf_file, h=480, w=640, n=0.1, f=100):
    scene = sapien.Scene()
    scene.set_timestep(1 / 100.0)

    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    asset = loader.load(urdf_file)
    assert asset, "failed to load URDF."

    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
    scene.add_point_light([1, 2, 2], [1, 1, 1])
    scene.add_point_light([1, -2, 2], [1, 1, 1])
    scene.add_point_light([-1, 0, 1], [1, 1, 1])
    near, far = n, f
    width, height = w, h
    camera = scene.add_camera(
        name="camera",
        width=width,
        height=height,
        fovy=np.deg2rad(40),
        near=near,
        far=far,
    )
    return scene, camera, asset

def generate_path(phi_range, theta_range, fps, duration):
    # loop camera motions
    vertex_frames = int(fps * duration / 2)
    # phi_step = (phi_range[1] - phi_range[0]) / vertex_frames
    theta_step = (theta_range[1] - theta_range[0]) / vertex_frames
    accu_step = np.array(list(range(vertex_frames)))
    fix_step = accu_step * 0
    # phi movement
    # [phi, theta] 
    # vertix_0 = np.stack((accu_step * phi_step + phi_range[0], fix_step + theta_range[0]), axis=1)
    vertix_1 = np.stack((fix_step + phi_range[1], accu_step * theta_step + theta_range[0]), axis=1)
    # vertix_2 = np.stack((accu_step * (-phi_step) + phi_range[1], fix_step + theta_range[1]), axis=1)
    vertix_3 = np.stack((fix_step + phi_range[1], accu_step * (-theta_step) + theta_range[1]), axis=1)
    # final_motion = np.concatenate([vertix_0, vertix_1, vertix_2, vertix_3], axis=0)
    final_motion = np.concatenate([vertix_1, vertix_3], axis=0)
    return final_motion

def generate_art_motion(asset: sapien.pysapien.physx.PhysxArticulation, fps, duration, joint_id=None):
    limits = []
    for joint in asset.get_joints():
        if joint.get_dof() > 0:
            limits += [joint.get_limit()]
            
    limits = []
    for joint in asset.get_joints():
        if joint.get_dof() > 0:
            limits += [joint.get_limit()]
            
            
    # todo: add random phase shift for different parts
    limits = np.concatenate(limits, axis=0)
    total_frames = fps * duration
    step_radian = 0.03
    steps = np.arange(total_frames) * step_radian
    motion_param = np.array(list(map(np.sin, steps))) / 2 + 0.5
    motion_range = limits[:, 1] - limits[:, 0]
    motion_range = motion_range.reshape(-1, 1)*0.5
    motion_final = limits[:,0:1] + 0.3*motion_range + motion_range * motion_param
    return motion_final.T

def phi_theta_to_cam_ext(radius, phi_theta):
    
    phi = phi_theta[:, 0]
    theta = phi_theta[:, 1]
    rad = np.ones_like(phi) * radius
    points = list(map(point_in_sphere, rad, theta, phi))
    exts = list(map(calculate_cam_ext, points))
    return np.stack(exts, axis=0)

def render_img_v3(ext, camera, scene):

    camera.set_entity_pose(sapien.Pose(ext))
    scene.step()
    scene.update_render()
    camera.take_picture()
    rgba = camera.get_picture("Color")  # [H, W, 4]
    rgba_img = (rgba * 255).clip(0, 255).astype("uint8")
    seg_labels = camera.get_picture("Segmentation")  # [H, W, 4]
    mask = seg_labels.sum(axis=-1)
    mask[mask>0] = 1
    rgba_img[:, :, -1] = rgba_img[:, :, -1] * mask
    # colormap = sorted(set(ImageColor.colormap.values()))
    # color_palette = np.array([ImageColor.getrgb(color) for color in colormap],
                                # dtype=np.uint8)
    # label0_image = seg_labels[..., 0].astype(np.uint8)  # mesh-level
    label1_image = seg_labels[..., 1].astype(np.uint8)  # actor-level
    # label0_pil = Image.fromarray(color_palette[label0_image])
    # label_part_pil_vis = Image.fromarray(color_palette[label1_image])
    label_part_pil = Image.fromarray(label1_image) # label actor for part segmentation

    rgba_pil = Image.fromarray(rgba_img)
    # rgba_pil.save("color.png")
    render_dict = {
        'rgba': rgba_pil,
        'label_part': label_part_pil,
        'c2w': camera.get_model_matrix()
    }
    return render_dict

object_dict = {
    "laptop": "/afs/inf.ed.ac.uk/user/s22/s2262444/djn/10211/mobility.urdf",
    "storage": "/afs/inf.ed.ac.uk/user/s22/s2262444/djn/41083/mobility.urdf",
    "glasses": "/afs/inf.ed.ac.uk/user/s22/s2262444/djn/102599/mobility.urdf",
    "scissor": "/afs/inf.ed.ac.uk/user/s22/s2262444/djn/11100/mobility.urdf",
    "oven": "/afs/inf.ed.ac.uk/user/s22/s2262444/djn/7179/mobility.urdf",
    "blade": "/afs/inf.ed.ac.uk/user/s22/s2262444/djn/103706/mobility.urdf"
}


urdf_file = "/afs/inf.ed.ac.uk/user/s22/s2262444/djn/10211/mobility.urdf"
scene, camera, asset = scene_setup_v3(urdf_file=urdf_file)

radius = 4
fps = 30
duration = 10
phi_range = [0.3*math.pi, 0.3*math.pi]
theta_range = [0.8*math.pi, 1.8*math.pi]
camera_motion = generate_path(phi_range, theta_range, fps, duration)
art_motion = generate_art_motion(asset, fps, duration)

exts = phi_theta_to_cam_ext(radius, camera_motion)


total_frames = fps * duration
save_path = P('./test')
save_path.mkdir(exist_ok=True)
seg_path = save_path / 'seg'
rgb_path = save_path / 'rgb'

seg_path.mkdir(exist_ok=True)
rgb_path.mkdir(exist_ok=True)

meta_dict = {}

for i in tqdm(range(total_frames)):
    cur_ext = exts[i]
    cur_q_pose = art_motion[i]
    asset.set_qpos(cur_q_pose)
    render_dict = render_img_v3(cur_ext, camera, scene)
    frame_id = str(i).zfill(4)
    img_name = frame_id + '.png'
    render_dict['label_part'].save(str(seg_path / img_name))
    render_dict['rgba'].save(str(rgb_path / img_name))
    meta_dict[frame_id] = render_dict['c2w'].tolist()
    
# add intrinsic to meta dict
meta_dict['K'] = camera.get_intrinsic_matrix().tolist()

# write to save path
json_fname = save_path / 'meta.json'
with open(json_fname, 'w') as f:
    json.dump(meta_dict, f)
rgbs = [str(i) for i in list(rgb_path.glob('*.png'))]
clip = ImageSequenceClip(rgbs, fps=25)
clip.write_videofile('sapien_test/laptop.mp4', codec="libx264", fps=25, logger=None)
# clip.write_gif('test.gif')
# %%
