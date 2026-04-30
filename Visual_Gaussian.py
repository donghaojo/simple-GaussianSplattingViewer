from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, OptimizationParams
from argparse import ArgumentParser
from scene.cameras import Camera
from scene import GaussianModel
import  numpy as np
import  torch
import  cv2
import sys

def getWorld2View(R, t):
    T = np.eye(4)
    Rt = np.eye(4)
    Rt[:3, :3] = R
    T[:3, 3] = t
    R_T = Rt @ T
    C2W = np.linalg.inv(R_T)
    cam_center = C2W[:3, 3]
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def euler_angles_to_rotation_matrix(theta):
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(theta[0]), -np.sin(theta[0])],
                    [0, np.sin(theta[0]), np.cos(theta[0])]
                    ])

    R_y = np.array([[np.cos(theta[1]), 0, np.sin(theta[1])],
                    [0, 1, 0],
                    [-np.sin(theta[1]), 0, np.cos(theta[1])]
                    ])

    R_z = np.array([[np.cos(theta[2]), -np.sin(theta[2]), 0],
                    [np.sin(theta[2]), np.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])

    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

def _refresh_camera(Cam):
    """Recompute derived camera matrices after R or T changes."""
    Cam.world_view_transform = torch.tensor(getWorld2View(Cam.R, Cam.T)).transpose(0, 1).cuda()
    Cam.full_proj_transform = (
        Cam.world_view_transform.unsqueeze(0).bmm(Cam.projection_matrix.unsqueeze(0))
    ).squeeze(0)
    Cam.camera_center = Cam.world_view_transform.inverse()[3, :3]

def update_camera_position(Cam, delta_x, delta_y, delta_z):
    # delta is expressed in camera local space; transform to world space via R^T (= R^{-1}).
    # Cam.T = -camera_world_pos, so moving the camera forward means subtracting the world delta.
    delta_cam = np.array([delta_x, delta_y, delta_z])
    delta_world = Cam.R.T @ delta_cam
    Cam.T -= delta_world
    _refresh_camera(Cam)

def update_camera_rotation(Cam, camera_theta, delta_theta_x, delta_theta_y, delta_theta_z):
    """更新相机旋转"""
    camera_theta[0] -= delta_theta_x
    camera_theta[1] -= delta_theta_y
    camera_theta[2] -= delta_theta_z
    Cam.R = euler_angles_to_rotation_matrix(camera_theta)
    _refresh_camera(Cam)

def create_camera(camera_init_position, camera_init_theta):
    camera_position = np.array(camera_init_position).astype(float)
    theta = np.array(camera_init_theta).astype(float)
    R = euler_angles_to_rotation_matrix(theta)
    T = -camera_position
    Cam = Camera(colmap_id=0, R=R, T=T,
                 FoVx=1, FoVy=1, image=torch.tensor(torch.zeros((3, 1024, 1024)), dtype=torch.float32), gt_alpha_mask=None,
                 image_name=None, data_device=0, uid=id)
    _refresh_camera(Cam)
    return Cam


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    # parser.add_argument('--ip', type=str, default="127.0.0.1")
    # parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[6000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--near", type=int, default=0)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    #The initialization of camera coordinates and viewing angle parameters
    camera_position = [0, 0, 0]
    camera_theta = [0, 0, 0]
    checkpoint = r".\apples\point_cloud\iteration_30000\point_cloud.ply" #ply file address
    speed_move = 1              # Camera translation speed
    speed_rotation = 0.05       # Camera rotation speed (keyboard)
    mouse_rotation_speed = 0.005  # Camera rotation speed (mouse drag)
    mouse_zoom_speed = 0.05     # Gaussian scale change per scroll tick
    min_gaussian_scale = 0.01   # Minimum allowed s_mod value
    s_mod = 1                   # Gaussian size scaling

    Cam = create_camera(camera_position, camera_theta)

    gaussians = GaussianModel(3)
    #scene = Scene(lp.extract(args), gaussians)
    gaussians.training_setup(op.extract(args))
    gaussians.load_ply(checkpoint)
    gaussians.training_setup(op)
    gaussians._features_dc.requires_grad_(False)
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    # Mouse state for drag-to-rotate and scroll-to-zoom
    mouse_state = {'dragging': False, 'last_x': 0, 'last_y': 0}

    def mouse_callback(event, x, y, flags, param):
        nonlocal s_mod
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse_state['dragging'] = True
            mouse_state['last_x'] = x
            mouse_state['last_y'] = y
        elif event == cv2.EVENT_LBUTTONUP:
            mouse_state['dragging'] = False
        elif event == cv2.EVENT_MOUSEMOVE and mouse_state['dragging']:
            dx = x - mouse_state['last_x']
            dy = y - mouse_state['last_y']
            mouse_state['last_x'] = x
            mouse_state['last_y'] = y
            # dy → pitch (theta_x),  dx → yaw (theta_y)
            update_camera_rotation(Cam, camera_theta,
                                   dy * mouse_rotation_speed,
                                   dx * mouse_rotation_speed, 0)
        elif event == cv2.EVENT_MOUSEWHEEL:
            s_mod = max(min_gaussian_scale, s_mod + (mouse_zoom_speed if flags > 0 else -mouse_zoom_speed))

    cv2.namedWindow('Rendered Video Stream')
    cv2.setMouseCallback('Rendered Video Stream', mouse_callback)

    while True:
        image = render(Cam, gaussians, pp, background, scaling_modifier=s_mod)['render']
        image = image.permute(1, 2, 0).clamp(0, 1).detach().cpu().numpy()
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow('Rendered Video Stream', image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            break
        # WASD + QE: deltas are in camera-local space (fixed sign relative to original)
        elif key == ord('a'):
            update_camera_position(Cam, -speed_move, 0, 0)   # strafe left  (−x)
        elif key == ord('d'):
            update_camera_position(Cam,  speed_move, 0, 0)   # strafe right (+x)
        elif key == ord('w'):
            update_camera_position(Cam, 0, 0, -speed_move)   # move forward (−z)
        elif key == ord('s'):
            update_camera_position(Cam, 0, 0,  speed_move)   # move back    (+z)
        elif key == ord('q'):
            update_camera_position(Cam, 0, -speed_move, 0)   # move up      (−y)
        elif key == ord('e'):
            update_camera_position(Cam, 0,  speed_move, 0)   # move down    (+y)
        elif key == ord('i'):
            update_camera_rotation(Cam, camera_theta,  speed_rotation, 0, 0)
        elif key == ord('k'):
            update_camera_rotation(Cam, camera_theta, -speed_rotation, 0, 0)
        elif key == ord('j'):
            update_camera_rotation(Cam, camera_theta, 0, -speed_rotation, 0)
        elif key == ord('l'):
            update_camera_rotation(Cam, camera_theta, 0,  speed_rotation, 0)
        elif key == ord('u'):
            update_camera_rotation(Cam, camera_theta, 0, 0, -speed_rotation)
        elif key == ord('o'):
            update_camera_rotation(Cam, camera_theta, 0, 0,  speed_rotation)
        elif key == ord('='):
            s_mod += 0.01
        elif key == ord('-'):
            s_mod = max(min_gaussian_scale, s_mod - 0.01)
        elif key == ord('9'):
            cv2.imwrite("image.jpg", image * 255)

    cv2.destroyAllWindows()
