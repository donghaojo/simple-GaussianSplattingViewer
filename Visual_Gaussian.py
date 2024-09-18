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

def update_camera_position(Cam,delta_x, delta_y, delta_z):
    Cam.T[0] += delta_x
    Cam.T[1] += delta_y
    Cam.T[2] += delta_z
    Cam.world_view_transform = torch.tensor(getWorld2View(Cam.R, Cam.T)).transpose(0,1).cuda()
    Cam.full_proj_transform = (
        Cam.world_view_transform.unsqueeze(0).bmm(Cam.projection_matrix.unsqueeze(0))
    ).squeeze(0)
    Cam.camera_center = Cam.world_view_transform.inverse()[3, :3]

def update_camera_rotation(Cam,delta_theta_x, delta_theta_y,delta_theta_z):
    """ 更新相机旋转 """
    camera_theta[0] -= delta_theta_x
    camera_theta[1] -= delta_theta_y
    camera_theta[2] -= delta_theta_z
    Cam.R =euler_angles_to_rotation_matrix(camera_theta)
    Cam.world_view_transform = torch.tensor(getWorld2View(Cam.R, Cam.T)).transpose(0, 1).cuda()
    Cam.full_proj_transform = (
        Cam.world_view_transform.unsqueeze(0).bmm(Cam.projection_matrix.unsqueeze(0))
    ).squeeze(0)
    Cam.camera_center = Cam.world_view_transform.inverse()[3, :3]

def create_camera(camera_init_position,camera_init_theta):
    camera_position = np.array(camera_init_position).astype(float)
    theta = np.array(camera_init_theta).astype(float)
    R = euler_angles_to_rotation_matrix(theta)
    T = -camera_position
    Cam = Camera(colmap_id=0, R=R, T=T,
                 FoVx=1, FoVy=1, image=torch.tensor(torch.zeros((3, 1024, 1024)), dtype=torch.float32), gt_alpha_mask=None,
                 image_name=None, data_device=0, uid=id)
    update_camera_position(Cam, 0, 0, 0)
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
    camera_position = [0,0,0]
    camera_theta = [0,0,0]
    checkpoint = r".\output\apples\point_cloud\iteration_30000\point_cloud.ply" #ply file address
    speed_move = 5  #Camera speed
    s_mod = 0.01 #Gaussian size scaling

    Cam = create_camera(camera_position,camera_theta)

    gaussians = GaussianModel(3)
    #scene = Scene(lp.extract(args), gaussians)
    gaussians.training_setup(op.extract(args))
    gaussians.load_ply(checkpoint)
    gaussians.training_setup(op)
    gaussians._features_dc.requires_grad_(False)
    background = torch.tensor([0,0,0], dtype=torch.float32, device="cuda")

    while True:
        image = render(Cam, gaussians, pp, background, scaling_modifier=s_mod)['render']
        image = image.permute(1, 2, 0).clamp(0, 1).detach().cpu().numpy()
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow('Rendered Video Stream', image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            break
        elif key == ord('a'):
            update_camera_position(Cam,1, 0, 0)
        elif key == ord('d'):
            update_camera_position(Cam,-1, 0, 0)
        elif key == ord('w'):
            update_camera_position(Cam,0, 0, -1)
        elif key == ord('s'):
            update_camera_position(Cam,0, 0, 1)
        elif key == ord('q'):
            update_camera_position(Cam,0, 1, 0)
        elif key == ord('e'):
            update_camera_position(Cam,0, -1, 0)
        elif key == ord('i'):
            update_camera_rotation(Cam,0.05, 0,0)
        elif key == ord('k'):
            update_camera_rotation(Cam,-0.05, 0,0)
        elif key == ord('j'):
            update_camera_rotation(Cam,0, -0.05,0)
        elif key == ord('l'):
            update_camera_rotation(Cam,0, 0.05,0)
        elif key == ord('u'):
            update_camera_rotation(Cam,0,0, -0.05)
        elif key == ord('o'):
            update_camera_rotation(Cam,0, 0,0.05)
        elif key == ord('='):
            s_mod += 0.01
        elif key == ord('-'):
            s_mod -= 0.01
        elif key == ord('9'):
            cv2.imwrite("image.jpg", image*255)


    cv2.destroyAllWindows()
