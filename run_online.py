import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
import cv2

from hardware.device import get_device
from inference.post_process import post_process_output
from utils.data.camera_data import CameraData
from utils.visualisation.plot import plot_results, save_results

logging.basicConfig(level=logging.INFO)


cap = cv2.VideoCapture(1)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate network')
    parser.add_argument('--network', type=str,
                        help='Path to saved network to evaluate')
    parser.add_argument('--save', type=int, default=0,
                        help='Save the results')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # Get the compute device
    device = get_device(force_cpu=True)

    # Load Network
    logging.info('Loading model...')
    net = torch.load(args.network, map_location=device)
    logging.info('Done')

    while True:
        ret, frame = cap.read()
        pic = cv2.flip(frame, 1)
        cv2.imshow('video', pic)

        rgb = np.array(pic)

        img_data = CameraData(include_depth=False, include_rgb=True)

        x, depth_img, rgb_img = img_data.get_data(rgb=rgb)

        with torch.no_grad():
            xc = x.to(device)
            pred = net.predict(xc)

            q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])

            fig = plt.figure(figsize=(10, 10))
            cv2.imshow('jaja', img_data.get_rgb(rgb, False))
            # plot_results(fig=fig,
            #              rgb_img=img_data.get_rgb(rgb, False),
            #              grasp_q_img=q_img,
            #              grasp_angle_img=ang_img,
            #              no_grasps=1,
            #              grasp_width_img=width_img)

        if cv2.waitKey(1) == ord('z'):
            break

        # if args.save:
        #     save_results(
        #         rgb_img=img_data.get_rgb(rgb, False),
        #         grasp_q_img=q_img,
        #         grasp_angle_img=ang_img,
        #         no_grasps=1,
        #         grasp_width_img=width_img
        #     )
        # else:
        #     fig = plt.figure(figsize=(10, 10))
        #     plot_results(fig=fig,
        #                  rgb_img=img_data.get_rgb(rgb, False),
        #                  grasp_q_img=q_img,
        #                  grasp_angle_img=ang_img,
        #                  no_grasps=1,
        #                  grasp_width_img=width_img)
        #     fig.savefig('img_result.pdf')
