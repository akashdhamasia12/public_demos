import config_devcloud as config
import numpy as np
import matplotlib.pyplot as plt

sequence_length = config.SEQ_LENGTH
past_trajectory = config.HISTORY
history_frames = past_trajectory*2 + 3
total_maneuvers = ["none", "straight", "right", "left"]


def trajectories_plot(image, outputs, targets, j, image_agent_all, ade, fde, history_traj, plot_figures):
    """
    This function plots the regressed (predicted) keypoints and the actual 
    keypoints for each test image in the batch.
    """

    fig, (ax2, ax3) = plt.subplots(1, 2, figsize=(14,7))

    fig.suptitle("Seq {}".format(j))

    img_map = np.copy(image[0, history_frames-3:history_frames, :, :])
    img_map = np.transpose(img_map, (1, 2, 0))
    img_map1 = np.copy(img_map)

    img_agent = np.copy(image[0, 0, :, :])
    img_agent = img_agent > 0.2
    img_map[img_agent] = [0.0, 0.0, 1.0] #(RGB) #blue for agents

    image_agent_all = image_agent_all > 40
    img_map1[image_agent_all] = [0.0, 0.0, 1.0] #(RGB) #blue for agents

    img_ego = np.copy(image[0, past_trajectory, :, :])
    img_ego = img_ego > 0.2
    img_map[img_ego] = [0.0, 1.0, 0.0] #(RGB) #green for ego
    img_map1[img_ego] = [0.0, 1.0, 0.0] #(RGB) #green for ego

    ax2.imshow(img_map1)
    
    ax3.set(xlim=(0, config.IMAGE_SIZE), ylim=(config.IMAGE_SIZE, 0))

    t_= "ADE = " + str(round(ade,3)) + "\nFDE = " + str(round(fde,3))  
    ax3.text(10,10,t_)

    ax2.plot(targets[:,0], targets[:,1], color='green', marker='o', linewidth=0.1, markersize=1.5, label='Ground-Truth')
    ax2.plot(outputs[:,0], outputs[:,1], color='red', marker='x', linewidth=0.1, markersize=1.5, label='Predictions')
    ax2.plot(history_traj[:,0], history_traj[:,1], color='cyan', marker='+', linewidth=0.1, markersize=1.5, label='history')

    ax3.plot(targets[:,0], targets[:,1], color='green', marker='o', markersize=1, linewidth=0.3, label='Ground-Truth')
    ax3.plot(outputs[:,0], outputs[:,1], color='red', marker='+', markersize=1, linewidth=0.3, label='Predictions')
    ax3.plot(history_traj[:,0], history_traj[:,1], color='cyan', marker='+', markersize=1, label='history')


    fig.savefig(f"{config.plots}/seq_" + str(j) + ".png")

    if plot_figures:
        plt.show()

    # plt.close()
    plt.close('all')
