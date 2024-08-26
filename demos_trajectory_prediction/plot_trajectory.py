import config
import numpy as np
import matplotlib.pyplot as plt

sequence_length = config.SEQ_LENGTH
past_trajectory = config.HISTORY
history_frames = past_trajectory*2 + 3
total_maneuvers = ["none", "straight", "right", "left"]


def trajectories_plot(image, outputs, targets, j, image_agent_all, centroid_, ade, fde, history_traj, maneuver_label, maneuver_pred, output_list, outputs_var_mean, final_index, noise, plot_figures):
    """
    This function plots the regressed (predicted) keypoints and the actual 
    keypoints for each test image in the batch.
    """

    # fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # fig1, (ax3, ax4) = plt.subplots(1, 2)
    fig, (ax2, ax3) = plt.subplots(1, 2, figsize=(14,7))
    # fig1, (ax3, ax4) = plt.subplots(1, 2)

    fig.suptitle("Seq {}".format(j))
    # fig1.suptitle("Seq {}".format(j))

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

    if config.NEIGHBOUR_RADIUS > 0:
        circle2 = plt.Circle((int(centroid_[0]), int(centroid_[1])), int(config.NEIGHBOUR_RADIUS), color='b', fill=False)
        ax2.add_patch(circle2)

    # ax1.imshow(img_map)
    ax2.imshow(img_map1)
    
    # outputs = outputs.reshape(-1,2)
    # targets = targets.reshape(-1,2)
    # history_traj = history_traj.reshape(-1,2)

    ax3.set(xlim=(0, config.IMAGE_SIZE), ylim=(config.IMAGE_SIZE, 0))
    # ax4.set(xlim=(0, config.IMAGE_SIZE), ylim=(config.IMAGE_SIZE, 0))

    # ax5.set(xlim=(0, config.IMAGE_SIZE), ylim=(config.IMAGE_SIZE, 0))
    # ax6.set(xlim=(0, config.IMAGE_SIZE), ylim=(config.IMAGE_SIZE, 0))

    t_= "ADE = " + str(round(ade,3)) + "\nFDE = " + str(round(fde,3)) + "\nManeuver_gt = " + total_maneuvers[maneuver_label] + "\nManeuver_pred = " + total_maneuvers[maneuver_pred] + "\nVariance = " + str(np.mean(outputs_var_mean)) 
    ax3.text(10,10,t_)

    # ax1.plot(targets[:,0], targets[:,1], color='green', marker='o', linewidth=0.1, markersize=1, label='Ground-Truth')
    # ax1.plot(outputs[:,0], outputs[:,1], color='red', marker='x', linewidth=0.1, markersize=1, label='Predictions')
    # ax1.plot(history_traj[:,0], history_traj[:,1], color='cyan', marker='+', linewidth=0.1, markersize=1, label='history')

    ax2.plot(targets[:,0], targets[:,1], color='green', marker='o', linewidth=0.1, markersize=1.5, label='Ground-Truth')
    ax2.plot(outputs[:,0], outputs[:,1], color='red', marker='x', linewidth=0.1, markersize=1.5, label='Predictions')
    ax2.plot(history_traj[:,0], history_traj[:,1], color='cyan', marker='+', linewidth=0.1, markersize=1.5, label='history')

    ax3.plot(targets[:,0], targets[:,1], color='green', marker='o', markersize=1, linewidth=0.3, label='Ground-Truth')
    # ax3.plot(outputs[:,0], outputs[:,1], color='red', marker='x', markersize=1, label='Predictions')
    ax3.errorbar(outputs[:,0], outputs[:,1], outputs_var_mean, color='red', linestyle='None', marker='+', markersize=1, linewidth=0.4, label='Predictions') #capsize=0.2
    ax3.plot(history_traj[:,0], history_traj[:,1], color='cyan', marker='+', markersize=1, label='history')

    # if final_index != -1:
    #     closest_output = output_list[final_index]
    #     closest_output = closest_output.reshape(-1,2)
    #     ax3.plot(closest_output[:,0], closest_output[:,1], color='b', marker='+', markersize=0.3, linewidth=0.5, label='Predictions')

    # ax4.plot(targets[:,0], targets[:,1], color='green', marker='o', markersize=0.1, linewidth=0.5, label='Ground-Truth')
    # ax4.plot(outputs[:,0], outputs[:,1], color='red', linestyle='None', marker='x', markersize=0.3, linewidth=0.5, label='Predictions') #capsize=0.2
    # ax4.plot(history_traj[:,0], history_traj[:,1], color='cyan', marker='x', markersize=0.3, label='history')

    # ax5.plot(targets[:,0], targets[:,1], color='green', marker='o', markersize=0.3, linewidth=0.3, label='Ground-Truth')
    # # ax3.plot(outputs[:,0], outputs[:,1], color='red', marker='x', markersize=1, label='Predictions')
    # ax5.errorbar(outputs[:,0], outputs[:,1], outputs_var_mean, color='red', linestyle='None', marker='+', markersize=0.4, linewidth=0.4, label='Predictions') #capsize=0.2
    # ax5.plot(history_traj[:,0], history_traj[:,1], color='cyan', marker='+', markersize=0.3, label='history')

    # if final_index != -1:
    #     closest_output = output_list[final_index]
    #     closest_output = closest_output.reshape(-1,2)
    #     ax5.plot(closest_output[:,0], closest_output[:,1], color='b', marker='+', markersize=0.3, linewidth=0.5, label='Predictions')

    # ax6.plot(targets[:,0], targets[:,1], color='green', marker='o', markersize=0.1, linewidth=0.5, label='Ground-Truth')
    # ax6.plot(outputs[:,0], outputs[:,1], color='red', linestyle='None', marker='x', markersize=0.3, linewidth=0.5, label='Predictions') #capsize=0.2
    # ax6.plot(history_traj[:,0], history_traj[:,1], color='cyan', marker='x', markersize=0.3, label='history')

    # for multiple_output in output_list:
    #     multiple_output = multiple_output.reshape(-1,2)
    #     c = np.random.rand(3,)
    #     ax1.plot(multiple_output[:,0], multiple_output[:,1], color=c, marker='x', markersize=0.2, linewidth=1, label='Predictions')
    #     ax4.plot(multiple_output[:,0], multiple_output[:,1], color=c, marker='x', markersize=0.2, linewidth=1, label='Predictions')
        # ax6.plot(multiple_output[:,0], multiple_output[:,1], color=c, marker='x', markersize=0.2, linewidth=1, label='Predictions')


    # for i in range(outputs.shape[0]):
    #     ax3.annotate(str(i), # this is the text
    #                 (outputs[i][0],outputs[i][1]), # this is the point to label
    #                 textcoords="offset points", # how to position the text
    #                 xytext=(0,10), # distance from text to points (x,y)
    #                 ha='center') # horizontal alignment can be left, right or center

    #     ax3.annotate(str(i), # this is the text
    #                 (targets[i][0],targets[i][1]), # this is the point to label
    #                 textcoords="offset points", # how to position the text
    #                 xytext=(0,10), # distance from text to points (x,y)
    #                 ha='center') # horizontal alignment can be left, right or center


    if noise:
        fig.savefig(f"{config.plots}/seq_" + str(j) + "_noise.png")
        # fig1.savefig(f"{config.plots}/seq_" + str(j) + "_1_noise.png")
    else:
        fig.savefig(f"{config.plots}/seq_" + str(j) + ".png")
        # fig1.savefig(f"{config.plots}/seq_" + str(j) + "_1.png")

    if plot_figures:
        plt.show()

    # plt.close()
    plt.close('all')
