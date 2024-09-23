import matplotlib.pyplot as plt
import numpy as np

def cumulative_average(input_list, window_width):
    """
    Computes the cumulative moving average of the input list over a specified window width.

    Parameters:
    input_list (list): List of values to compute the moving average for.
    window_width (int): The size of the window over which to compute the average.

    Returns:
    output_list (list): List of cumulative averages computed using the specified window width.
    """

    output_list = []
    
    for i in range(len(input_list)):
        if i + 1 >= window_width:
            window_sum = sum(input_list[i+1-window_width:i+1])
            average = window_sum / window_width
            output_list.append(average)
    
    return output_list

def compute_avg_and_stddev(data_list):
    """
    Computes the mean and standard deviation of the input data along a specific axis.

    Parameters:
    data_list (list): List of data arrays to compute the average and standard deviation for.

    Returns:
    averages (ndarray): Array of averages for the input data.
    std_devs (ndarray): Array of standard deviations for the input data.
    """
    
    data_array = np.array(data_list)
    averages = np.mean(data_array, axis=0)
    std_devs = np.std(data_array, axis=0)

    return averages, std_devs

def plot_test_and_rnd_rewards(test_rewards, rnd_rewards, window_width):
    """
    Plots the average test and random (RND) rewards along with their standard deviation.

    Parameters:
    test_rewards (list): List of test reward sequences.
    rnd_rewards (list): List of random (RND) reward sequences.
    window_width (int): The window width used to compute the cumulative average.

    Returns:
    None: Displays the plot of rewards and their standard deviations.
    """
    
    test_reward_trends = []
    rnd_reward_trands = []
    for i,(test_reward, rnd_reward) in enumerate(zip(test_rewards, rnd_rewards)):
        avg_test_rewards = cumulative_average(test_reward, window_width)
        avg_rnd_rewards = cumulative_average(rnd_reward, window_width)

        test_reward_trends.append(avg_test_rewards)
        rnd_reward_trands.append(avg_rnd_rewards)
    avg_test_reward, stddev_test_reward = compute_avg_and_stddev(test_reward_trends)
    avg_rnd_reward, stddev_rnd_reward = compute_avg_and_stddev(rnd_reward_trands)

    x_values = range(window_width, len(test_reward) + 1)

    plt.plot(x_values, avg_test_reward, label='Test Rewards', color='blue', alpha=1)
    plt.fill_between(x_values, avg_test_reward+stddev_test_reward, avg_test_reward-stddev_test_reward,label = ' Test Standard Deviation', color='blue', alpha=0.4)
    plt.plot(x_values, avg_rnd_reward, label='RND Rewards', color='orange', alpha=1)
    plt.fill_between(x_values, avg_rnd_reward+stddev_rnd_reward, avg_rnd_reward-stddev_rnd_reward,label = ' Random Standard Deviation', color='orange', alpha=0.4)
    plt.xlabel('Index')
    plt.ylabel('Rewards')
    plt.title('Test and RND Average Rewards with Standard Deviation')
    plt.legend()
    plt.show()

def plot_test_and_rnd_ep_length(test_ep_lengths, rnd_ep_lengths, window_width):
    """
    Plots the average episode lengths for test and random (RND) episodes along with their standard deviation.

    Parameters:
    test_ep_lengths (list): List of test episode length sequences.
    rnd_ep_lengths (list): List of random (RND) episode length sequences.
    window_width (int): The window width used to compute the cumulative average.

    Returns:
    None: Displays the plot of episode lengths and their standard deviations.
    """
    
    test_ep_length_trends = []
    rnd_ep_length_trands = []
    for i,(test_ep_length, rnd_ep_length) in enumerate(zip(test_ep_lengths, rnd_ep_lengths)):
        avg_test_ep_lengths = cumulative_average(test_ep_length, window_width)
        avg_rnd_ep_lengths = cumulative_average(rnd_ep_length, window_width)

        test_ep_length_trends.append(avg_test_ep_lengths)
        rnd_ep_length_trands.append(avg_rnd_ep_lengths)

    avg_test_ep_length, stddev_test_ep_length = compute_avg_and_stddev(test_ep_length_trends)
    avg_rnd_ep_length, stddev_rnd_ep_length = compute_avg_and_stddev(rnd_ep_length_trands)

    x_values = range(window_width, len(test_ep_length) + 1)

    plt.plot(x_values, avg_test_ep_length, label='Test Episode Lengths', color='blue', alpha=1)
    plt.fill_between(x_values, avg_test_ep_length+stddev_test_ep_length, avg_test_ep_length-stddev_test_ep_length,label = ' Test Standard Deviation', color='blue', alpha=0.4)
    plt.plot(x_values, avg_rnd_ep_length, label='RND Episode Lengths', color='orange', alpha=1)
    plt.fill_between(x_values, avg_rnd_ep_length+stddev_rnd_ep_length, avg_rnd_ep_length-stddev_rnd_ep_length,label = ' Random Standard Deviation', color='orange', alpha=0.4)
    plt.xlabel('Index')
    plt.ylabel('Episode Lengths')
    plt.title('Test and RND Average Episode Lengths with Standard Deviation')
    plt.legend()
    plt.show()

def plot_avg_reward(reward_list, games_list, window_width):
    """
    Plots the average reward trend for a list of games over a specified window width.

    Parameters:
    reward_list (list): List of reward sequences for different games.
    games_list (list): List of names corresponding to each reward sequence.
    window_width (int): The window width used to compute the cumulative average.

    Returns:
    None: Displays the plot for average rewards for each game.
    """
    
    for i, (rewards, name) in enumerate(zip(reward_list, games_list)):
        avg_reward = cumulative_average(rewards, window_width)
        x_values = range(window_width, len(rewards) + 1)

        plt.figure(i + 1)
        plt.plot(x_values, avg_reward, label=f'{name} Average Reward')
        plt.xlabel('Index')
        plt.ylabel('Values')
        plt.title(f'{name} Reward')
        plt.legend()

    plt.show()

def plot_avg_ep_length(ep_length_list, games_list, window_width):
    """
    Plots the average episode length trend for a list of games over a specified window width.

    Parameters:
    ep_length_list (list): List of episode length sequences for different games.
    games_list (list): List of names corresponding to each episode length sequence.
    window_width (int): The window width used to compute the cumulative average.

    Returns:
    None: Displays the plot for average episode lengths for each game.
    """
    
    for i, (ep_length, name) in enumerate(zip(ep_length_list, games_list)):
        avg_length = cumulative_average(ep_length, window_width)
        x_values = range(window_width, len(ep_length) + 1)

        plt.figure(i + 1)
        plt.plot(x_values, avg_length, label=f'{name} Average Episode Length')
        plt.xlabel('Index')
        plt.ylabel('Values')
        plt.title(f'{name} Episode Length')
        plt.legend()

    plt.show()