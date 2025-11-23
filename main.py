from argparse import ArgumentParser

from Reinf_Learn import launch_q_learning_simulation

if __name__ == '__main__':
    parser = ArgumentParser(description="Dynamic Traffic Signal Control System")

    parser.add_argument(
        "-e", "--episodes",
        metavar='N',
        type=int,
        required=True,
        help="Number of evaluation episodes to run"
    )

    parser.add_argument(
        "-r", "--render",
        action='store_true',
        help="Displays the simulation window"
    )

    args = parser.parse_args()

    launch_q_learning_simulation(num_episodes=args.episodes, render=args.render)
