import argparse


def get_config():
    parser = argparse.ArgumentParser(
        description="EPUCK_SWARM_BEHAVIOR", formatter_class=argparse.RawDescriptionHelpFormatter)

    # Global_params
    parser.add_argument("--max_translational_velocity", type=float, default=0.3)
    parser.add_argument("--max_rotational_velocity", type=float, default=1.0)
    parser.add_argument("--direction_linear_if_alone", type=float, default=0.0)
    parser.add_argument("--direction_angular_if_alone", type=float, default=2.0)

    # Random
    parser.add_argument('--random_walk_timer_period', type=float, default=0.2)
    parser.add_argument("--random_walk_linear", type=float, default=0.15)
    parser.add_argument("--random_walk_angular", type=float, default=3.0)
    parser.add_argument("--random_walk_rot_interval", type=float, default=2.0)
    parser.add_argument("--random_walk_lin_interval_min", type=float, default=2.5)
    parser.add_argument("--random_walk_lin_interval_max", type=float, default=4.5)

    # Attraction
    parser.add_argument('--attraction_max_range', type=float, default=1.50)
    parser.add_argument("--attraction_min_range", type=float, default=0.15)
    parser.add_argument("--attraction_front_attraction", type=float, default=0.0)
    parser.add_argument("--attraction_threshold", type=int, default=1)

    # Dispersion
    parser.add_argument('--dispersion_max_range', type=float, default=0.8)
    parser.add_argument("--dispersion_min_range", type=float, default=0.04)
    parser.add_argument("--dispersion_front_attraction", type=float, default=-1.0)
    parser.add_argument("--dispersion_threshold", type=int, default=1)


    return parser