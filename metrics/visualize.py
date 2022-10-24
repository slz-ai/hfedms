import visualization_utils

SHOW_WEIGHTED = True  # show weighted accuracy instead of unweighted accuracy
PLOT_CLIENTS = True
PLOT_SET = "Test"  # "Test" or "Train"
PLOT_MOVE_AVG = False
WINDOW_SIZE = 5
stat_file = "metrics_stat_0.csv"  # change to None if desired
sys_file = "metrics_sys_0.csv"  # change to None if desired


def plot_acc_vs_round(stat_metrics, sys_metrics):
    """Plots accuracy vs. round number."""
    if stat_metrics is not None:
        visualization_utils.plot_accuracy_vs_round_number(
            stat_metrics, use_set=PLOT_SET, weighted=SHOW_WEIGHTED, plot_stds=False)
    if PLOT_CLIENTS and stat_metrics is not None:
        visualization_utils.plot_accuracy_vs_round_number_per_client(
            stat_metrics, sys_metrics, max_num_clients=10, use_set=PLOT_SET)


def plot_loss_vs_round(stat_metrics, sys_metrics):
    """Plots loss vs. round number."""
    if stat_metrics is not None:
        visualization_utils.plot_loss_vs_round_number(
            stat_metrics, use_set=PLOT_SET, weighted=SHOW_WEIGHTED, plot_stds=False)
    if PLOT_CLIENTS and stat_metrics is not None:
        visualization_utils.plot_loss_vs_round_number_per_client(
            stat_metrics, sys_metrics, max_num_clients=10, use_set=PLOT_SET)


def plot_bytes_vs_round(stat_metrics, sys_metrics):
    """Plots the cumulative sum of the bytes pushed and pulled by clients in
    the past rolling_window rounds versus the round number.
    """
    if stat_metrics is not None:
        visualization_utils.plot_bytes_written_and_read(
            sys_metrics, rolling_window=500)


def plot_comp_vs_round(stat_metrics, sys_metrics):
    visualization_utils.plot_client_computations_vs_round_number(
        sys_metrics, aggregate_window=10, max_num_clients=20, range_rounds=(0, 499))


def calc_longest_flops(stat_metrics, sys_metrics):
    print("Longest FLOPs path: %s" %
          visualization_utils.get_longest_flops_path(sys_metrics))


def compare_accuracy_vs_round(metrics, legend):
    """Compare accuracy vs. round number across experiments."""
    visualization_utils.compare_accuracy_vs_round_number(
        metrics, legend, use_set=PLOT_SET, weighted=SHOW_WEIGHTED,
        move_avg=PLOT_MOVE_AVG, window_size=WINDOW_SIZE, plot_stds=False)


def compare_loss_vs_round(metrics, legend):
    """Compare loss vs. round number across experiments."""
    visualization_utils.compare_loss_vs_round_number(
        metrics, legend, use_set=PLOT_SET, weighted=SHOW_WEIGHTED,
        move_avg=PLOT_MOVE_AVG, window_size=WINDOW_SIZE, plot_stds=False)


if __name__ == "__main__":
    stat_files = (
        "metrics_stat_65_linear_0.02_10_3_500.csv",
        "metrics_stat_67_linear_0.02_10_3_500.csv",
        "metrics_stat_30_log_2.0_5.csv",
        "metrics_stat_0(y=5.793lnx-0.5-1-500).csv"


    )
    legend = (
        "a",
        "b",
        "c",
        "d"

    )

    metrics = [visualization_utils.load_data(f)[0]
               for f in stat_files]

    compare_accuracy_vs_round(metrics, legend)
    compare_loss_vs_round(metrics, legend)
