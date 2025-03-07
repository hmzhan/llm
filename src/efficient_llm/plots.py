import pandas as pd
import matplotlib.pyplot as plt


def plot_metrics(perf_metrics, current_optim_type):
    df = pd.DataFrame.from_dict(perf_metrics, orient='index')
    for idx in df.index:
        df_opt = df.loc[idx]
        if idx == current_optim_type:
            plt.scatter(df_opt['time_avg_ms'], df_opt['accuracy'] * 100, alpha=0.5, s=df_opt['size_mb'], label=idx,
                        marker='$\u25CC$')
        else:
            plt.scatter(df_opt['time_avg_ms'], df_opt['accuracy'] * 100, alpha=0.5, s=df_opt['size_mb'], label=idx)

    legend = plt.legend(bbox_to_anchor=(1, 1))
    for handle in legend.legendHandles:
        handle.set_sizes([20])

    plt.ylim(80, 90)
    xlim = int(perf_metrics['BERT baseline']['time_avg_ms'] + 3)
    plt.xlim(1, xlim)
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Average latency (ms)')
    plt.show()
