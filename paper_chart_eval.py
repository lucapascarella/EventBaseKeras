import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _main(flags: argparse) -> None:
    df_result = pd.read_csv(flags.input_file)

    df_result = df_result.loc[250:450]

    plt.plot(df_result["real_angle"])
    plt.plot(df_result["prediction_angle_aps"])
    plt.plot(df_result["prediction_angle_dvs"])
    plt.plot(df_result["prediction_angle_cmb"])
    plt.plot(df_result["prediction_angle_dbl"])


    plt.title("Steering Predictions vs. Ground Truth")
    plt.ylabel("Steering angles")
    plt.xlabel("Frames")
    plt.gca().set_ylim([-35, 35])
    plt.legend(["Ground Truth", "GEFU@1 (APS)", "GEFU@1 (DVS)", "GEFU@1 (CMB)", "GEFU@2 (DBL)"], loc='upper right')

    plt.savefig("steering_predictions.pdf")
    plt.show()

    # error_steer = np.sqrt(np.mean(np.square(pred_steer - gt_steer), axis=0))
    # print("RMSE: {:.2f}".format(error_steer))
    # ex_variance = explained_variance_1d(pred_steer, gt_steer)
    # print("EVA: {:.2f}".format(ex_variance))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", help="Input file", type=str, default=None)
    args = parser.parse_args()

    _main(args)
