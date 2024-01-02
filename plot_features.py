import numpy
import matplotlib.pyplot as plt
import pandas as pd

hs_right = pd.read_csv("hs_right.csv", header=None)
hs_right = hs_right.drop(hs_right.columns[0], axis=1)
STEP = 7

def _normalise_skeleton_joints(df):
    """
    Normalises joints in the skeleton dataframe.
    For each frame, the joint locations are normalised relative to the length between MIDDLE_MCP and WRIST

    Args:
        df: dataframe | contains the joint (col) pose/orientation per frame (rows)

    Returns:
        skeleton_df: dataframe | contains normalised joint locations
    """

    skeleton_df = df.copy(deep=True)

    WRIST = skeleton_df.iloc[:, 0:3].to_numpy(copy=True)  # Neck joint position
    MIDDLE_MCP = skeleton_df.iloc[:, 27:30].to_numpy(copy=True)  # Torso joint position

    for i in range(1, len(skeleton_df.columns), STEP):
        # Obtain the current joint's xyz position then normalise the value in the original df
        current_joint_xyz = skeleton_df.iloc[:, i:i + 3].to_numpy()
        # if i < 2:
        #     try:
        #         t_lt = skeleton_df.loc[9059].iloc[i:i+3]
        #         print("success")
        #         print(skeleton_df.loc[9059].iloc[8:11])
        #         print(skeleton_df.loc[9059].iloc[15:18])
        #         print(t_lt)
        #         print('success')
        #     except Exception as e:
        #         pass
        skeleton_df.iloc[:, i:i + 3] = (current_joint_xyz - WRIST) / (WRIST - MIDDLE_MCP)

    return skeleton_df