from humoro.trajectory import Trajectory

full_traj = Trajectory()
full_traj.loadTrajHDF5("../Pose2Gaze/datasets/public/mogaze/p1_1_human_data.hdf5")

print("The data has dimension timeframe, state_size:")
print(full_traj.data.shape)  # (53899, 66)
print("")
print("This is a list of jointnames (from the urdf) corresponding to the state dimensions:")
print(list(full_traj.description))
"""
['baseTransX', 'baseTransY', 'baseTransZ', 'baseRotX', 'baseRotY', 'baseRotZ', 'pelvisRotX', 'pelvisRotY', 'pelvisRotZ', 'torsoRotX', 
 'torsoRotY', 'torsoRotZ', 'neckRotX', 'neckRotY', 'neckRotZ', 'headRotX', 'headRotY', 'headRotZ', 'linnerShoulderRotX', 'linnerShoulderRotY', 
 'linnerShoulderRotZ', 'lShoulderRotX', 'lShoulderRotY', 'lShoulderRotZ', 'lElbowRotX', 'lElbowRotY', 'lElbowRotZ', 'lWristRotX', 'lWristRotY', 
 'lWristRotZ', 'rinnerShoulderRotX', 'rinnerShoulderRotY', 'rinnerShoulderRotZ', 'rShoulderRotX', 'rShoulderRotY', 'rShoulderRotZ', 'rElbowRotX', 
 'rElbowRotY', 'rElbowRotZ', 'rWristRotX', 'rWristRotY', 'rWristRotZ', 'lHipRotX', 'lHipRotY', 'lHipRotZ', 'lKneeRotX', 'lKneeRotY', 'lKneeRotZ', 
 'lAnkleRotX', 'lAnkleRotY', 'lAnkleRotZ', 'lToeRotX', 'lToeRotY', 'lToeRotZ', 'rHipRotX', 'rHipRotY', 'rHipRotZ', 'rKneeRotX', 'rKneeRotY', 'rKneeRotZ', 
 'rAnkleRotX', 'rAnkleRotY', 'rAnkleRotZ', 'rToeRotX', 'rToeRotY', 'rToeRotZ']
"""
print("")
print("Some joints are used for scaling the human and do not change over time")
print("They are available in a dictionary:")
print(full_traj.data_fixed)
"""
{ 'pelvisTransX': np.float64(3.477253617581374e-11), 
 'pelvisTransY': np.float64(9.836376779497708e-11), 
 'pelvisTransZ': np.float64(0.07405368236962184), 
 'torsoTransX': np.float64(-1.0047164753054265e-05), 
 'torsoTransY': np.float64(-3.0239514916602615e-05), 
 'torsoTransZ': np.float64(0.20083891659362765), 
 'neckTransX': np.float64(-1.2600904410529728e-10), 
 'neckTransY': np.float64(-1.7764181211437644e-09), 
 'neckTransZ': np.float64(0.23359122887702816), 
 'headTransX': np.float64(3.1869674957629284e-09), 
 'headTransY': np.float64(-0.018057959551160224), 
 'headTransZ': np.float64(0.13994911705542687), 
 'linnerShoulderTransX': np.float64(0.036502066101529215), 
 'linnerShoulderTransY': np.float64(0.00031381713305931404), 
 'linnerShoulderTransZ': np.float64(0.1832770110533476), 
 'lShoulderTransX': np.float64(0.15280389623890986), 
 'lShoulderTransY': np.float64(-1.1382671917429665e-09), 
 'lShoulderTransZ': np.float64(-8.843481859722201e-10), 
 'lElbowTransX': np.float64(0.24337935578849854), 
 'lElbowTransY': np.float64(3.1846829220430116e-09), 
 'lElbowTransZ': np.float64(1.4634072008156868e-08), 
 'lWristTransX': np.float64(0.2587863246939117), 
 'lWristTransY': np.float64(-0.001083441187206969), 
 'lWristTransZ': np.float64(-2.579650873773653e-05), 
 'rinnerShoulderTransX': np.float64(-0.035725060316154335), 
 'rinnerShoulderTransY': np.float64(0.00031381736864692205), 
 'rinnerShoulderTransZ': np.float64(0.18327701132224603), 
 'rShoulderTransX': np.float64(-0.15280389611921505), 
 'rShoulderTransY': np.float64(1.5047363672598406e-10), 
 'rShoulderTransZ': np.float64(-3.5061608640836945e-10), 
 'rElbowTransX': np.float64(-0.24337935345513284), 
 'rElbowTransY': np.float64(1.7367526549061563e-09), 
 'rElbowTransZ': np.float64(8.129849550012404e-09), 
 'rWristTransX': np.float64(-0.26714046782640866), 
 'rWristTransY': np.float64(-0.0018438929787374019), 
 'rWristTransZ': np.float64(0.0002976280123487421), 
 'lHipTransX': np.float64(0.09030938177511867), 
 'lHipTransY': np.float64(-3.9169896274006325e-10), 
 'lHipTransZ': np.float64(6.535337361293777e-11), 
 'lKneeTransX': np.float64(2.3363012617863798e-09), 
 'lKneeTransY': np.float64(-6.968930551857006e-09), 
 'lKneeTransZ': np.float64(-0.3833567539280885), 
 'lAnkleTransX': np.float64(-0.00022773850984784621), 
 'lAnkleTransY': np.float64(-0.0010201879980305337), 
 'lAnkleTransZ': np.float64(-0.35354742240551973), 
 'lToeTransX': np.float64(-8.62756280059957e-10), 
 'lToeTransY': np.float64(-0.13546407147725723), 
 'lToeTransZ': np.float64(-0.05870108800865852), 
 'rHipTransX': np.float64(-0.09030938169548784), 
 'rHipTransY': np.float64(3.1646243068853757e-10), 
 'rHipTransZ': np.float64(-6.760126625141776e-11), 
 'rKneeTransX': np.float64(-2.5236199070228006e-09), 
 'rKneeTransY': np.float64(-1.1574579905441608e-09), 
 'rKneeTransZ': np.float64(-0.38335675152180304), 
 'rAnkleTransX': np.float64(0.0005504841925316687), 
 'rAnkleTransY': np.float64(0.0004893600191017677), 
 'rAnkleTransZ': np.float64(-0.3414929392712298), 
 'rToeTransX': np.float64(-1.2153921360447088e-09), 
 'rToeTransY': np.float64(-0.13546407016650086), 
 'rToeTransZ': np.float64(-0.058701085864675616)}
"""

from humoro.player_pybullet import Player
pp = Player()
pp.spawnHuman("Human1")
pp.addPlaybackTraj(full_traj, "Human1")
# pp.showFrame(3000)
# pp.play(duration=3600, startframe=3000)

# Playback multiple humans at the same time

pp.spawnHuman("Human2", color=[0., 1., 0., 1.])
# this extracts a subtrajectory from the full trajectory:
sub_traj = full_traj.subTraj(3000, 3360)
# we change the startframe of the sub_traj,
# thus the player will play it at a different time:
sub_traj.startframe = 4000
pp.addPlaybackTraj(sub_traj, "Human2")
# pp.play(duration=3600, startframe=4000)

# Loading Objects
from humoro.load_scenes import autoload_objects
obj_trajs, obj_names = autoload_objects(pp, "../Pose2Gaze/datasets/public/mogaze/p1_1_object_data.hdf5", "../Pose2Gaze/datasets/public/mogaze/scene.xml")
pp.play(duration=360, startframe=3000)

# Loading Gaze
from humoro.gaze import load_gaze
gaze_traj = load_gaze("../Pose2Gaze/datasets/public/mogaze/p1_1_gaze_data.hdf5")
pp.addPlaybackTrajGaze(gaze_traj)
pp.play(duration=360, startframe=3000)

pp.play_controls("../Pose2Gaze/datasets/public/mogaze/p1_1_segmentations.hdf5")

import h5py
with h5py.File("mogaze/p1_1_segmentations.hdf5", "r") as segfile:
    # print first 5 segments:
    for i in range(5):
        print(segfile["segments"][i])


# 阻塞防止窗口立即关闭
import time
print("Simulation running... Close window or stop manually to exit.")
while True:
    time.sleep(1)


