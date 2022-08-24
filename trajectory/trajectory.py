import numpy as np
import sys
from pose_utils import mat2euler, euler2mat

def pred_to_traj(seq, name):
    ret = np.zeros((len(seq)+1, 4, 4))
    init_pose = np.eye(4)
    ret[0] = init_pose

    for idx, pose in enumerate(seq):
        prev_pose = ret[idx]
        ret[idx+1] = prev_pose @ pose

    for i in range(len(ret)):
        rot = ret[i][:3, :3]
        trans = ret[i][:3, 3]

        z,y,x = mat2euler(rot)
        rot = euler2mat(-z,-y,x)
        trans[1:] *= -1

        ret[i][:3, :3] = rot
        ret[i][:3, 3] = trans

    ret = np.reshape(ret[:,:3,:], (-1, 12))
    np.savetxt(name, ret, fmt='%1.6e')

def main(argv):
    seqs = ['00', '03', '04', '05', '07', '09', '10']
    model = argv[1]
    for seq in seqs:
        try:
            pred = np.load(f"{model}/{seq}.npy")
        except FileNotFoundError:
            print(f"{seq} not exists")
            continue
        pred_to_traj(pred, f"{seq}/{model}.txt")
        pred_to_traj(pred, f"{model}/{seq}.txt")

if __name__ == '__main__':
    main(sys.argv)