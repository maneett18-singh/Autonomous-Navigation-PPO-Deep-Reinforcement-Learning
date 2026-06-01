import matplotlib.pyplot as plt
import numpy as np
try:
    import keyboard  # type: ignore
except Exception:
    keyboard = None
 
DT = 0.2
ROBOT_RADIUS = 0.3      # radius of the vehicle
ACC_TRANS_LIMIT = 0.5   # acceleration limit [m/s^2]
ACC_ROT_LIMIT = 0.2     # acceleration limit [rad/s^2]
VEL_TRANS_LIMIT = 5     # velocity limit [m/s]
VEL_ROT_LIMIT = 2       # velocity limit [rad/s]
VEL_ROT_FRACTION = 0.3  # less steering possible at higher velocities

ADD_ACC_NOISE = True  # add noise to acceleration commands

class RaceEnv():
    def __init__(self, gates):
        self.gates = gates
        self.gate_idx = 0
        self.reset()


    def get_action_dim(self):
        # return dimension of action vector
        return 2


    def get_observation_dim(self):
        # return dimension of state/observation vector
        return 5


    def get_action_limits(self):
        # return upper/lower bounds of action vector elements
        return (VEL_TRANS_LIMIT, VEL_ROT_LIMIT)


    def get_gates(self):
        # return list of all gates along the track
        return self.gates


    def get_gate_idx(self):
        # return index of next gate
        return self.gate_idx


    def step(self, action):
        self._calc_next_state(action)
        obs = np.copy(self.state)
        reward = self._calc_reward()
        done = self._calc_done()
        return obs, reward, done


    def reset(self):
        self.gate_idx = 0
        self.last_state = np.zeros(self.get_observation_dim())
        self.state = (np.random.rand(self.get_observation_dim()) - 0.5) * 0.5
        self.state[-2:] = 0
        # print(self.state)
        return np.copy(self.state)


    def manual_control(self):
        if keyboard is None:
            raise RuntimeError(
                "Manual control requires the optional 'keyboard' package. "
                "Install it with: pip install keyboard"
            )
        action = np.zeros(2)

        if keyboard.is_pressed("up"):
            action[0] = ACC_TRANS_LIMIT
        elif keyboard.is_pressed("down"):
            action[0] = -ACC_TRANS_LIMIT
        if keyboard.is_pressed("left"):
            action[1] = ACC_ROT_LIMIT
        elif keyboard.is_pressed("right"):
            action[1] = -ACC_ROT_LIMIT
        return action


    def _calc_next_state(self, action):
        # store last state
        self.last_state = np.copy(self.state)

        # limit acceleration and add noise
        acc_trans, acc_rot = action
        acc_trans = max(acc_trans, -ACC_TRANS_LIMIT)
        acc_trans = min(acc_trans, +ACC_TRANS_LIMIT)
        if ADD_ACC_NOISE:
            acc_trans += np.random.randn() * ACC_TRANS_LIMIT/5

        acc_rot = max(acc_rot, -ACC_ROT_LIMIT)
        acc_rot = min(acc_rot, +ACC_ROT_LIMIT)
        if ADD_ACC_NOISE:
            acc_rot += np.random.randn() * ACC_ROT_LIMIT/5

        vel_trans = self.state[3] + acc_trans
        vel_rot = self.state[4] + acc_rot

        # limit translational velocity
        vel_trans = max(vel_trans, 0)
        vel_trans = min(vel_trans, +VEL_TRANS_LIMIT)

        # limit rotational velocity
        vel_rot_limit = VEL_ROT_LIMIT - VEL_ROT_LIMIT * (1-VEL_ROT_FRACTION)*(abs(vel_trans))/VEL_TRANS_LIMIT
        vel_rot = max(vel_rot, -vel_rot_limit)
        vel_rot = min(vel_rot, +vel_rot_limit)

        # update state
        self.state[0] += DT * np.cos(self.state[2]) * vel_trans
        self.state[1] += DT * np.sin(self.state[2]) * vel_trans
        self.state[2] += DT * vel_rot
        self.state[3] = vel_trans
        self.state[4] = vel_rot


    def _calc_reward(self):
        # calculate reward
        # early out if we are above the maximum number of steps
        # if self.step_idx > self.max_steps: return 0

        if self.gate_idx >= len(self.gates): return self.step_reward

        if self._do_intersect(self.last_state[0:2], self.state[0:2], self.gates[self.gate_idx][0], self.gates[self.gate_idx][1]):
            self.gate_idx += 1
            return self.gate_reward + self.step_reward

        return self.step_reward


    def _calc_done(self):
        # calculate done flag
        if self.gate_idx >= len(self.gates):
            return True
        else:
            return False


    def _orientation(self, p, q, r):
        # 0 -> p, q and r are collinear
        # 1 -> clockwise
        # 2 -> counterclockwise
        val = (q[1]-p[1])*(r[0]-q[0]) - (q[0]-p[0])*(r[1]-q[1])
        if val == 0: return 0
        return 1 if val > 0 else 2


    def _do_intersect(self, p1, q1, p2, q2):
        # return true if line segments p1q1 and p2q2 intersect
        o1 = self._orientation(p1, q1, p2)
        o2 = self._orientation(p1, q1, q2)
        o3 = self._orientation(p2, q2, p1)
        o4 = self._orientation(p2, q2, q1)

        # general case, does not consider the collinear case
        if o1 != o2 and o3 != o4: return True

        return False


    def plot(self):
        plt.figure(1)
        plt.clf()

        # plot robot
        c = plt.Circle((self.state[0], self.state[1]), ROBOT_RADIUS, facecolor='w', edgecolor='k')
        plt.gca().add_patch(c)

        # plot line indicating the robot direction
        plt.plot([self.state[0], self.state[0] + ROBOT_RADIUS * np.cos(self.state[2])],
                 [self.state[1], self.state[1] + ROBOT_RADIUS * np.sin(self.state[2])], 'k')

        # plot gates
        for gate in self.gates:
            plt.plot([gate[0][0], gate[1][0]], [gate[0][1], gate[1][1]], 'k')

        # highlight next gate
        if self.gate_idx < len(self.gates):
            highl = self.gates[self.gate_idx]
            plt.plot([highl[0][0], highl[1][0]], [highl[0][1], highl[1][1]], 'r', linewidth="2")

        plt.gca().axis('equal')
        plt.pause(0.001)  # pause a bit so that plots are updated


class StraightRace(RaceEnv):
    def __init__(self):
        self.step_reward = 0
        self.gate_reward = 1
        gates = [[[5, -1],  [5, +1]],
                 [[10, -1], [10, +1]],
                 [[15, -1], [15, +1]],
                 [[20, -1], [20, +1]],
                 [[25, -1], [25, +1]],
                 [[30, -1], [30, +1]],
                 [[35, -1], [35, +1]],
                 [[40, -1], [40, +1]]];
        RaceEnv.__init__(self, gates=gates)


class CurvyRace(RaceEnv):
    def __init__(self):
        self.step_reward = 0
        self.gate_reward = 1
        gates = [[[2, -1],  [2, 1]],
                 [[4, -1],  [4, 1]],
                 [[6, 0], [6, 2]],
                 [[8, 0.5], [8, 2.5]],
                 [[10, 0], [10, 2]],
                 [[12, -1], [12, 1]],
                 [[14, -2], [14, 0]],
                 [[16, -2.5], [16, -0.5]],
                 [[18, -2], [18, 0]],
                 [[20, -1], [20, 1]],
                 [[22, 0], [22, 2]],
                 [[24, 0.5], [24, 2.5]],
                 [[26, 0], [26, 2]],
                 [[28, -1], [28, 1]],
                 [[30, -2], [30, 0]],
                 [[32, -2.5], [32, -0.5]]
                 ]
        RaceEnv.__init__(self, gates=gates)


class Parcour(RaceEnv):
    def __init__(self):
        self.step_reward = 0
        self.gate_reward = 1
        gates = [[[2,1], [2,-1]],
                 [[4,1], [4,-1]],
                 [[6,1], [7,-1]],
                 [[7,2], [9,1]],
                 [[7,4], [10,4]],
                 [[7,6], [9,7]],
                 [[6,8], [8,9]],
                 [[5,10], [7,11]],
                 [[4,13], [7,13]],
                 [[5,16], [7,15]],
                 [[7,18], [8,16]],
                 [[10,18], [10,16]],
                 [[12,18], [12,16]],
                 [[14,18], [14,16]],
                 [[17,18], [16,16]],
                 [[19,16], [17,15]],
                 [[19,13], [17,13]],
                 [[19,11], [17,11]],
                 [[19,9], [18,9]],
                 [[19,7], [17,7]],
                 [[19,5], [17,5]],
                 [[19,3], [17,3]],
                 [[18,1], [17,1]],
                 [[19,-1], [17,-1]],
                 [[19,-3], [17,-3]],
                 [[19,-6], [17,-5]],
                 [[17,-9], [16,-6]],
                 [[14,-10], [14,-6]],
                 [[12,-9], [12,-6]],
                 [[10,-8], [10,-6]],
                 [[8,-8], [8,-6]],
                 [[6,-7], [6,-6]],
                 [[4,-7], [4,-6]],
                 [[2,-7], [2,-6]],
                 ]
        RaceEnv.__init__(self, gates=gates)



if __name__ == "__main__":
    env = Parcour()
    done = False
    t = 0

    while(not done):
        t+=1
        print(t)
        action = env.manual_control()
        obs, reward, done = env.step(action)
        env.plot()

    print(t)
    plt.show()

