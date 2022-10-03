import argparse
import time
import msgpack
from enum import Enum, auto
import copy

import numpy as np
import random

from planning_utils import a_star, heuristic, create_grid
from udacidrone import Drone
from udacidrone.connection import MavlinkConnection
from udacidrone.messaging import MsgID
from udacidrone.frame_utils import global_to_local


class States(Enum):
    MANUAL = auto()
    ARMING = auto()
    TAKEOFF = auto()
    WAYPOINT = auto()
    LANDING = auto()
    DISARMING = auto()
    PLANNING = auto()


class MotionPlanning(Drone):

    def __init__(self, connection):
        super().__init__(connection)

        self.target_position = np.array([0.0, 0.0, 0.0])
        self.waypoints = []
        self.in_mission = True
        self.check_state = {}

        # initial state
        self.flight_state = States.MANUAL

        # register all your callbacks here
        self.register_callback(MsgID.LOCAL_POSITION, self.local_position_callback)
        self.register_callback(MsgID.LOCAL_VELOCITY, self.velocity_callback)
        self.register_callback(MsgID.STATE, self.state_callback)

    def local_position_callback(self):
        if self.flight_state == States.TAKEOFF:
            if -1.0 * self.local_position[2] > 0.95 * self.target_position[2]:
                self.waypoint_transition()
        elif self.flight_state == States.WAYPOINT:
            if np.linalg.norm(self.target_position[0:2] - self.local_position[0:2]) < 1.0:
                if len(self.waypoints) > 0:
                    self.waypoint_transition()
                else:
                    if np.linalg.norm(self.local_velocity[0:2]) < 1.0:
                        self.landing_transition()

    def velocity_callback(self):
        if self.flight_state == States.LANDING:
            if self.global_position[2] - self.global_home[2] < 0.1:
                if abs(self.local_position[2]) < 0.01:
                    self.disarming_transition()

    def state_callback(self):
        if self.in_mission:
            if self.flight_state == States.MANUAL:
                self.arming_transition()
            elif self.flight_state == States.ARMING:
                if self.armed:
                    self.plan_path()
            elif self.flight_state == States.PLANNING:
                self.takeoff_transition()
            elif self.flight_state == States.DISARMING:
                if ~self.armed & ~self.guided:
                    self.manual_transition()

    def arming_transition(self):
        self.flight_state = States.ARMING
        print("arming transition")
        self.arm()
        self.take_control()

    def takeoff_transition(self):
        self.flight_state = States.TAKEOFF
        print("takeoff transition")
        self.takeoff(self.target_position[2])

    def waypoint_transition(self):
        self.flight_state = States.WAYPOINT
        print("waypoint transition")
        self.target_position = self.waypoints.pop(0)
        print('target position', self.target_position)
        self.cmd_position(self.target_position[0], self.target_position[1], self.target_position[2], self.target_position[3])

    def landing_transition(self):
        self.flight_state = States.LANDING
        print("landing transition")
        self.land()

    def disarming_transition(self):
        self.flight_state = States.DISARMING
        print("disarm transition")
        self.disarm()
        self.release_control()

    def manual_transition(self):
        self.flight_state = States.MANUAL
        print("manual transition")
        self.stop()
        self.in_mission = False

    def send_waypoints(self):
        print("Sending waypoints to simulator ...")
        data = msgpack.dumps(self.waypoints)
        self.connection._master.write(data)

    def plan_path(self):
        self.flight_state = States.PLANNING
        print("Searching for a path ...")
        TARGET_ALTITUDE = 20
        SAFETY_DISTANCE = 5

        self.target_position[2] = TARGET_ALTITUDE

        # read lat0, lon0 from colliders into floating point values
        data = np.loadtxt('colliders.csv', delimiter=',', dtype='U16', usecols = (0,1), max_rows=1)
        lat0 = np.float64(data[0].split(' ')[1])
        lon0 = np.float(data[1].split(' ')[2])
        
        # set home position to (lon0, lat0, 0)
        self.set_home_position(lon0,lat0,0)

        global_pos = self.global_position

        # retrieve current global position
        # convert to current local position using global_to_local()
        # Converts to NED using set_home_position as global_home origin 
        # And global_position as current Geodetic coordinates of drone

        local_pos = global_to_local(self.global_position, self.global_home)
        
        print('global home {0}, position {1}, local position {2}'.format(self.global_home, self.global_position,
                                                                         self.local_position))
        
        # Read in obstacle map
        # NED Coordinates in meters from geodetic origin
        data = np.loadtxt('colliders.csv', delimiter=',', dtype='float', skiprows=2)
        print(data)
        
        # Define a grid for a particular altitude and safety margin around obstacles
        grid, north_offset, east_offset = create_grid(data, TARGET_ALTITUDE, SAFETY_DISTANCE)
        print("North offset = {0}, east offset = {1}".format(north_offset, east_offset))
       
        # TODO: convert start position to current position rather than map center
       




       # grid_start = (int(local_pos[0]), int(local_pos[1]))
        grid_start = (north_offset,east_offset)
        print(grid_start)

        # Set goal as some arbitrary position on the grid
        north_min = np.floor(np.min(data[:, 0] - data[:, 3]))
        north_max = np.ceil(np.max(data[:, 0] + data[:, 3]))

        east_min = np.floor(np.min(data[:, 1] - data[:, 4]))
        east_max = np.ceil(np.max(data[:, 1] + data[:, 4]))

        grid_goal_north = np.min(data[:, 0])
        grid_goal_east = np.min(data[:, 1])

        print(f'{grid_goal_north}, {grid_goal_east}')

        while(grid_goal_north in (data[:, 0])):
            grid_goal_north = random.randrange(north_min, north_max)

        while(grid_goal_east in (data[:, 1])):
            grid_goal_east = random.randrange(east_min, east_max)

        grid_goal = (grid_goal_north, grid_goal_east)

        # TODO: adapt to set goal as latitude / longitude position and convert
        #grid_start = local_pos
        #local_goal = global_to_local([lon0,lat0, 0], self.global_home)
        #grid_goal = (int(-north_offset + local_goal[0]), int(-east_offset + local_goal[1]))

        # Run A* to find a path from start to goal
        # TODO: add diagonal motions with a cost of sqrt(2) to your A* implementation
        # or move to a different search space such as a graph (not done here)
        print('Local Start and Goal: ', grid_start, grid_goal)
        #print(grid)
        path, _ = a_star(grid, heuristic, grid_start, grid_goal)
        #print(path)

        # TODO: prune path to minimize number of waypoints
        # TODO (if you're feeling ambitious): Try a different approach altogether!

        pruned_path =  copy.deepcopy(path)
        for i in range(1, len(path) - 1):
            p1 = path[i-1]
            p2 = path[i]
            p3 = path[i+1]
            det = p1[0]*(p2[1] - p3[1]) + p2[0]*(p3[1] - p1[1]) + p3[0]*(p1[1] - p2[1])
            if det == 0:
                pruned_path.remove(p2)
        path = pruned_path

        print(f'Path: {path}')

        # Convert path to waypoints
        waypoints = [[p[0] + local_pos[0], p[1] + local_pos[1], TARGET_ALTITUDE, 0] for p in path]
        # Set self.waypoints
        self.waypoints = waypoints
        
        # TODO: send waypoints to sim (this is just for visualization of waypoints)
        self.send_waypoints()

    def start(self):
        self.start_log("Logs", "NavLog.txt")

        print("starting connection")
        self.connection.start()

        # Only required if they do threaded
        # while self.in_mission:
        #    pass

        self.stop_log()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5760, help='Port number')
    parser.add_argument('--host', type=str, default='127.0.0.1', help="host address, i.e. '127.0.0.1'")
    args = parser.parse_args()

    conn = MavlinkConnection('tcp:{0}:{1}'.format(args.host, args.port), timeout=60)
    drone = MotionPlanning(conn)
    time.sleep(1)

    drone.start()
