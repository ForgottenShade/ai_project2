from traffic.data import samples
from geopy.distance import geodesic
from numpy.random import default_rng
import numpy as np
import pyproj

from pykalman import KalmanFilter
import matplotlib.pyplot as plt


# returns a list of flights with the original GPS data
def get_ground_truth_data():
    names = ['liguria', 'pixair_toulouse', 'indiana', 'texas', 'georeal_fyn_island', 'ign_mercantour',
             'ign_fontainebleau', 'mecsek_mountains', 'ign_lot_et_garonne', 'inflight_refuelling', 'aircraft_carrier',
             'luberon', 'alto_adige', 'franconia', 'danube_valley', 'cevennes', 'oxford_cambridge', 'alpi_italiane',
             'rega_zh', 'samu31', 'rega_sg', 'monastir', 'guatemala', 'london_heathrow', 'cardiff', 'sydney',
             'brussels_ils', 'ajaccio', 'toulouse', 'noumea', 'london_gatwick', 'perth', 'kota_kinabalu', 'montreal',
             'funchal', 'nice', 'munich', 'vancouver', 'lisbon', 'liege_sprimont', 'kiruna', 'bornholm', 'kingston',
             'brussels_vor', 'vienna', 'border_control', 'dreamliner_boeing', 'texas_longhorn', 'zero_gravity',
             'qantas747', 'turkish_flag', 'airbus_tree', 'easter_rabbit', 'belevingsvlucht', 'anzac_day', 'thankyou',
             'vasaloppet']
    return [samples.__getattr__(x) for x in names]


# needed for set_lat_lon_from_x_y below
# is set by get_radar_data()
projection_for_flight = {}


# Returns the same list of flights as get_ground_truth_data(), but with the position data modified as if it was a reading from a radar
# i.e., the data is less accurate and with fewer points than the one from get_ground_truth_data()
# The flights in this list will have x, y coordinates set to suitable 2d projection of the lat/lon positions.
# You can access these coordinates in the Flight.data attribute, which is a Pandas DataFrame.
def get_radar_data():
    rng = default_rng()
    radar_error = 0.1  # in kilometers
    gt = get_ground_truth_data()
    radar_data = []
    for flight in gt:
        print("flight: %s" % (str(flight)))
        flight_radar = flight.resample("10s")
        for i in range(len(flight_radar.data)):
            point = geodesic(kilometers=rng.normal() * radar_error).destination(
                (flight_radar.data.at[i, "latitude"], flight_radar.data.at[i, "longitude"]), rng.random() * 360)
            (flight_radar.data.at[i, "latitude"], flight_radar.data.at[i, "longitude"]) = (
                point.latitude, point.longitude)
            # print("after: %f, %f" % (flight_radar.data.at[i,"latitude"], flight_radar.data.at[i,"longitude"]))
        projection = pyproj.Proj(proj="lcc", ellps="WGS84", lat_1=flight_radar.data.latitude.min(),
                                 lat_2=flight_radar.data.latitude.max(), lat_0=flight_radar.data.latitude.mean(),
                                 lon_0=flight_radar.data.longitude.mean())
        flight_radar = flight_radar.compute_xy(projection)
        projection_for_flight[flight_radar.callsign] = projection
        radar_data.append(flight_radar)
    return radar_data


# returns the same flight with latitude and longitude changed to reflect the x, y positions in the data
# The intended use of this function is to:
#  1. make a copy of a flight that you got from get_radar_data
#  2. use a kalman filter on that flight and set the x, y columns of the data to the filtered positions
#  3. call set_lat_lon_from_x_y() on that flight to set its latitude and longitude columns according to the filitered x,y positions
# Step 3 is necessary, if you want to plot the data, because plotting is based on the lat/lon coordinates.
def set_lat_lon_from_x_y(flight):
    projection = projection_for_flight[flight.callsign]
    if projection is None:
        print("No projection found for flight %s. You probably did not get this flight from get_radar_data()." % (
            str(flight.flight_id)))

    lons, lats = projection(flight.data["x"], flight.data["y"], inverse=True)
    flight.data["longitude"] = lons
    flight.data["latitute"] = lats
    return flight

DELTA_T = 10  # Time in seconds between observations
STD_DEV = 50  # Standard deviation in meters for independent x(long) and y(lat)
ACCEL = 3     # Acceleration will not exceed 3m/sec**2


def get_flight_radar(f_pos):
    data = get_radar_data()
    flight = data[f_pos]
    return flight


def get_flight_real(f_pos):
    data = get_ground_truth_data()
    flight = data[f_pos]
    return flight

def get_state_k(flight, k):
    current_x = flight.data["longitude"][k]
    current_y = flight.data["latitude"][k]

    if(k == 0):
        return np.array([[current_x],[current_y],[0],[0]])
    else:
        current_x_vel = (flight.data["longitude"][k] - flight.data["longitude"][k-1]) / DELTA_T
        current_y_vel = (flight.data["latitude"][k] - flight.data["latitude"][k - 1]) / DELTA_T
        return np.array([[current_x],[current_y],[current_x_vel],[current_y_vel]])


# Plots via position
def plot_flight(flight_radar, flight_real):
    plt.plot(flight_radar.data["longitude"], flight_radar.data["latitude"])
    plt.plot(flight_radar.data["longitude"], flight_radar.data["latitude"])
    plt.show()


##TODO:Implement

# State matrix setup: |   x   |
#                     |   y   |
#                     | x_hat |
#                     | y_hat |
def get_filtered_positions(flight_radar, flight_real):
    observations = set_lat_lon_from_x_y(flight_radar)

    real_states = []
    radar_states = []

    for i in range(flight_real.data.length):
        current_state = get_state_k(flight_real, i)
        current_obs = get_state_k(flight_radar, i)
        real_states.append(current_state)
        radar_states.append(current_obs)

    transition_matrix = np.array([[1, 0, DELTA_T, 0],
                                  [0, 1, 0, DELTA_T],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]])

    # Compute standard deviation for velocities of real data
    r_x_vel_sum = 0
    r_y_vel_sum = 0
    for i in range(real_states):
        r_x_vel_sum += real_states[i][0][2]
        r_y_vel_sum += real_states[i][0][3]
    r_avg_x_vel = r_x_vel_sum / len(real_states)
    r_avg_y_vel = r_y_vel_sum / len(real_states)

    r_std_x_sum = 0
    r_std_y_sum = 0
    for i in range(real_states):
        r_std_x_sum += real_states[i][0][2] - r_avg_x_vel
        r_std_y_sum += real_states[i][0][3] - r_avg_y_vel
    r_std_x = np.sqrt(r_std_x_sum / len(real_states))
    r_std_y = np.sqrt(r_std_y_sum / len(real_states))

    transition_covariance = np.array([[STD_DEV**2, STD_DEV**2, STD_DEV * r_std_x, STD_DEV * r_std_y],
                                      [STD_DEV**2, STD_DEV**2, STD_DEV * r_std_x, STD_DEV * r_std_y],
                                      [STD_DEV * r_std_x, STD_DEV * r_std_x, r_std_x**2, r_std_x * r_std_y],
                                      [STD_DEV * r_std_y, STD_DEV * r_std_y, r_std_x * r_std_y, r_std_y**2]])

    # Compute standard deviation for velocities of radar data
    o_x_vel_sum = 0
    o_y_vel_sum = 0
    for i in range(radar_states):
        o_x_vel_sum += radar_states[i][0][2]
        o_y_vel_sum += radar_states[i][0][3]
    o_avg_x_vel = o_x_vel_sum / len(radar_states)
    o_avg_y_vel = o_y_vel_sum / len(radar_states)

    o_std_x_sum = 0
    o_std_y_sum = 0
    for i in range(radar_states):
        o_std_x_sum += radar_states[i][0][2] - o_avg_x_vel
        o_std_y_sum += radar_states[i][0][3] - o_avg_y_vel
    o_std_x = np.sqrt(o_std_x_sum / len(radar_states))
    o_std_y = np.sqrt(o_std_y_sum / len(radar_states))

    observation_covariance = np.array([[STD_DEV ** 2, STD_DEV ** 2, STD_DEV * o_std_x, STD_DEV * o_std_y],
                                      [STD_DEV ** 2, STD_DEV ** 2, STD_DEV * o_std_x, STD_DEV * o_std_y],
                                      [STD_DEV * o_std_x, STD_DEV * o_std_x, o_std_x ** 2, o_std_x * o_std_y],
                                      [STD_DEV * o_std_y, STD_DEV * o_std_y, o_std_x * o_std_y, o_std_y ** 2]])

    ##Get covariance with function_base.cov()
    observation_data = np.stack((flight_radar.data["longitude"], flight_radar.data["latitude"]), axis=0)
    observation_matrix = np.array([[1, 0, 0, 0],  # Identity matrix   C and H?
                                   [0, 1, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]])
    # https://youtu.be/Nfrk2UdEOcQ?t=73
    transition_offsets = np.array([[0.5*DELTA_T**2, 0],
                                   [0, 0.5*DELTA_T**2],
                                   [DELTA_T, 0],
                                   [0, DELTA_T]])  # Bu
    observation_offsets = None  # z


    initial_state_mean = np.stack((observation_data[0][0], observation_data[1][0], 0, 0), axis=0)
    initial_state_covariance = observation_covariance


    # Needed for smoothing
    n_dim_state = len(real_states)  # Size of the state space
    n_dim_obs = len(radar_states)  # Size of the observation space


    kf = KalmanFilter(transition_matrices=transition_matrix,
                      transition_covariance=transition_covariance,
                      observation_matrices=observation_matrix,
                      observation_covariance=observation_covariance,
                      initial_state_mean=initial_state_mean,
                      initial_state_covariance=initial_state_covariance)

    # kf = KalmanFilter(transition_matrices=transition_matrix,
    #                   observation_matrices=observation_matrix,
    #                   transition_covariance=transition_covariance,
    #                   observation_covariance=observation_covariance,
    #                   transition_offsets=transition_offsets,
    #                   observation_offsets=observation_offsets,
    #                   initial_state_mean=initial_state_mean,
    #                   initial_state_covariance=initial_state_covariance)
    # random_state: Any = None,
    # em_vars: List[str] = ['transition_covariance', 'observation_covariance',
    #        'initial_state_mean', 'initial_state_covariance'],
    # n_dim_state,
    # n_dim_obs)

    return


if __name__ == "__main__":
    flight_arr_pos = 0  ##For selecting which flight in the data set to consider
    tracked_flight = get_flight_radar(flight_arr_pos)  ##gets a singular flight to track from radar
    real_flight = get_flight_real(flight_arr_pos)  ##gets a singular flight to track from real data

    get_filtered_positions(tracked_flight, real_flight)  ##performs the kalman filter to update predictions
    # plot_flight(tracked_flight, real_flight)                    ##plots the filtered data against the real
