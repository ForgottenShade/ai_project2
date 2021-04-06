from traffic.data import samples
from geopy.distance import geodesic
from numpy.random import default_rng
import numpy as np
import pyproj

from pykalman import KalmanFilter
import matplotlib.pyplot as plt


# returns a list of flights with the original GPS data
def get_ground_truth_data():
    names = ['liguria']
    # names = ['liguria', 'pixair_toulouse', 'indiana', 'texas', 'georeal_fyn_island', 'ign_mercantour',
    #          'ign_fontainebleau', 'mecsek_mountains', 'ign_lot_et_garonne', 'inflight_refuelling', 'aircraft_carrier',
    #          'luberon', 'alto_adige', 'franconia', 'danube_valley', 'cevennes', 'oxford_cambridge', 'alpi_italiane',
    #          'rega_zh', 'samu31', 'rega_sg', 'monastir', 'guatemala', 'london_heathrow', 'cardiff', 'sydney',
    #          'brussels_ils', 'ajaccio', 'toulouse', 'noumea', 'london_gatwick', 'perth', 'kota_kinabalu', 'montreal',
    #          'funchal', 'nice', 'munich', 'vancouver', 'lisbon', 'liege_sprimont', 'kiruna', 'bornholm', 'kingston',
    #          'brussels_vor', 'vienna', 'border_control', 'dreamliner_boeing', 'texas_longhorn', 'zero_gravity',
    #          'qantas747', 'turkish_flag', 'airbus_tree', 'easter_rabbit', 'belevingsvlucht', 'anzac_day', 'thankyou',
    #          'vasaloppet']
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
    # radar_error changed to 0.2 and 0.3 to increase noise generated for some experiments
    radar_error = 0.3  # in kilometers
    radar_altitude_error = 330  # in feet ( ~ 100 meters)
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
            flight_radar.data.at[i, "altitude"] += rng.normal() * radar_altitude_error
            # print("after: %f, %f" % (flight_radar.data.at[i,"latitude"], flight_radar.data.at[i,"longitude"]))
        projection = pyproj.Proj(proj="lcc", ellps="WGS84", lat_1=flight_radar.data.latitude.min(),
                                 lat_2=flight_radar.data.latitude.max(), lat_0=flight_radar.data.latitude.mean(),
                                 lon_0=flight_radar.data.longitude.mean())
        flight_radar = flight_radar.compute_xy(projection)
        flightid = flight_radar.callsign + str(flight_radar.start)
        if flightid in projection_for_flight:
            print("ERROR: duplicate flight ids: %s" % (flightid))
        projection_for_flight[flight_radar.callsign + str(flight_radar.start)] = projection
        radar_data.append(flight_radar)
    return radar_data


# returns the same flight with latitude and longitude changed to reflect the x, y positions in the data
# The intended use of this function is to:
#  1. make a copy of a flight that you got from get_radar_data
#  2. use a kalman filter on that flight and set the x, y columns of the data to the filtered positions
#  3. call set_lat_lon_from_x_y() on that flight to set its latitude and longitude columns according to the filitered x,y positions
# Step 3 is necessary, if you want to plot the data, because plotting is based on the lat/lon coordinates.
def set_lat_lon_from_x_y(flight):
    flightid = flight.callsign + str(flight.start)
    projection = projection_for_flight[flightid]
    if projection is None:
        print("No projection found for flight %s. You probably did not get this flight from get_radar_data()." % (
            flightid))

    lons, lats = projection(flight.data["x"], flight.data["y"], inverse=True)
    flight.data["longitude"] = lons
    flight.data["latitude"] = lats
    return flight


DELTA_T = 10  # Time in seconds between observations
DELTA_T_COV = (0.25 * DELTA_T) ** 4
DELTA_T_COV_Mix = (0.5 * DELTA_T) ** 3
STD_DEV = 150  # Standard deviation in meters for independent x(long) and y(lat)
ACCEL = 3 # Acceleration will not exceed 3m/sec**2
LAT_M = 111131.745  # constants to convert degree of lat to m
LON_M = 78846.806  # lon to m


def get_flight_radar(f_pos):
    data = get_radar_data()
    flight = data[f_pos]
    return flight


def get_flight_real(f_pos):
    data = get_ground_truth_data()
    flight = data[f_pos]
    return flight


def get_state_k(flight, k):
    current_x = flight.data["longitude"][k] * LON_M
    current_y = flight.data["latitude"][k] * LAT_M

    if k == 0:
        return np.array([[current_x], [current_y], [0], [0]])
    else:
        current_x_vel = (flight.data["longitude"][k] * LON_M - flight.data["longitude"][k - 1] * LON_M) / DELTA_T
        current_y_vel = (flight.data["latitude"][k] * LAT_M - flight.data["latitude"][k - 1] * LAT_M) / DELTA_T
        return np.array([[current_x], [current_y], [current_x_vel], [current_y_vel]])


def get_measurements_k(flight, k):
    current_x = flight.data["longitude"][k] * LON_M
    current_y = flight.data["latitude"][k] * LAT_M

    if k == 0:
        return np.array([current_x, current_y, 0, 0])
    else:
        current_x_vel = (flight.data["longitude"][k] * LON_M - flight.data["longitude"][k - 1] * LON_M) / DELTA_T
        current_y_vel = (flight.data["latitude"][k] * LAT_M - flight.data["latitude"][k - 1] * LAT_M) / DELTA_T
        return np.array([current_x, current_y, current_x_vel, current_y_vel])


# Plots via position
def plot_flight(flight_radar, flight_real):
    # plt.plot(flight_radar[0], flight_radar[1])
    plt.plot(flight_radar.data["longitude"], flight_radar.data["latitude"], 'c', label="RadarData")
    plt.plot(flight_real.data["longitude"], flight_real.data["latitude"], 'k', label="RealData")
    plt.legend(loc="upper left")
    plt.show()


def plot_all(smoothed_x, smoothed_y, filtered_x, filtered_y, flight_real, flight_radar, name=""):
    plt.plot(smoothed_x, smoothed_y, 'r', label="Smoothed")
    plt.plot(filtered_x, filtered_y, 'b', label="Filtered")
    plt.plot(flight_radar.data["longitude"] * LON_M, flight_radar.data["latitude"] * LAT_M, 'c', label="RadarData")
    plt.plot(flight_real.data["longitude"] * LON_M, flight_real.data["latitude"] * LAT_M, 'k', label="RealData")
    if name != "":
        plt.title(name)
    plt.legend(loc="upper left")
    plt.show()


# TODO:Implement

# State matrix setup: |   x   |
#                     |   y   |
#                     | x_hat |
#                     | y_hat |

# Scalars are there do be able to reuse the data collected at the beginning to get more accurate comparisons,
# since noise is generated at random.
def get_filtered_positions(flight_radar, flight_real, flight, t_cov_scalar=1, o_cov_scalar=1):
    observations = set_lat_lon_from_x_y(flight_radar)

    real_states = []
    radar_states = []
    measurements = []

    for i in range(flight_real.data['longitude'].size):
        current_state = get_state_k(flight_real, i)
        real_states.append(current_state)

    for j in range(flight_radar.data['longitude'].size):
        current_obs = get_state_k(flight_radar, j)
        current_measure = get_measurements_k(flight_radar, j)
        radar_states.append(current_obs)
        measurements.append(current_measure)

    transition_matrix = np.array([[1, 0, DELTA_T, 0],
                                  [0, 1, 0, DELTA_T],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]])

    # Compute standard deviation for velocities of real data
    r_x_vel_sum = 0
    r_y_vel_sum = 0
    for i in range(len(real_states)):
        r_x_vel_sum += real_states[i][2][0]
        r_y_vel_sum += real_states[i][3][0]
    r_avg_x_vel = r_x_vel_sum / len(real_states)
    r_avg_y_vel = r_y_vel_sum / len(real_states)

    r_std_x_sum = 0
    r_std_y_sum = 0
    for i in range(len(real_states)):
        r_std_x_sum += (real_states[i][2][0] - r_avg_x_vel) ** 2
        r_std_y_sum += (real_states[i][3][0] - r_avg_y_vel) ** 2
    r_std_x = np.sqrt(r_std_x_sum / len(real_states))
    r_std_y = np.sqrt(r_std_y_sum / len(real_states))

    transition_covariance = np.array(
        [[DELTA_T_COV * STD_DEV ** 2, DELTA_T_COV * STD_DEV ** 2, DELTA_T_COV_Mix * STD_DEV * r_std_x, 0],
         [DELTA_T_COV * STD_DEV ** 2, DELTA_T_COV * STD_DEV ** 2, 0, DELTA_T_COV_Mix * STD_DEV * r_std_y],
         [DELTA_T_COV_Mix * STD_DEV * r_std_x, 0, DELTA_T ** 2 * r_std_x ** 2, DELTA_T ** 2 * r_std_x * r_std_y],
         [0, DELTA_T_COV_Mix * STD_DEV * r_std_y, DELTA_T ** 2 * r_std_x * r_std_y, DELTA_T ** 2 * r_std_y ** 2]]) \
        * t_cov_scalar

    # Compute standard deviation for velocities of radar data
    o_x_vel_sum = 0
    o_y_vel_sum = 0
    for i in range(len(radar_states)):
        o_x_vel_sum += radar_states[i][2][0]
        o_y_vel_sum += radar_states[i][3][0]
    o_avg_x_vel = o_x_vel_sum / len(radar_states)
    o_avg_y_vel = o_y_vel_sum / len(radar_states)

    o_std_x_sum = 0
    o_std_y_sum = 0
    for i in range(len(radar_states)):
        o_std_x_sum += (radar_states[i][2][0] - o_avg_x_vel) ** 2
        o_std_y_sum += (radar_states[i][3][0] - o_avg_y_vel) ** 2
    o_std_x = np.sqrt(o_std_x_sum / len(radar_states))
    o_std_y = np.sqrt(o_std_y_sum / len(radar_states))


    observation_covariance = np.array(
        [[DELTA_T_COV * STD_DEV ** 2, DELTA_T_COV * STD_DEV ** 2, DELTA_T_COV_Mix * STD_DEV * o_std_x, 0],
         [DELTA_T_COV * STD_DEV ** 2, DELTA_T_COV * STD_DEV ** 2, 0, DELTA_T_COV_Mix * STD_DEV * o_std_y],
         [DELTA_T_COV_Mix * STD_DEV * o_std_x, 0, DELTA_T ** 2 * o_std_x ** 2, DELTA_T ** 2 * o_std_x * o_std_y],
         [0, DELTA_T_COV_Mix * STD_DEV * o_std_y, DELTA_T ** 2 * o_std_x * o_std_y, DELTA_T ** 2 * o_std_y ** 2]])\
        * o_cov_scalar

    # Get covariance with function_base.cov()
    # observation_data = np.stack((flight_radar.data["longitude"] * LON_M, flight_radar.data["latitude"] * LAT_M), axis=0)
    observation_matrix = np.array([[1, 0, 0, 0],  # Identity matrix   C and H?
                                   [0, 1, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]])


    # Some ideas on how the transition_offsets matrix should look like. It did not improve the filter so
    # it was not used.

    # transition_offsets = np.dot(np.array([[0, 0, 0.5 * DELTA_T ** 2, 0],
    #                                       [0, 0, 0, 0.5 * DELTA_T ** 2],
    #                                       [0, 0, DELTA_T, 0],  # purely guesswork here
    #                                       [0, 0, 0, DELTA_T]]), np.array(
    #     [0, 0, ACCEL, ACCEL]))  # Bu. Needs to be multiplied with a [2x1] matrix before usable?

    # transition_offsets = np.dot(np.array([[0.5 * DELTA_T ** 2, 0, 0, 0],
    #                                       [0, 0.5 * DELTA_T ** 2, 0, 0],
    #                                       [DELTA_T, 0, 0, 0],  # purely guesswork here
    #                                       [0, DELTA_T, 0, 0]]), np.array(
    #     [1, 1, ACCEL, ACCEL]))  # Bu. Needs to be multiplied with a [2x1] matrix before usable?
    # Turns out NOPE... No clue what shape this needs to be...

    observation_offsets = None  # z

    initial_state_mean = np.stack((measurements[0][0], measurements[0][1], measurements[0][2], measurements[0][3]),
                                  axis=0)
    initial_state_covariance = observation_covariance

    kf = KalmanFilter(transition_matrices=transition_matrix,
                      observation_matrices=observation_matrix,
                      observation_covariance=observation_covariance,
                      initial_state_mean=initial_state_mean,
                      initial_state_covariance=initial_state_covariance,
                      transition_covariance=transition_covariance,
                      n_dim_obs=4,
                      n_dim_state=4)

    filtered = kf.filter(measurements)
    # Extracting the positions from the filtered data.
    filtered_xs = filtered[0][:, 0]
    filtered_ys = filtered[0][:, 1]
    smoothed = kf.smooth(measurements)
    # Extracting the positions from the smoothed data.
    smoothed_xs = smoothed[0][:, 0]
    smoothed_ys = smoothed[0][:, 1]

    # TODO: Measure the error of the filtered positions. Compute the mean and max distance to the real data at delta_t
    # Currently measuring the distance between the real state points for the first flight only
    # The first flight has equal rows of observations data and real data.
    # The timestamps of the observation and real data match completely.
    if flight == 0:
        f_max_dist = 0
        s_max_dist = 0
        o_max_dist = 0
        f_sum = 0
        s_sum = 0
        o_sum = 0
        n = len(filtered_xs)
        for i in range(n):
            f_dist = np.sqrt((real_states[i][0][0] - filtered_xs[i]) ** 2 + (real_states[i][1][0] - filtered_ys[i]) ** 2)
            s_dist = np.sqrt((real_states[i][0][0] - smoothed_xs[i]) ** 2 + (real_states[i][1][0] - smoothed_ys[i]) ** 2)
            o_dist = np.sqrt((real_states[i][0][0] - measurements[i][0]) ** 2 + (real_states[i][1][0] - measurements[i][1]) ** 2)
            if f_dist > f_max_dist:
                f_max_dist = f_dist
            if s_dist > s_max_dist:
                s_max_dist = s_dist
            if o_dist > o_max_dist:
                o_max_dist = o_dist

            o_sum += o_dist
            f_sum += f_dist
            s_sum += s_dist

        print("Filtered data. - Max distance: ",  f_max_dist, "    Mean: ", f_sum/n)
        print("Smoothed data: - Max distance: ", s_max_dist, "    Mean: ", s_sum/n)
        print("Original data: - Max distance: ", o_max_dist, "    Mean : ", o_sum/n)

    return smoothed_xs, smoothed_ys, filtered_xs, filtered_ys


if __name__ == "__main__":
    radar_data = get_radar_data()
    real_data = get_ground_truth_data()
    for i in range(len(real_data)):
        tracked_flight = radar_data[i]  # gets a singular flight to track from radar

        real_flight = real_data[i]  # gets a singular flight to track from real data

        ## Used to get some error comparisons on the first flight with some matrices scaled
        # if i == 0:  # Just to show that the conversion doesn't affect the data
        #     # plots the real data and the radar data.
        #     #plot_flight(tracked_flight, real_flight)
        #
        #     # Applying Kalman Filter with different scalars for comparisons.
        #     smooth_kf_x, smooth_kf_y, filtered_kf_x, filtered_kf_y = \
        #         get_filtered_positions(tracked_flight, real_flight, i, 0.5, 1)
        #     plot_all(smooth_kf_x, smooth_kf_y, filtered_kf_x, filtered_kf_y, real_flight, tracked_flight,
        #              "0.5 * trans_covar_matrix, normal obs_covar_matrix")
        #
        #     smooth_kf_x, smooth_kf_y, filtered_kf_x, filtered_kf_y = \
        #         get_filtered_positions(tracked_flight, real_flight, i, 1, 0.5)
        #     plot_all(smooth_kf_x, smooth_kf_y, filtered_kf_x, filtered_kf_y, real_flight, tracked_flight,
        #              "normal trans_covar_matrix, 0.5 * obs_covar_matrix")
        #
        #
        #     smooth_kf_x, smooth_kf_y, filtered_kf_x, filtered_kf_y = \
        #         get_filtered_positions(tracked_flight, real_flight, i, 2, 1)
        #     plot_all(smooth_kf_x, smooth_kf_y, filtered_kf_x, filtered_kf_y, real_flight, tracked_flight,
        #              "2 * trans_covar_matrix, normal obs_covar_matrix")
        #
        #
        #     smooth_kf_x, smooth_kf_y, filtered_kf_x, filtered_kf_y = \
        #         get_filtered_positions(tracked_flight, real_flight, i, 1, 2)
        #     plot_all(smooth_kf_x, smooth_kf_y, filtered_kf_x, filtered_kf_y, real_flight, tracked_flight,
        #              "normal trans_covar_matrix, 2 * obs_covar_matrix")


        smooth_kf_x, smooth_kf_y, filtered_kf_x, filtered_kf_y = \
            get_filtered_positions(tracked_flight, real_flight, i)  # performs the kalman filter to update predictions

        # print(real_flight.data)
        # print(tracked_flight.data)

        # plots the filtered and smoothed data against the real and radar data
        plot_all(smooth_kf_x, smooth_kf_y, filtered_kf_x, filtered_kf_y, real_flight, tracked_flight)
        i += 1



