
from traffic.data import samples
from geopy.distance import geodesic
from numpy.random import default_rng
import pyproj

# returns a list of flights with the original GPS data
def get_ground_truth_data():
    names=['liguria', 'pixair_toulouse', 'indiana', 'texas', 'georeal_fyn_island', 'ign_mercantour', 'ign_fontainebleau', 'mecsek_mountains', 'ign_lot_et_garonne', 'inflight_refuelling', 'aircraft_carrier', 'luberon', 'alto_adige', 'franconia', 'danube_valley', 'cevennes', 'oxford_cambridge', 'alpi_italiane', 'rega_zh', 'samu31', 'rega_sg', 'monastir', 'guatemala', 'london_heathrow', 'cardiff', 'sydney', 'brussels_ils', 'ajaccio', 'toulouse', 'noumea', 'london_gatwick', 'perth', 'kota_kinabalu', 'montreal', 'funchal', 'nice', 'munich', 'vancouver', 'lisbon', 'liege_sprimont', 'kiruna', 'bornholm', 'kingston', 'brussels_vor', 'vienna', 'border_control', 'dreamliner_boeing', 'texas_longhorn', 'zero_gravity', 'qantas747', 'turkish_flag', 'airbus_tree', 'easter_rabbit', 'belevingsvlucht', 'anzac_day', 'thankyou', 'vasaloppet']
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
    radar_error = 0.1 # in kilometers
    gt = get_ground_truth_data()
    radar_data = []
    for flight in gt:
        print("flight: %s" % (str(flight)))
        flight_radar = flight.resample("10s")
        for i in range(len(flight_radar.data)):
            point = geodesic(kilometers=rng.normal()*radar_error).destination((flight_radar.data.at[i,"latitude"], flight_radar.data.at[i,"longitude"]), rng.random()*360)
            (flight_radar.data.at[i,"latitude"], flight_radar.data.at[i,"longitude"]) = (point.latitude, point.longitude)
            # print("after: %f, %f" % (flight_radar.data.at[i,"latitude"], flight_radar.data.at[i,"longitude"]))
        projection = pyproj.Proj(proj="lcc", ellps="WGS84", lat_1=flight_radar.data.latitude.min(), lat_2=flight_radar.data.latitude.max(), lat_0=flight_radar.data.latitude.mean(), lon_0=flight_radar.data.longitude.mean())
        flight_radar = flight_radar.compute_xy(projection)
        projection_for_flight[flight_radar.callsign]=projection
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
        print("No projection found for flight %s. You probably did not get this flight from get_radar_data()." % (str(flight.flight_id)))
    
    lons, lats = projection(flight.data["x"], flight.data["y"], inverse=True)
    flight.data["longitude"] = lons
    flight.data["latitute"] = lats
    return flight

