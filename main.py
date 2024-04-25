# %%
import pydeck as pdk
import pandas as pd
from pandas import DataFrame
from geopy.distance import distance

pd.options.display.max_colwidth = 250


# %%
def calculat_distances_warper(df: DataFrame):
    def calculat_distances(row):
        distances = {}
        for _, other_row in df.iterrows():
            if row["name"] != other_row["name"]:
                actual_distance: float = distance(
                    (row["latitude"], row["longitude"]),
                    (other_row["latitude"], other_row["longitude"]),
                ).meters
                distances[other_row["name"]] = actual_distance
        return distances

    return calculat_distances


# %%
departure_points = [
    {"latitude": 33.06035, "longitude": 35.14627, "elevation": 10},
    # {"latitude": 33.06031, "longitude": 35.15406, "elevation": 10},
    # {"latitude": 33.06044, "longitude": 35.14752, "elevation": 10},
]


vertices_to_explore = [
    {"latitude": 33.09357, "longitude": 35.13925, "elevation": 10},
    {"latitude": 33.094, "longitude": 35.16036, "elevation": 10},
    {"latitude": 33.10269, "longitude": 35.18285, "elevation": 10},
    {"latitude": 33.09888, "longitude": 35.18114, "elevation": 10},
    {"latitude": 33.09616, "longitude": 35.16774, "elevation": 10},
    {"latitude": 33.10119, "longitude": 35.14199, "elevation": 10},
    {"latitude": 33.09918, "longitude": 35.13585, "elevation": 10},
    {"latitude": 33.10001, "longitude": 35.15152, "elevation": 10},
    {"latitude": 33.09929, "longitude": 35.14118, "elevation": 10},
    {"latitude": 33.11903, "longitude": 35.15402, "elevation": 10},
    {"latitude": 33.12069, "longitude": 35.13669, "elevation": 10},
    {"latitude": 33.1135, "longitude": 35.1269, "elevation": 10},
    {"latitude": 33.1243, "longitude": 35.19115, "elevation": 10},
]


# %%
df_departure_points = pd.DataFrame(departure_points)
df_departure_points["name"] = df_departure_points.apply(
    lambda row: f"depatrure_{row.name}", axis=1
)
df_departure_points

# %%
df_vertices_to_explore = pd.DataFrame(vertices_to_explore)
df_vertices_to_explore["name"] = df_vertices_to_explore.apply(
    lambda row: f"explore_{row.name}", axis=1
)
df_vertices_to_explore


# %%
def create_distance_matrix(df: DataFrame):
    distance_matrix = pd.DataFrame(index=df["name"], columns=df["name"])
    for _, row1 in df.iterrows():
        for _, row2 in df.iterrows():
            dist: float = distance(
                (row1["latitude"], row1["longitude"]),
                (row2["latitude"], row2["longitude"]),
            ).meters
            distance_matrix.loc[row1["name"], row2["name"]] = int(dist)
    return distance_matrix


distance_matrix_meters = create_distance_matrix(
    pd.concat([df_departure_points, df_vertices_to_explore], ignore_index=True)
)
distance_matrix_meters

# %%
view_state = pdk.ViewState(latitude=33.08913, longitude=35.14388, zoom=9)


departure_points_layer = pdk.Layer(
    "PointCloudLayer",
    data=df_departure_points,
    get_position=["longitude", "latitude", "elevation"],
    get_color=[255, 0, 0, 160],  # RGBA color, here red
    get_radius=100,  # Radius of each point in meters
    pickable=True,
    point_size=10,
    auto_highlight=True,
)

vertices_to_explore_layer = pdk.Layer(
    "PointCloudLayer",
    data=df_vertices_to_explore,
    get_position=["longitude", "latitude", "elevation"],
    get_color=[0, 255, 0, 160],  # RGBA color, here red
    get_radius=100,  # Radius of each point in meters
    pickable=True,
    point_size=8,
    auto_highlight=True,
)


# %%
from pydantic import BaseModel


class Drone(BaseModel):
    fly_capacity_time_minutes: int  # How much time the drone can fly
    speed_kmh: int  # speed of the drone, NOTE: for now drone speed dont change
    slack_at_vertex: (
        int  # How much the drone need to stay in the vertice he needs to explore
    )
    drone_departure_location: list[float]  # lat, lon, ele


# We can see that we have 3 deparutes, so we have 3 drones, drone per deparute
departure_locations_lat = df_departure_points["latitude"].to_list()
departure_locations_lon = df_departure_points["longitude"].to_list()
departure_locations_ele = df_departure_points["elevation"].to_list()
ziped_departure_locations = list(
    zip(departure_locations_lat, departure_locations_lon, departure_locations_ele)
)

drone_a = Drone(
    **{
        "fly_capacity_time_minutes": 60,
        "speed_kmh": 50,
        "slack_at_vertex": 50,
        "drone_departure_location": ziped_departure_locations[0],
    }
)
drone_b = Drone(
    **{
        "fly_capacity_time_minutes": 60,
        "speed_kmh": 50,
        "slack_at_vertex": 50,
        "drone_departure_location": ziped_departure_locations[0],
    }
)
drone_c = Drone(
    **{
        "fly_capacity_time_minutes": 60,
        "speed_kmh": 50,
        "slack_at_vertex": 50,
        "drone_departure_location": ziped_departure_locations[0],
    }
)
drone_d = Drone(
    **{
        "fly_capacity_time_minutes": 60,
        "speed_kmh": 50,
        "slack_at_vertex": 50,
        "drone_departure_location": ziped_departure_locations[0],
    }
)


# %%
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


def create_data_model():
    data = {
        "distance_matrix": distance_matrix_meters.values,  # Use the distance matrix you calculated
        "num_vehicles": 3,  # Number of drones
        "depot": 0,  # Assuming the first vertex in your list is the depot
    }
    return data


# %%
def model():
    data = create_data_model()
    manager = pywrapcp.RoutingIndexManager(
        len(data["distance_matrix"]), data["num_vehicles"], data["depot"]
    )
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(data["distance_matrix"][from_node][to_node])

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # cost defenition
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Distance constraint for each drone.
    for vehicle_id in range(data["num_vehicles"]):
        dimesnsion_name = f"distance_{vehicle_id}"
        routing.AddDimension(
            transit_callback_index,
            0,  # no slack, why no slack??
            18800,  # maximum travel distance for the drone, this needs calculation based on speed and time
            True,  # start cumul to zero
            dimesnsion_name,
        )
        distance_dimension = routing.GetDimensionOrDie(dimesnsion_name)
        distance_dimension.SetGlobalSpanCostCoefficient(1)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.seconds = 3
    search_parameters.log_search = True

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)
    if solution:
        print_solution(data, manager, routing, solution)

    return (data, manager, routing, solution)


def print_solution(data, manager, routing, solution):
    """Prints solution on console."""
    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        plan_output = f"Route for vehicle {vehicle_id}:\n"
        route_distance = 0
        while not routing.IsEnd(index):
            plan_output += f"{manager.IndexToNode(index)} -> "
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id
            )
        plan_output += f"{manager.IndexToNode(index)}\n"
        plan_output += f"Distance of the route: {route_distance}m\n"
        print(plan_output)


data, manager, routing, solution = model()


# %%
import random


def random_bright_rgba_color():
    min_threshold = 128
    r = random.randint(min_threshold, 255)
    g = random.randint(min_threshold, 255)
    b = random.randint(min_threshold, 255)
    a = random.randint(0, 255)
    return [r, g, b, a]


def get_routes(data, manager, routing, solution):
    """Extracts the routes from the solution and returns them including the depot as start and end point."""
    routes = []
    for vehicle_id in range(data["num_vehicles"]):
        route = []
        index = routing.Start(vehicle_id)
        route.append(manager.IndexToNode(index))
        while not routing.IsEnd(index):
            index = solution.Value(routing.NextVar(index))
            if not routing.IsEnd(index):
                route.append(manager.IndexToNode(index))
        route.append(route[0])
        routes.append(route)
    return routes


def create_route_layers(routes, all_nodes):
    layers = []
    for route in routes:
        route_data = []
        for i in range(len(route) - 1):
            start_node = all_nodes.iloc[route[i]]
            end_node = all_nodes.iloc[route[i + 1]]
            segment = {
                "source": [start_node["longitude"], start_node["latitude"]],
                "target": [end_node["longitude"], end_node["latitude"]],
            }
            route_data.append(segment)

        layer = pdk.Layer(
            "LineLayer",
            pd.DataFrame(route_data),
            get_source_position="source",
            get_target_position="target",
            get_color=random_bright_rgba_color(),
            get_width=5,
            pickable=True,
            auto_highlight=True,
        )
        layers.append(layer)

    return layers


if solution is None:
    raise BaseException("Couldnt find a solution")

routes = get_routes(data, manager, routing, solution)
all_nodes = pd.concat([df_departure_points, df_vertices_to_explore], ignore_index=True)
route_layers = create_route_layers(routes, all_nodes)

r1 = pdk.Deck(
    layers=[vertices_to_explore_layer, departure_points_layer, *route_layers],
    initial_view_state=view_state,
)

r1.to_html("mission_plan_result.html")
