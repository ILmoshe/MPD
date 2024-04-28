import pydeck as pdk
import pandas as pd
from pandas import DataFrame
from geopy.distance import distance

pd.options.display.max_colwidth = 250


departure_points = [
    {"latitude": 33.06035, "longitude": 35.14627, "elevation": 1},
    # {"latitude": 33.06031, "longitude": 35.15406, "elevation": 10},
    # {"latitude": 33.06044, "longitude": 35.14752, "elevation": 10},
]


vertices_to_explore = [
    {"latitude": 33.09357, "longitude": 35.13925, "elevation": 1000},
    {"latitude": 33.094, "longitude": 35.16036, "elevation": 1000},
    {"latitude": 33.10269, "longitude": 35.18285, "elevation": 1000},
    {"latitude": 33.09888, "longitude": 35.18114, "elevation": 1000},
    {"latitude": 33.09616, "longitude": 35.16774, "elevation": 1000},
    {"latitude": 33.10119, "longitude": 35.14199, "elevation": 1000},
    {"latitude": 33.09918, "longitude": 35.13585, "elevation": 1000},
    {"latitude": 33.10001, "longitude": 35.15152, "elevation": 1000},
    {"latitude": 33.09929, "longitude": 35.14118, "elevation": 1000},
    {"latitude": 33.11903, "longitude": 35.15402, "elevation": 1000},
    {"latitude": 33.12069, "longitude": 35.13669, "elevation": 1000},
    {"latitude": 33.1135, "longitude": 35.1269, "elevation": 1000},
    {"latitude": 33.1243, "longitude": 35.19115, "elevation": 1000},
    {"latitude": 33.10925, "longitude": 35.16032, "elevation": 1000},
]


df_departure_points = pd.DataFrame(departure_points)
df_departure_points["name"] = df_departure_points.apply(
    lambda row: f"depatrure_{row.name}", axis=1
)
df_departure_points


df_vertices_to_explore = pd.DataFrame(vertices_to_explore)
df_vertices_to_explore["name"] = df_vertices_to_explore.apply(
    lambda row: f"explore_{row.name}", axis=1
)
df_vertices_to_explore


from shapely import Polygon, MultiPolygon, LineString, intersection

view_state = pdk.ViewState(latitude=33.08913, longitude=35.14388, zoom=9)


def swap(polygon):
    result = [(coord[1], coord[0]) for coord in polygon]
    return result


# swap because geoJSON lon first
nogo_zone1 = swap(
    [
        [33.107063367495, 35.15697128055],
        [33.100520570157, 35.159031217074],
        [33.103324685807, 35.167871777987],
        [33.107279055816, 35.159889523958],
    ]
)
nogo_zone1_polygon_without_buffer: Polygon = Polygon(nogo_zone1).buffer(
    0.001, join_style=2
)
given_from_vertex = [35.15152, 33.10001]
given_to_vertex = [35.16032, 33.10925]
linestring = LineString([given_from_vertex, given_to_vertex])  # long first

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

nogo_zone_layer = pdk.Layer(
    "PolygonLayer",
    [nogo_zone1],
    stroked=False,
    get_polygon="-",
    get_elevation=2000,
    get_fill_color=[255, 0, 0, 100],
    extruded=True,
    wireframe=True,
    auto_highlight=True,
    pickable=True,
)

nogo_zone_layer_buffered = pdk.Layer(
    "PolygonLayer",
    [list(nogo_zone1_polygon_without_buffer.exterior.coords)],
    stroked=False,
    get_polygon="-",
    get_elevation=2000,
    get_fill_color=[245, 222, 179, 100],
    extruded=True,
    wireframe=True,
    auto_highlight=True,
    pickable=True,
)

intersects: LineString = intersection(nogo_zone1_polygon_without_buffer, linestring)


point_intersection_a = [
    intersects.coords.xy[0].tolist()[0],
    intersects.coords.xy[1].tolist()[0],
]
point_intersection_b = [
    intersects.coords.xy[0].tolist()[1],
    intersects.coords.xy[1].tolist()[1],
]


intersection_point_with_buffer = pd.DataFrame(
    [
        [
            *point_intersection_a,
            1000,
            "first",
        ],
        [
            *point_intersection_b,
            1000,
            "second",
        ],
    ],
    columns=["longitude", "latitude", "elevation", "name"],
)


intersection_point_with_buffer_layer = pdk.Layer(
    "PointCloudLayer",
    data=intersection_point_with_buffer,
    get_position=["longitude", "latitude", "elevation"],
    get_color=[0, 0, 255, 160],  # RGBA color, here red
    get_radius=200,  # Radius of each point in meters
    pickable=True,
    point_size=6,
    auto_highlight=True,
)

r0 = pdk.Deck(
    layers=[
        departure_points_layer,
        vertices_to_explore_layer,
        intersection_point_with_buffer_layer,
        nogo_zone_layer,
        nogo_zone_layer_buffered,
    ],
    initial_view_state=view_state,
)


# # in order to make sure that first point is from and second point is to, we need to check distance,
# # and see which is furthest from the to point
if distance(point_intersection_a, given_to_vertex) > distance(
    point_intersection_b, given_to_vertex
):
    point_start_with, point_end_with = point_intersection_a, point_intersection_b
else:
    point_start_with, point_end_with = point_intersection_b, point_intersection_a

r0.show()


from shapely.ops import split
from shapely import GeometryCollection
from shapely.geometry import Point

split_result: GeometryCollection = split(nogo_zone1_polygon_without_buffer, linestring)
first_polygon_after_cut, second_polygon_after_cut = split_result.geoms


def find_index_of_point(polygon, point):
    try:
        return list(polygon.exterior.coords).index((point.x, point.y))
    except ValueError:
        return None


def get_sum_distance(polygon, idx1, idx2, mode: str = "forword") -> int:
    polygon_length = len(polygon)
    factor = 1 if mode == "forword" else -1

    idx2_point = polygon[idx2]
    current_point = polygon[idx1]

    sum_distance = 0
    curr_index = idx1
    while True:
        curr_index = curr_index % polygon_length
        next_index = (curr_index + factor) % polygon_length
        current_point = polygon[curr_index]
        next_point = polygon[next_index]
        sum_distance += distance(current_point, next_point).meters
        curr_index += factor
        if current_point == idx2_point:
            break

    return int(sum_distance)


def calculate_distances(polygon, point1, point2):
    if not isinstance(point1, Point):
        point1 = Point(point1)
    if not isinstance(point2, Point):
        point2 = Point(point2)

    idx1 = find_index_of_point(polygon, point1)
    idx2 = find_index_of_point(polygon, point2)

    if idx1 is None or idx2 is None:
        raise BaseException(
            "One or both points do not exactly match any vertex in the polygon."
        )
    if idx1 == idx2:
        return 0

    polygon_raw = list(polygon.exterior.coords)
    if idx1 > idx2:
        sum_distance = get_sum_distance(polygon_raw, idx1, idx2, "forword")
    else:
        sum_distance = get_sum_distance(polygon_raw, idx1, idx2, "backword")

    return sum_distance


# After spliting # we need to chech which distance is lower so we know we should go that area
distance_from_first_polygon = calculate_distances(
    first_polygon_after_cut, point_start_with, point_end_with
)
distance_from_second_polygon = calculate_distances(
    second_polygon_after_cut, point_start_with, point_end_with
)

# take shortest
distance_with_nogo_zone: int = (
    distance_from_first_polygon
    if distance_from_first_polygon < distance_from_second_polygon
    else distance_from_second_polygon
)


distance_from_first_polygon, distance_from_second_polygon


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


from pydantic import BaseModel


class Drone(BaseModel):
    id: str
    fly_capacity_time_minutes: int  # Total time the drone can fly
    speed_kmh: int  # Speed of the drone
    slack_at_vertex_seconds: int  # Time needed at each vertex
    departure_location: list[float]  # Latitude, Longitude, Elevation


drones = [
    Drone(
        id="first",
        fly_capacity_time_minutes=40,
        speed_kmh=30,
        slack_at_vertex_seconds=50,
        departure_location=[33.06035, 35.14627],
    ),
    Drone(
        id="second",
        fly_capacity_time_minutes=15,
        speed_kmh=120,
        slack_at_vertex_seconds=50,
        departure_location=[33.06035, 35.14627],
    ),
    Drone(
        id="third",
        fly_capacity_time_minutes=30,
        speed_kmh=40,
        slack_at_vertex_seconds=50,
        departure_location=[33.06035, 35.14627],
    ),
    # Add more drones as needed
]


from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


def create_data_model():
    data = {
        "distance_matrix": distance_matrix_meters.values.astype(
            int
        ),  # Ensure distances are integers
        "num_vehicles": len(drones),  # Number of drones
        "depot": 0,  # Assuming the first vertex in your list is the depot
    }
    return data


def calculate_max_travel_seconds(drone: Drone, average_stops: int = 4) -> int:
    total_slack_time_seconds = drone.slack_at_vertex_seconds * average_stops
    effective_flying_time_seconds = int(
        max(0, (drone.fly_capacity_time_minutes * 60) - total_slack_time_seconds)
    )
    return effective_flying_time_seconds


memorize = {}


def model():
    data = create_data_model()
    manager = pywrapcp.RoutingIndexManager(
        len(data["distance_matrix"]), data["num_vehicles"], data["depot"]
    )
    routing = pywrapcp.RoutingModel(manager)

    def create_callback(drone):
        def travel_time_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            if from_node == to_node:
                return 0
            travel_distance = data["distance_matrix"][from_node][to_node]
            travel_time_seconds = int(
                travel_distance / (drone.speed_kmh * 1000 / 3600)
            )  # Convert to seconds
            return travel_time_seconds

        return travel_time_callback

    for idx, drone in enumerate(drones):
        slack_time = drone.slack_at_vertex_seconds

        callback = create_callback(drone)
        callback_index = routing.RegisterTransitCallback(callback)
        routing.SetArcCostEvaluatorOfVehicle(callback_index, idx)

        max_travel_time_seconds = calculate_max_travel_seconds(drone, 5)
        routing.AddDimension(
            callback_index,
            slack_time,  # Slack at each vertex in seconds
            max_travel_time_seconds,  # Maximum travel time in seconds
            False,  # Don't start cumul to zero because we have slack
            f"Time_{idx}",
        )
        time_dimension = routing.GetDimensionOrDie(f"Time_{idx}")
        time_dimension.SetGlobalSpanCostCoefficient(1)

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
    print("Solution:")
    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        plan_output = f"Route for vehicle {vehicle_id}: "
        while not routing.IsEnd(index):
            plan_output += f"{manager.IndexToNode(index)} -> "
            index = solution.Value(routing.NextVar(index))
        plan_output += f"{manager.IndexToNode(index)}"
        print(plan_output)


data, manager, routing, solution = model()


def get_colors():
    a = 255
    green = [144, 238, 144, a]
    aqua = [0, 255, 255, a]
    wheat = [245, 222, 179, a]
    magenta = [255, 0, 255, a]
    white = [255, 255, 255, a]
    gold = [255, 215, 0, a]
    return [green, aqua, wheat, magenta, white, gold]


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


def create_route_layers(routes, all_nodes, drones):
    layers = []
    for drone, (route, color) in zip(drones, zip(routes, get_colors())):
        route_data = []
        for i in range(len(route) - 1):
            start_node = all_nodes.iloc[route[i]]
            end_node = all_nodes.iloc[route[i + 1]]
            segment = {
                "source": [start_node["longitude"], start_node["latitude"], 1000],
                "target": [end_node["longitude"], end_node["latitude"], 1000],
                "drone_id": drone.id,  # Using the new id field for labeling
            }
            route_data.append(segment)

        # Create a LineLayer for the route
        line_layer = pdk.Layer(
            "LineLayer",
            pd.DataFrame(route_data),
            get_source_position="source",
            get_target_position="target",
            get_color=color,
            get_width=5,
            pickable=True,
            auto_highlight=True,
            # get_elevation=1000,
            # get_fill_color=[255, 0, 0, 100],
            extruded=True,
            wireframe=True,
        )

        # Optional: Create a TextLayer to label parts of the route with the drone ID
        text_layer = pdk.Layer(
            "TextLayer",
            pd.DataFrame(route_data),
            get_position="target",  # Position text at the target of each segment
            get_text="drone_id",  # Field containing the text to display
            get_size=16,
            get_color=color,
            get_angle=0,
            getTextAnchor='"middle"',
            get_alignment_baseline='"center"',
        )

        layers.append(line_layer)
        layers.append(text_layer)

    return layers


if solution is None:
    raise BaseException("Couldnt find a solution")

routes = get_routes(data, manager, routing, solution)
all_nodes = pd.concat([df_departure_points, df_vertices_to_explore], ignore_index=True)
route_layers = create_route_layers(routes, all_nodes, drones)


r1 = pdk.Deck(
    layers=[
        vertices_to_explore_layer,
        departure_points_layer,
        *route_layers,
    ],
    initial_view_state=view_state,
)

r1.to_html("mission_plan_result.html")
r1.show()
