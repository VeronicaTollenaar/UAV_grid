import os
import zipfile
from xml.etree import ElementTree as ET
import geopandas as gpd
import numpy as np
import math
from shapely.geometry import LineString, Point, Polygon, MultiPolygon, box
from shapely.affinity import rotate
import matplotlib.pyplot as plt
from pyproj import Transformer
import shutil
# set working directory
path = os.path.dirname(os.path.abspath(__file__))
os.chdir(path)

#%%
kmz_name = 'patch4_test.kmz'
new_kmz_name_start = 'PEA_Utsteinen'
attempt=0
kmz_path = '../drone_survey/'+kmz_name
extract_path = '../drone_survey/'

with zipfile.ZipFile(kmz_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)


#%%
# calculate new waypoints


# Load the shapefile
shapefile = gpd.read_file("../drone_survey/drone_PEA_Utsteinen_3031.shp")
#extent = shapefile.total_bounds  # [minx, miny, maxx, maxy]
polygon = shapefile.unary_union

# Camera and flight parameters
altitude = 80  # meters
fov_degrees = 82
image_width_px = 8064
image_length_px = 6048
lateral_overlap = 0.5 #0.4 is minimum, 0.70 recommended # side
longitudinal_overlap = 0.7 #0.6 is minimum, 0.90 recommended # front
time_between_shots = 5 # seconds
flight_length = 20 # 30 is max # minutes

# Calculate GSD and spacings
fov_radians = fov_degrees * (math.pi / 180)
gsd = (2 * altitude * math.tan(fov_radians / 2)) / image_width_px
longitudinal_spacing = gsd * image_length_px * (1 - longitudinal_overlap)
lateral_spacing = gsd * image_width_px * (1 - lateral_overlap)
print(f'Ground resolution is: {gsd*100} cm')

# Calculate flight speed
flight_speed = np.round(longitudinal_spacing / time_between_shots,2)  # meters per second

def create_transect_lines(polygon, spacing, angle):
    """Create transect lines across the polygon at a specified spacing and angle."""
    # Calculate the bounds of the polygon and generate an extended box
    minx, miny, maxx, maxy = polygon.bounds
    width = maxx - minx
    height = maxy - miny
    diagonal = np.sqrt(width**2 + height**2)  # Max distance any line might need to cover
    centerx, centery = (minx + maxx) / 2, (miny + maxy) / 2

    # Generate lines from center extended by diagonal in both directions
    lines = []
    num_lines = int(diagonal / spacing)  # Number of lines based on spacing and diagonal length
    for i in range(-num_lines, num_lines + 1):
        # Start with a horizontal line at the center and displace vertically by spacing
        line = LineString([(centerx - diagonal, centery + i * spacing), (centerx + diagonal, centery + i * spacing)])
        # Rotate line around the center of the bounding box
        rotated_line = rotate(line, angle, origin=(centerx, centery), use_radians=False)
        # Only add the line if it intersects with the polygon to avoid excessive calculations
        if polygon.intersects(rotated_line):
            lines.append(rotated_line)
    return lines

# def create_transect_lines(polygon, spacing, angle):
#     """Create transect lines across the polygon at a specified spacing and angle."""
#     minx, miny, maxx, maxy = polygon.bounds
#     # Create a bounding box slightly larger than the polygon
#     extended_box = box(minx, miny, maxx, maxy).buffer(spacing)
#     # Start points are along the left side of the bounding box
#     startx, starty = extended_box.bounds[0], extended_box.bounds[1]
#     endx, endy = extended_box.bounds[2], extended_box.bounds[3]

#     # Generate points along the left and right bounds at the given spacing
#     y_starts = np.arange(starty, endy, spacing)
#     y_ends = np.arange(starty, endy, spacing)

#     lines = []
#     for ys, ye in zip(y_starts, y_ends):
#         line = LineString([(startx, ys), (endx, ye)])
#         # Rotate the line around the center of the bounding box
#         line = rotate(line, angle, origin='centroid', use_radians=False)
#         lines.append(line)
    
#     return lines

def calculate_intersections(polygon, lines):
    """Calculate the intersections of lines with the polygon, returning points."""
    points = []
    for line in lines:
        intersection = polygon.intersection(line)
        if not intersection.is_empty:
            if isinstance(intersection, LineString):
                points.extend([Point(p) for p in intersection.coords])
            elif isinstance(intersection, (Polygon, MultiPolygon)):
                # Handle complex intersections
                for part in intersection.geoms:
                    if isinstance(part, LineString):
                        points.extend([Point(p) for p in part.coords])
            elif isinstance(intersection, Point):
                points.append(intersection)
    return points

def plot_polygon_and_lines(polygon, lines, points):
    """Plot a polygon and transect lines using matplotlib."""
    fig, ax = plt.subplots()
    # Plot the polygon
    x, y = polygon.exterior.xy
    ax.fill(x, y, alpha=0.5, fc='green', label='Polygon Area')

    # Plot each line
    for line in lines:
        x, y = line.xy
        ax.plot(x, y, color="red", linewidth=1, label='Transect Line' if 'Transect Line' not in ax.get_legend_handles_labels()[1] else "")
    
    # Plot waypoints
    for point in points:
        ax.plot(point.x, point.y, 'bo', label='Waypoint' if 'Waypoint' not in ax.get_legend_handles_labels()[1] else "")

    ax.set_title("Polygon with Transect Lines")
    ax.legend()
    plt.show()
    
# Create transect lines
angle = 110#100
transect_lines = create_transect_lines(polygon, spacing=lateral_spacing, angle=angle)  # 30m spacing, 45-degree angle

# Calculate intersections
waypoints = calculate_intersections(polygon, transect_lines)

# Plot the polygon and the transect lines
plot_polygon_and_lines(polygon, transect_lines, waypoints)

# Save or use waypoints as needed
print(f"Calculated flight speed: {flight_speed:.2f} m/s")
print(f"Number of waypoints: {len(waypoints)}")

#%%
def rotate_point(x, y, angle, polygon):
    origin = (polygon.centroid.x,polygon.centroid.y)
    """Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = x, y

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def sort_waypoints(waypoints, angle_degrees):
    # Convert angle to radians
    angle_radians = -math.radians(angle_degrees)  # Negative to rotate back to axis alignment

    # Rotate waypoints
    rotated_waypoints = [Point(*rotate_point(wp.x, wp.y, angle_radians,polygon)) for wp in waypoints]

    # Sort rotated waypoints by y then x
    rotated_waypoints.sort(key=lambda p: (round(p.y, 1), p.x))

    # Group by y-coordinate, rounded to the nearest unit to stabilize grouping
    from collections import defaultdict
    rows = defaultdict(list)
    for wp in rotated_waypoints:
        rows[round(wp.y, 1)].append(wp)

    sorted_keys = sorted(rows.keys(), reverse=True)  # Start from the top

    # Alternate the order of waypoints in each row
    sorted_waypoints = []
    toggle = True  # Start with left to right
    for key in sorted_keys:
        row = sorted(rows[key], key=lambda p: p.x, reverse=not toggle)  # Toggle the direction
        sorted_waypoints.extend(row)
        toggle = not toggle  # Flip direction for next row

    # Rotate waypoints back to original orientation for output
    sorted_original_waypoints = [Point(rotate_point(wp.x, wp.y, -angle_radians, polygon)) for wp in sorted_waypoints]

    return sorted_original_waypoints

def calculate_path_distance(waypoints):
    """Calculate the total distance of the flight path."""
    path_distance = sum(waypoints[i].distance(waypoints[i + 1]) for i in range(len(waypoints) - 1))
    return path_distance / 1000

def plot_path(waypoints):
    """Plot the flight path and waypoints."""
    fig, ax = plt.subplots()
    # Plot the path
    line = LineString([(wp.x, wp.y) for wp in waypoints])
    x, y = line.xy
    ax.plot(x, y, 'g-', label='Flight Path', linewidth=2, alpha=0.6)
    
    # Plot waypoints
    for wp in waypoints:
        ax.plot(wp.x, wp.y, 'ro', label='Waypoint' if 'Waypoint' not in ax.get_legend_handles_labels()[1] else "")
    
    ax.set_title("Flight Path")
    ax.legend()
    plt.show()

def calculate_flight_time(distance, speed):
    """Calculate the flight time based on total distance and speed."""
    return distance * 1000 / speed / 60

# Example Usage
sorted_waypoints = sort_waypoints(waypoints,angle)
path_distance = calculate_path_distance(sorted_waypoints)
flight_time = calculate_flight_time(path_distance, flight_speed)

# Output and Plot
print(f"Total path distance: {path_distance:.2f} kilometers")
print(f"Estimated flight time: {flight_time:.2f} minutes")
plot_path(sorted_waypoints)

# calculate groups of waypoints (less than 50 per time, estimated time of flight ca. 20 minutes).
max_path_length = flight_speed * flight_length * 60 /1000 # kilometers
subsets_waypoints = [] 
k = 0
for i in range(len(sorted_waypoints)):
    path_length = calculate_path_distance(sorted_waypoints[k:i])
    if (path_length > max_path_length) or (i-k)>50:
        waypoints_subset = sorted_waypoints[k:i-1]
        subsets_waypoints.append(waypoints_subset)
        k = i - 1
    last_k = k
subsets_waypoints.append(sorted_waypoints[last_k::])
print(f"Total number of flights: {len(subsets_waypoints)}")

#%%
# transform waypoints to epsg:4326
def transform_waypoints(waypoints):
    coords = []
    transformer = Transformer.from_crs("EPSG:3031", "EPSG:4326")
    for waypoint in waypoints:
        lat,lon = transformer.transform(waypoint.x,waypoint.y)
        coords.append([lat,lon])
    # repeat first row to have the startpoint correctly in the kml
    repeat_array = np.append(np.array([2.]),np.ones(len(coords)-1))
    coords_repeated = np.repeat(coords,repeat_array.astype(int),axis=0)
    return(coords_repeated)


# generate kml's of groups of waypoints

#%%
def register_namespaces():
    # Register any namespace if needed to maintain the same prefixes
    ET.register_namespace('', "http://www.opengis.net/kml/2.2")
    ET.register_namespace('wpml', "http://www.dji.com/wpmz/1.0.2")


def load_wpml(path):
    register_namespaces()
    tree = ET.parse(path)
    root = tree.getroot()
    return tree, root


def remove_excess_placemarks(input_filename, output_filename, start_index):
    """
    Manually remove placemarks from an XML file starting from a specified index.
    
    Args:
    input_filename (str): The name of the input XML file.
    output_filename (str): The name of the output XML file where modifications are saved.
    start_index (int): The index of the first placemark to remove.
    """
    with open(input_filename, 'r') as file:
        lines = file.readlines()

    with open(output_filename, 'w') as file:
        skip = False
        placemark_count = 0
        skip_this_placemark = False
        
        for line in lines:
            if '<Placemark>' in line:
                placemark_count += 1  # Increment count every time a new Placemark starts
            
            if placemark_count >= start_index:
                skip_this_placemark = True
            
            if skip_this_placemark:
                if '</Placemark>' in line:
                    skip_this_placemark = False  # Stop skipping after this line
                    continue  # Skip writing this line too

    # Write the line if it's not part of a skipped placemark
            if not skip_this_placemark:
                file.write(line)

def modify_waypoints(root, new_waypoints):
    # This example will shift each waypoint by a fixed latitude and longitude
    namespaces =  namespaces = {
        'kml': 'http://www.opengis.net/kml/2.2',  # Default KML namespace
        'wpml': 'http://www.dji.com/wpmz/1.0.2'  # Custom namespace for DJI specific elements
    }
    placemarks = root.findall('.//kml:Placemark', namespaces=namespaces)
    print(f"Found {len(placemarks)} placemark elements.")
    
    for number_point,placemark in enumerate(placemarks):
        # replace points with new coordinates
        if number_point < len(new_waypoints):
            point = placemark.find('.//kml:Point/kml:coordinates', namespaces=namespaces)
            if point is not None:
                coords = point.text.strip().split(',')
                #print(f"Original coords: {coords}")
                new_lat = new_waypoints[number_point][1]  # Add latitude shift
                new_lon = new_waypoints[number_point][0] # Add longitude shift
                point.text = f"{new_lat},{new_lon}"
        
def change_height_speed(root,new_height,new_speed):
    namespaces =  namespaces = {
        'kml': 'http://www.opengis.net/kml/2.2',  # Default KML namespace
        'wpml': 'http://www.dji.com/wpmz/1.0.2'  # Custom namespace for DJI specific elements
    }
    # Find all executeHeight elements and update them
    execute_heights = root.findall('.//wpml:executeHeight', namespaces=namespaces)
    waypoint_speeds = root.findall('.//wpml:waypointSpeed', namespaces=namespaces)
    # Find all waypointTurnMode elements and update them
    turn_modes = root.findall('.//wpml:waypointTurnMode', namespaces=namespaces)
    straight_line = root.findall('.//wpml:useStraightLine',namespaces=namespaces)
    heading_modes = root.findall('.//wpml:waypointHeadingMode',namespaces=namespaces)
    #print(f"Found {len(execute_heights)} executeHeight elements.")
    for height_element in execute_heights:
        height_element.text = str(new_height)
        #print(f"Updated height to {new_height}")
    for speed_element in waypoint_speeds:
        speed_element.text = str(new_speed)
        #print(f"Updated speed to {new_speed}")
    for straight_line_element in straight_line:
        straight_line_element.text = str(1)
    for turn_mode_element in turn_modes:
        turn_mode_element.text = str('toPointAndStopWithDiscontinuityCurvature')
    for heading_mode_element in heading_modes:
        heading_mode_element.text = str('followWayline')
    # updated_count = 0
    # for mode in turn_modes:
    #     if mode.text == 'toPointAndPassWithContinuityCurvature':
    #         mode.text = waypoint_turn_new #toPointAndStopWithDiscontinuityCurvature
    #         updated_count += 1
    #         print(f"Updated waypoint turn mode from  to {waypoint_turn_new}")


def save_wpml(tree, path):
    register_namespaces()
    
    # Writing to a string to replace single quotes
    xml_string = ET.tostring(tree.getroot(), encoding='unicode')
    xml_declaration = '<?xml version="1.0" encoding="UTF-8"?>\n'
    modified_xml_string = xml_declaration + xml_string

    # Write the final string to file
    with open(path, 'w', encoding='UTF-8') as f:
        f.write(modified_xml_string)
        
    #tree.write(path, xml_declaration=True, encoding='UTF-8')

def append_missing_placemark_tag(file_path):
    """
    Ensure the closing </Placemark> tag is present at the end of the file.
    
    Args:
    file_path (str): Path to the XML/KML file to check and modify.
    """
    with open(file_path, 'r+') as file:  # Open the file for reading and writing
        lines = file.readlines()  # Read all lines into a list
        
        # Check if the target insertion line already contains the closing tag
        if '</Placemark>' in lines[-4]:  # Check the third line from the end
            print("No need to insert </Placemark>, it's already there.")
            return

        # Insert the </Placemark> tag three lines from the end with 7 spaces of indentation
        lines.insert(-3, '       </Placemark>\n')  # '       ' represents 7 spaces

        # Write the modified lines back to the file
        with open(file_path, 'w') as file:
            file.writelines(lines)
        print("Inserted missing </Placemark> tag appropriately.")
        
def fix_file_ending(file_path):
    """
    Ensure the file ends with the correct closing tags and insert </Placemark> if missing.
    
    Args:
    file_path (str): Path to the XML/KML file to modify.
    """
    ending_tags = [
        '      </Placemark>\n',
        '    </Folder>\n',
    '  </Document>\n',
    '</kml>\n'
    ]

    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Determine if any of the required ending tags are missing
        is_missing = not all(lines[-len(ending_tags) + i].strip() == ending_tags[i].strip() for i in range(len(ending_tags)))

        # If missing, append the necessary tags
        if is_missing:
            with open(file_path, 'a') as file:  # Open file in append mode to ensure writability
                for tag in ending_tags:
                    file.write(tag)
                print("Appended missing ending tags to the file.")
        else:
            print("All ending tags are present. No changes made.")


# Load WPML file
tree, root = load_wpml('../drone_survey/wpmz/waylines.wpml')
#%%
for flight_n in range(len(subsets_waypoints)):
    attempt = attempt + flight_n
    new_kmz_name = f'{new_kmz_name_start}part{flight_n+1}'
    
    # Modify the waypoints
    new_waypoints = transform_waypoints(subsets_waypoints[flight_n])
    modify_waypoints(root, new_waypoints)  # Define your shifts here
    
    # change height
    change_height_speed(root,altitude,flight_speed)
    
    # Save the modified WPML file
    save_wpml(tree, '../drone_survey/wpmz/modified_waylines.wpml')
    
    # remove excess waypoints
    remove_excess_placemarks('../drone_survey/wpmz/modified_waylines.wpml', '../drone_survey/wpmz/modified_waylines_noexcess.wpml', len(new_waypoints))
    
    # add missing placemark
    fix_file_ending('../drone_survey/wpmz/modified_waylines_noexcess.wpml')
    #append_missing_placemark_tag('../drone_survey/wpmz/modified_waylines_noexcess.wpml')
    
    print("WPML modification complete.")
    
    try:
        os.rename('../drone_survey/wpmz', f'../drone_survey/wpmz_{kmz_name[:-4]}_{attempt}')
    except FileNotFoundError:
        print('')
    os.mkdir('../drone_survey/wpmz')
    
    shutil.copy(f'../drone_survey/wpmz_{kmz_name[:-4]}_{attempt}/modified_waylines_noexcess.wpml','../drone_survey/wpmz')
    
    os.rename('../drone_survey/wpmz/modified_waylines_noexcess.wpml','../drone_survey/wpmz/waylines.wpml')
    
    shutil.copy(f'../drone_survey/wpmz_{kmz_name[:-4]}_{attempt}/template.kml','../drone_survey/wpmz')
    
    def zipdir(path, ziph):
        # ziph is zipfile handle
        for root, dirs, files in os.walk(path):
            for file in files:
                ziph.write(os.path.join(root, file),
                           os.path.relpath(os.path.join(root, file),
                                           os.path.join(path, '..')))
    
    new_kmz_path = f'../drone_survey/{new_kmz_name}.kmz'
    with zipfile.ZipFile(new_kmz_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipdir('../drone_survey/wpmz', zipf)
    
    print("New KMZ file created at:", new_kmz_path)
    
    
    # check new file
    extract_path_results = '../drone_survey/check_results/'
    
    with zipfile.ZipFile(f'../drone_survey/{new_kmz_name}.kmz', 'r') as zip_ref:
        zip_ref.extractall(extract_path_results)
    
    tree_results, root_results = load_wpml('../drone_survey/check_results/wpmz/waylines.wpml')
    namespaces =  namespaces = {
            'kml': 'http://www.opengis.net/kml/2.2',  # Default KML namespace
            'wpml': 'http://www.dji.com/wpmz/1.0.2'  # Custom namespace for DJI specific elements
        }
    placemarks = root_results.findall('.//kml:Placemark', namespaces=namespaces)
    print(f"Found {len(placemarks)} waypoints.")
 