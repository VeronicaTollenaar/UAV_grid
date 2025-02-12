# UAV_grid
Generates KML file needed to fly a grid with cheap consumer DJI UAVs
In this case it is developped for a DJI Air 3.
I have used ChatGPT for help in writing this code

Problems to fix:
- When starting the waypoints file on the UAV, the first waypoint needs to be deleted as to fly without errors.
- The extent shapefile needs to be in EPSG:3031 (or in any other projection that uses meters as unit).

