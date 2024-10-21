import os
from datetime import datetime
from qgis.core import QgsProject, QgsVectorLayer

def load_geojson_files(directory_path):
    # Ensure the directory exists
    if not os.path.isdir(directory_path):
        print(f"The directory {directory_path} does not exist.")
        return

    # Capture the start time
    start_time = datetime.now()
    print(f"Start time: {start_time}")

    # Initialize a counter to keep track of loaded files
    loaded_files_count = 0

    # Get a list of all .geojson files in the directory
    geojson_files = [f for f in os.listdir(directory_path) if f.endswith(".geojson")]
    
    # Sort the files to ensure a consistent loading order (optional)
    geojson_files.sort()
    
    # Iterate through the list of GeoJSON files and load them
    for filename in geojson_files:
        file_path = os.path.join(directory_path, filename)
        # Create a vector layer from the GeoJSON file
        layer = QgsVectorLayer(file_path, filename, "ogr")
        
        # Check if the layer is valid
        if not layer.isValid():
            print(f"Failed to load {file_path}")
            continue
        
        # Add the layer to the QGIS project
        QgsProject.instance().addMapLayer(layer)
        loaded_files_count += 1

        # Provide progress feedback
        if loaded_files_count % 100 == 0:
            print(f"{loaded_files_count} files loaded...")

    # Capture the end time
    end_time = datetime.now()
    print(f"End time: {end_time}")
    print(f"Total loaded files: {loaded_files_count}")
    print(f"Total time taken: {end_time - start_time}")

# Define the directory containing your .geojson files
geojson_directory = "/home/oem/.local/share/QGIS/QGIS3/profiles/default/python/plugins/atr/CD_pipeline_multi/geojson/seperate_jsons"

# Call the function to load all .geojson files in the specified directory
load_geojson_files(geojson_directory)

# Optional: Refresh the QGIS interface if running within QGIS
iface.mapCanvas().refreshAllLayers()
