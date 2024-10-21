import json
import os

# Directory containing the GeoJSON file and where output files will be saved
input_dir = "/home/oem/.local/share/QGIS/QGIS3/profiles/default/python/plugins/atr/CD_pipeline_multi/geojson"
output_dir = "/home/oem/.local/share/QGIS/QGIS3/profiles/default/python/plugins/atr/CD_pipeline_multi/multi_json_converter/output/"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# File name of the GeoJSON file to read
input_file = "json_file.geojson"
input_file_path = os.path.join(input_dir, input_file)

# Read the GeoJSON data from the file
try:
    with open(input_file_path, 'r') as file:
        data = json.load(file)
except FileNotFoundError:
    print(f"File not found: {input_file_path}")
    exit(1)
except json.JSONDecodeError:
    print("Error decoding JSON. Please check the file format.")
    exit(1)

# Debug output to check the structure of the GeoJSON data
print("Data loaded successfully. Here is the structure:")
print(json.dumps(data, indent=4))

# Correct the feature type if necessary
for feature in data.get('result', {}).get('features', []):
    if feature.get('type') != 'Feature':
        feature['type'] = 'Feature'

# Process and create files for each feature
if data.get('result', {}).get('type') == 'FeatureCollection' and 'features' in data.get('result', {}):
    if len(data['result']['features']) == 0:
        print("No features found in the GeoJSON file.")
    else:
        for idx, feature in enumerate(data['result']['features']):
            feature_filename = f"feature_{idx + 1}.geojson"
            feature_filepath = os.path.join(output_dir, feature_filename)

            feature_data = {
                "type": "FeatureCollection",
                "features": [feature]
            }

            with open(feature_filepath, 'w') as file:
                json.dump(feature_data, file, indent=4)

            print(f"Created file: {feature_filepath}")
else:
    print("Invalid GeoJSON format. Ensure the file is a valid FeatureCollection and contains features.")
