import json
import os

# Sample data provided
data = {
    "status": 404,
    "msg": "changes detected",
    "result": {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[68.46135229098923, 28.274843806467704, 0], [68.46134797907587, 28.274843806467704, 0], [68.46134797907587, 28.274839494554342, 0]]]
                },
                "properties": {
                    "name": "changes",
                    "fill": "rgba'(255,255,255,1)",
                    "fill-opacity": 0.4,
                    "stroke": [255, 0, 0],
                    "stroke-width": 2,
                    "stroke-opacity": 1
                }
            },
            {
                "type": "feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[68.46579787366728, 28.274835182640977, 0], [68.46580218558064, 28.274800687334068, 0], [68.46587548810783, 28.27478343968061, 0], [68.46590567150137, 28.27475756820043, 0], [68.46593585489492, 28.27475756820043, 0], [68.46595310254837, 28.274774815853885, 0], [68.46595310254837, 28.274800687334068, 0], [68.46593585489492, 28.274817934987524, 0], [68.46588411193456, 28.27482655881425, 0], [68.46583236897419, 28.274852430294434, 0], [68.4658194332341, 28.274852430294434, 0]]]
                },
                "properties": {
                    "name": "changes",
                    "fill": "rgba'(255,255,255,1)",
                    "fill-opacity": 0.4,
                    "stroke": [255, 0, 0],
                    "stroke-width": 2,
                    "stroke-opacity": 1
                }
            }
        ]
    }
}

# Directory to save the files
output_dir = "output_features"
os.makedirs(output_dir, exist_ok=True)

# Process and create files for each feature
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

