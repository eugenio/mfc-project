#!/usr/bin/env python3
import re

# Read the file
with open('q-learning-mfcs/src/config/config_io.py', 'r') as f:
    content = f.read()

# Find the position after state_space handling
pattern = r"(        if 'state_space' in data and isinstance\(data\['state_space'\], dict\):\n            state_data = convert_lists_to_tuples_for_dataclass\(data\['state_space'\], StateSpaceConfig\)\n            data\['state_space'\] = StateSpaceConfig\(\*\*state_data\))"

addition = '''
        
        # Handle electrode configurations
        if 'anode_config' in data and isinstance(data['anode_config'], dict):
            anode = data['anode_config']
            if 'material' in anode and isinstance(anode['material'], str):
                anode['material'] = ElectrodeMaterial(anode['material'])
            if 'geometry' in anode and isinstance(anode['geometry'], dict):
                geom = anode['geometry']
                if 'geometry_type' in geom and isinstance(geom['geometry_type'], str):
                    geom['geometry_type'] = ElectrodeGeometry(geom['geometry_type'])
                from .electrode_config import ElectrodeGeometrySpec
                anode['geometry'] = ElectrodeGeometrySpec(**geom)
            data['anode_config'] = ElectrodeConfiguration(**anode)
            
        if 'cathode_config' in data and isinstance(data['cathode_config'], dict):
            cathode = data['cathode_config']
            if 'material' in cathode and isinstance(cathode['material'], str):
                cathode['material'] = ElectrodeMaterial(cathode['material'])
            if 'geometry' in cathode and isinstance(cathode['geometry'], dict):
                geom = cathode['geometry']
                if 'geometry_type' in geom and isinstance(geom['geometry_type'], str):
                    geom['geometry_type'] = ElectrodeGeometry(geom['geometry_type'])
                from .electrode_config import ElectrodeGeometrySpec
                cathode['geometry'] = ElectrodeGeometrySpec(**geom)
            data['cathode_config'] = ElectrodeConfiguration(**cathode)'''

# Replace
new_content = re.sub(pattern, r'\1' + addition, content)

# Write back
with open('q-learning-mfcs/src/config/config_io.py', 'w') as f:
    f.write(new_content)

print("Added electrode configuration handling")