import os
import json
import requests
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from groq import Groq

# Map variables from environmental configurations
load_dotenv()

# Resolve the absolute path to the templates folder dynamically to ensure reliable asset lookup on Vercel
base_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(base_dir, '../templates')
app = Flask(__name__, template_folder=template_dir)

# Establish clean Groq validation initialization layers
groq_api_key = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=groq_api_key) if groq_api_key else None

def fetch_real_geojson_boundary(location, feature):
    """
    Queries the public Overpass API database to retrieve actual geographic boundary 
    polygon coordinates covering real environmental features inside the requested location area.
    """
    feature_tag_mapping = {
        "forests": '["natural"="wood"]',
        "glaciers": '["natural"="glacier"]',
        "desertification": '["natural"="desert"]',
        "wildfires": '["boundary"="national_park"]',
        "ocean_levels": '["natural"="water"]["water"="lagoon"]'
    }
    
    osm_tag = feature_tag_mapping.get(feature, '["natural"="wood"]')
    
    query = f"""
    [out:json][timeout:25];
    area["name"~"{location}", i_case]->.searchArea;
    (
      nwr{osm_tag}(area.searchArea);
    );
    out geom;
    """
    
    try:
        response = requests.post("https://overpass-api.de/api/interpreter", data={"data": query}, timeout=15)
        if response.status_code != 200:
            return None
            
        osm_data = response.json()
        
        geojson = {
            "type": "FeatureCollection",
            "features": []
        }
        
        for element in osm_data.get('elements', []):
            if 'geometry' in element:
                coords = [[pt['lon'], pt['lat']] for pt in element['geometry']]
                
                if coords and coords[0] != coords[-1]:
                    coords.append(coords[0])
                    
                feature_node = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [coords]
                    },
                    "properties": element.get('tags', {})
                }
                geojson["features"].append(feature_node)
                
                if len(geojson["features"]) >= 35:
                    break
                    
        return geojson if geojson["features"] else None
    except Exception:
        return None

@app.route('/')
def home():
    """Serves the modernized global context-aware interface dashboard terminal."""
    return render_template('index.html')

@app.route('/api/timelapse', methods=['POST'])
def handle_timelapse():
    """Mode 1 Endpoint: Evaluates context compatibility, corrects mismatch flags, and yields AI forecasts."""
    if not client:
        return jsonify({"output": "Configuration Error: GROQ_API_KEY could not be validated by server runtime."}), 500

    data = request.get_json() or {}
    location = data.get('location', 'Global Core')
    feature = data.get('feature', 'glaciers')
    target_year = data.get('year', '2040')
    pathway = data.get('pathway', 'business_as_usual')

    validation_system_prompt = (
        "You are a strict biophysical geography validation matrix. "
        "Your task is to analyze if a requested environmental indicator feature makes physical, biological, "
        "and geographical sense for a given location. Respond strictly in valid JSON format matching this schema:\n"
        "{\n"
        "  \"context_mismatch\": true or false,\n"
        "  \"mismatch_message\": \"Explanation statement if mismatch is true, otherwise empty string.\",\n"
        "  \"resolved_feature\": \"The selected feature string, OR if mismatch is true, select the most scientifically relevant alternative out of: ['glaciers', 'forests', 'ocean_levels', 'wildfires', 'desertification']\"\n"
        "}"
    )
    
    validation_user_prompt = f"Evaluate target location context: '{location}' against requested simulation parameter: '{feature}'."

    context_mismatch = False
    mismatch_message = ""
    resolved_feature = feature

    try:
        val_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": validation_system_prompt},
                {"role": "user", "content": validation_user_prompt}
            ],
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"}
        )
        parsed_val = json.loads(val_completion.choices[0].message.content)
        context_mismatch = parsed_val.get("context_mismatch", False)
        mismatch_message = parsed_val.get("mismatch_message", "")
        resolved_feature = parsed_val.get("resolved_feature", feature)
    except Exception:
        resolved_feature = feature

    real_area_geojson = fetch_real_geojson_boundary(location, resolved_feature)

    system_prompt = (
        "You are an advanced planetary ecosystem simulation engine tracking climate risks globally. "
        "Formulate clear, analytical climate forecast metrics based on the active policy scenario. "
        "Keep your output highly precise, professional, and limited to 3 concise sentences."
    )
    
    user_prompt = (
        f"Generate a long-range tracking summary report for the geographic region: '{location}'. "
        f"Provide analytical data projections tracking the active parameter feature: {resolved_feature.upper()} by the year {target_year} "
        f"under the environmental policy pathway model context: {pathway.replace('_', ' ').upper()}."
    )

    try:
        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama-3.3-70b-versatile"
        )
        ai_response = completion.choices[0].message.content
        
        return jsonify({
            "context_mismatch": context_mismatch,
            "mismatch_message": mismatch_message,
            "resolved_feature": resolved_feature,
            "output": ai_response,
            "real_geojson": real_area_geojson
        })
    except Exception as e:
        return jsonify({"output": f"Groq Infrastructure Interruption encountered during runtime simulation loops: {str(e)}"}), 500

@app.route('/api/advisor', methods=['POST'])
def handle_advisor():
    """Mode 2 Endpoint: Discharges actionable global environmental mitigation roadmaps."""
    if not client:
        return jsonify({"output": "Configuration Error: GROQ_API_KEY could not be validated by server runtime."}), 500

    data = request.get_json() or {}
    crisis = data.get('crisis', 'deforestation')

    system_prompt = (
        "You are a master planetary green strategy director and ecosystem infrastructure mitigation architect. "
        "Formulate precise international development metrics in exactly 3 bullet points or fewer."
    )
    user_prompt = f"Provide clear structural target goals, regional carbon emission milestones, and adaptation plans for {crisis}."

    try:
        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama-3.3-70b-versatile"
        )
        ai_response = completion.choices[0].message.content
        return jsonify({"output": ai_response})
    except Exception as e:
        return jsonify({"output": f"Groq Pipeline Interruption encountered during advisor processing: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)