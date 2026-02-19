import requests
import json
import pdal

# The EPT URL for a typical USGS project (adjust as needed)
# Using a Long Island project as a stable test case
test_url = "http://usgs-lidar-public.s3.amazonaws.com/USGS_LPC_CA_SoCal_Wildfires_B1_2018_LAS_2019/ept.json"

def test_connection():
    print("--- USGS LIDAR CONNECTIVITY TEST ---")
    
    # 1. Test HTTP reachability
    try:
        response = requests.get(test_url, timeout=10)
        if response.status_code == 200:
            print("[SUCCESS] HTTP: Able to reach the USGS AWS server.")
        else:
            print(f"[FAILED] HTTP: Server responded with status code {response.status_code}")
            return
    except Exception as e:
        print(f"[FAILED] HTTP: Could not connect to the internet or AWS. Error: {e}")
        return

    # 2. Test PDAL's ability to "see" the data
    try:
        # We define a tiny pipeline just to ask for info/metadata
        pipeline_json = {
            "pipeline": [
                {
                    "type": "readers.ept",
                    "filename": test_url
                }
            ]
        }
        
        pipeline = pdal.Pipeline(json.dumps(pipeline_json))
        # This doesn't download points, it just validates the source
        metadata = pipeline.quickinfo
        
        print("[SUCCESS] PDAL: Successfully parsed the EPT metadata.")
        print(f"Dataset Info: Found {metadata['readers.ept']['num_points']:,} points.")
        
    except Exception as e:
        print(f"[FAILED] PDAL: PDAL is installed, but it can't read the EPT file. Error: {e}")

if __name__ == "__main__":
    test_connection()