import json


output_db_filepath = "./data/runways_database_all.json"
runway_db_to_merge = [
    "./data/runways_database.json",
    "./data/runways_database_LARDv2.json"
]

merged_db = {}

for database_file in runway_db_to_merge:
    with open(database_file, 'r') as f:
        data = json.load(f)

    for a_id, a_db in data.items():
        if not a_id in merged_db:
            merged_db[a_id] = {}
        for r_id, r_db in a_db.items():
            merged_db[a_id][r_id] = r_db.copy()

with open(output_db_filepath, "w") as f:
    json.dump(merged_db, f, indent=4)

print(f"Merged airports database in JSON file {output_db_filepath}.")