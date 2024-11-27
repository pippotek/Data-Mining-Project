import json

# Function to merge, deduplicate, and find duplicates
def merge_and_deduplicate(json_file1, json_file2, output_file_path):
    # Load JSON data from files
    with open(json_file1, "r") as file1:
        json_list1 = json.load(file1)
    with open(json_file2, "r") as file2:
        json_list2 = json.load(file2)
    
    # Combine both lists
    combined_list = json_list1 + json_list2
    
    # Find duplicates by `_id`
    duplicates = {}
    for item in combined_list:
        item_id = item["_id"]
        if item_id not in duplicates:
            duplicates[item_id] = []
        duplicates[item_id].append(item["title"])
    
    # Extract unique data
    unique_data = {item["_id"]: item for item in combined_list}.values()
    
    # Write duplicates to the output file
    with open(output_file_path, "w") as file:
        for _id, titles in duplicates.items():
            if len(titles) > 1:
                file.write(f"ID: {_id}\n")
                for title in titles:
                    file.write(f"- {title}\n")
                file.write("\n")
    
    # Return the deduplicated data
    return list(unique_data), output_file_path

# Input file paths
json_file1 = "sample data/sample_articles.json"  # Replace with your first JSON file path
json_file2 = "sample data/sample_articles2.json"  # Replace with your second JSON file path
output_file_path = "duplicate_titles.txt"

# Run the function
deduplicated_data, output_file = merge_and_deduplicate(json_file1, json_file2, output_file_path)

# Save deduplicated data to a new JSON file
with open("deduplicated_data.json", "w") as dedup_file:
    json.dump(deduplicated_data, dedup_file, indent=4)

# Print results
print(f"Deduplicated data saved to: deduplicated_data.json")
print(f"Duplicate titles saved to: {output_file}")
