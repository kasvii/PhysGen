import os
import json
import argparse
def find_obj_files(directory, file_type):
    files_ = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(f'{file_type}'):
                files_.append(os.path.join(root, file))
    return files_

def save_to_json(data, json_file):
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=4)

def main(directory_to_search, json_file_path, file_type) -> None:
    files = find_obj_files(directory_to_search, file_type)
    save_to_json(files, json_file_path)

    print(f"Found {len(files)} {file_type} files and saved to {json_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory_to_search",
        type= str,
        help="Directory to traverse",
    )
    parser.add_argument(
        "--json_file_path",
        type= str,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--file_type",
        type= str,
        help="Mesh file suffix to detect",
    )
    args, extras = parser.parse_known_args()
    main(args.directory_to_search, args.json_file_path, args.file_type)
