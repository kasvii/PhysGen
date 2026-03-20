python detect_path.py   --directory_to_search ../data/drivaernet_plus/meshes/3DMeshesSTL/ \
                        --json_file_path ../data/drivaernet_plus/meshes.json  \
                        --file_type .stl

python to_watertight_mesh.py  --resolution 512 \
                              --json_file_path ../data/drivaernet_plus/meshes.json \
                              --remesh_target_path ../data/drivaernet_plus/meshes/remesh

python detect_path.py   --directory_to_search ../data/drivaernet_plus/meshes/remesh/ \
                        --json_file_path ../data/drivaernet_plus/meshes/watertight_path.json \
                        --file_type .obj

python sharp_sample.py  --json_file_path ../data/drivaernet_plus/meshes/watertight_path.json  \
                        --point_number 371088 \
                        --angle_threshold 15 \
                        --sharp_point_path ../data/drivaernet_plus/meshes/dora_sharp_point_ply \
                        --sample_path ../data/drivaernet_plus/meshes/dora_extraction