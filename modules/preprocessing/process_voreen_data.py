import os
import json
import csv

from graph import CGraph, CCenterline, CNode

def parse_voreen_VesselGraphSave_file(p_vessel_graph_file: str):
    """
    Create a graph of a vessel structure by parsing a VesselGraphSave output file (*.vvg ; see Voreen).

    Parameters
        p_vessel_graph_file (str) : The path to the VesselGraphSave output file.

    Returns
        CGraph of a vessel structure.
    """
    filename, extension = os.path.splitext(p_vessel_graph_file)
    assert extension == ".vvg"
    
    with open(p_vessel_graph_file) as graph_file:
        json_data = json.load(graph_file)
        graph_data = json_data["graph"]
        
        mynodes = []
        for node in graph_data["nodes"]:
            node_id     = node["id"]
            node_pos    = tuple(node["pos"])
            node_degre  = len(node["edges"])

            mynodes.append(CNode(node_id, node_pos, node_degre))
            
        mycenterlines = []
        for centerline in graph_data["edges"]:
            centerline_id = centerline["id"]
            node1_id = centerline["node1"]
            node2_id = centerline["node2"]
            
            skeleton_points = []
            for point in centerline["skeletonVoxels"]:
                skeleton_points.append(tuple(point["pos"]))
                
            mycenterlines.append(CCenterline(centerline_id, node1_id, node2_id, p_skeleton_points=skeleton_points))
            
    return CGraph(mynodes, mycenterlines)

def parse_voreen_VesselGraphGlobalStats_files(p_nodes_file_path: str, p_centerlines_file_path: str):
    """
    Create a graph of a vessel structure by parsing a VesselGraphGlobalStats output files (*.csv ; see Voreen).

    Parameters
        p_nodes_file_path (str) : The path to the VesselGraphGlobalStats nodes export file.
        p_centerlines_file_path (str) : The path to the VesselGraphGlobalStats edges export file.

    Returns
        CGraph of a vessel structure.
    """
    filename_nodes_f, extension_nodes_f = os.path.splitext(p_nodes_file_path)
    filename_edges_f, extension_edges_f = os.path.splitext(p_centerlines_file_path)
    assert extension_nodes_f == ".csv" and extension_edges_f == ".csv"

    nodes = []
    with open(p_nodes_file_path, newline='') as nodes_csvfile:
        nodereader = csv.reader(nodes_csvfile, delimiter=';')
        
        for row in list(nodereader)[1:]:
            node_id     = int(row[0])
            node_pos    = (float(row[1]), float(row[2]), float(row[3]))
            node_degre  = int(row[4])

            nodes.append(CNode(node_id, node_pos, node_degre))

    centerlines = []
    with open(p_centerlines_file_path, newline='') as centerlines_csvfile:
        centerlinereader = csv.reader(centerlines_csvfile, delimiter=';')
        
        for row in list(centerlinereader)[1:]:
            centerline_id = int(row[0])
            node1_id = int(row[1])
            node2_id = int(row[2])
            curveness = float(row[5])
            
            centerlines.append(CCenterline(centerline_id, node1_id, node2_id, p_curveness=curveness))
    
    return CGraph(nodes, centerlines)    

#def process_voreen_vessel_graph(path: str):
#    nprocessed_files = 0
#nodes = []
#
#    if os.path.isfile(path):
#        nodes = parse_node_file(path)
#        nprocessed_files += 1
#    elif os.path.isdir(path):
#        for file in os.listdir(path):
#nodes.append(parse_node_file(os.path.join(path, file)))
#            nprocessed_files += 1
#    else:
#        raise FileNotFoundError()
#
#    return nprocessed_files, nodes
