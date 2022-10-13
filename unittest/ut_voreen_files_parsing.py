import os
import unittest

from graph import CGraph, CCenterline, CNode
import preprocessing.process_voreen_data as process_voreen_data

testsdata_dir = os.path.join(os.getcwd(), "data/")

class TestVoreenFileParsing(unittest.TestCase):
    ###
    # Set-up data
    ###
    def _createSkeletonPointsForSetUp(self):
        # 3 points for centerline 0
        skpoint_centerline_0 = [ 
            (-108.70704650878906, -238.46878051757813, 248.0),
            (-108.70704650878906, -239.15628051757813, 248.6875),
            (-108.68621826171875, -240.0729522705078, 249.3541717529297)
        ]

        # 2 points for centerline 1
        skpoint_centerline_1 = [ 
            (-91.70704650878906, -225.15628051757813, 245.0),
            (-91.70704650878906, -226.09378051757813, 245.0625)
        ]
                                
        # 1 point for centerline 2
        skpoint_centerline_2 = [(-99.76954650878906, -266.0937805175781, 256.0)]

        # 3 points for centerline 3
        skpoint_centerline_3 = [ 
            (-109.66539001464844, -242.98960876464845, 252.64584350585938),
            (-110.68621826171875, -243.6354522705078, 253.2291717529297),
            (-111.70704650878906, -243.78128051757813, 253.75)
        ]

        # 3 points for centerline 4
        skpoint_centerline_4 = [
            (-110.76954650878906, -257.4062805175781, 262.0),
            (-111.70704650878906, -257.0937805175781, 262.0),
            (-112.70704650878906, -256.7187805175781, 262.0),
        ]

        # 3 points for centerline 5
        skpoint_centerline_5 = [
            (-119.70704650878906, -203.15628051757813, 248.0),
            (-119.70704650878906, -204.09378051757813, 248.0625),
            (-119.70704650878906, -205.09378051757813, 248.3125)
        ]

        return [skpoint_centerline_0, skpoint_centerline_1, skpoint_centerline_2, skpoint_centerline_3, skpoint_centerline_4, skpoint_centerline_5]

    def setUp_for_VesselGraphGlobalStats(self):
        self.nodes = [
            CNode(0,    (-91.707,   -224.094,   245),       1),
            CNode(1,    (-108.707,  -238.094,   247),       1),
            CNode(2,    (-119.707,  -202.094,   248),       1),
            CNode(6,    (-98.707,   -266.094,   256),       1),
            CNode(61,   (-108.374,  -241.76,    251.667),   3),
            CNode(62,   (-109.707,  -258.094,   262),       3),
            CNode(63,   (-138.04,   -233.427,   264),       3),
            CNode(65,   (-123.707,  -248.094,   264),       4)
        ]

        self.centerlines = [
            CCenterline(0, 1,   61, p_curveness=1.02284 ),
            CCenterline(1, 0,   61, p_curveness=1.14496 ),
            CCenterline(2, 6,   62, p_curveness=1.1236  ),
            CCenterline(3, 61,  65, p_curveness=1.15592 ),
            CCenterline(4, 62,  65, p_curveness=1.0322  ),
            CCenterline(5, 2,   63, p_curveness=1.05286 )
        ]

        self.init_graph = CGraph(self.nodes, self.centerlines)

    def setUp_for_VesselGraphSave(self):
        self.nodes = [
            CNode(0,    (-91.70704650878906,    -224.09378051757813,    245.0),             1),
            CNode(1,    (-108.70704650878906,   -238.09378051757813,    247.0),             1),
            CNode(2,    (-119.70704650878906,   -202.09378051757813,    248.0),             1),
            CNode(6,    (-98.70704650878906,    -266.0937805175781,     256.0),             1),
            CNode(61,   (-108.37371826171875,   -241.76043701171876,    251.6666717529297), 3),
            CNode(62,   (-109.70704650878906,   -258.0937805175781,     262.0),             3),
            CNode(63,   (-138.04037475585938,   -233.42709350585938,    264.0),             3),
            CNode(65,   (-123.70704650878906,   -248.09378051757813,    264.0),             4)
        ]

        all_sk_points = self._createSkeletonPointsForSetUp()

        self.centerlines = [
            CCenterline(0, 1,   61, p_skeleton_points=all_sk_points[0]),
            CCenterline(1, 0,   61, p_skeleton_points=all_sk_points[1]),
            CCenterline(2, 6,   62, p_skeleton_points=all_sk_points[2]),
            CCenterline(3, 61,  65, p_skeleton_points=all_sk_points[3]),
            CCenterline(4, 62,  65, p_skeleton_points=all_sk_points[4]),
            CCenterline(5, 2,   63, p_skeleton_points=all_sk_points[5])
        ]

        self.init_graph = CGraph(self.nodes, self.centerlines)

    ###
    # Test
    ###
    def test_parsingVesselGraphGlobalStatsOutput(self):
        self.setUp_for_VesselGraphGlobalStats()

        nodes_file = os.path.join(testsdata_dir, "test_VesselGraphGlobalStats_nodes.csv")
        edges_file = os.path.join(testsdata_dir, "test_VesselGraphGlobalStats_centerlines.csv")
        parsed_graph = process_voreen_data.parse_voreen_VesselGraphGlobalStats_files(nodes_file, edges_file)

        self.assertEqual(parsed_graph, self.init_graph)

    def test_parsingVesselGraphSaveOutput(self):
        self.setUp_for_VesselGraphSave()

        graph_file = os.path.join(testsdata_dir, "test_VesselGraphSave_graph.vvg")
        parsed_graph = process_voreen_data.parse_voreen_VesselGraphSave_file(graph_file)
        
        self.assertEqual(parsed_graph, self.init_graph)

if __name__ == '__main__':
    unittest.main()