class CNode:
    """
    Class representing a graph node.

    Parameters
        p_id            : The ID of the node
        p_pos (tuple)   : The position of the node. The format is (x, y) for 2D node and (x, y, z) for 3D node 
        p_degree (int)  : The degree (number of connected nodes) of the node
    """
    def __init__(self, p_id, p_pos: tuple, p_degree: int):
        self._id = p_id
        self.pos = p_pos
        self.degree = p_degree

    def __eq__(self, p_node):
        return type(p_node) == type(self) and self._id == p_node._id and self.pos == p_node.pos and self.degree == p_node.degree

    def __repr__(self):
        return "CNode (id: {}, position: {}, degree: {})".format(self._id, self.pos, self.degree)
        
class CEdge:
    """
    Class representing a graph edge.

    Parameters
        p_id            : The ID of the edge
        p_node1 (CNode) : The first end point/node
        p_node2 (CNode) : The second end point/node
    """
    def __init__(self, p_id, p_node1: CNode, p_node2: CNode):
        self._id = p_id
        self.node1 = p_node1
        self.node2 = p_node2
    
    def getDistance(self, norm_type: str = "L2"):
        """
         Indicates the distance between the two nodes, depending on the chosen norm type.

        Parameters
            norm_type (str) : The type of norm to compute the distance. Implemented norms are :
                                  - None

        Returns
            The distance between the two nodes
        """
        from utils.metrics import distance
        return distance(self.node1.pos, self.node2.pos, norm_type)

    def getMidPoint(self):
        """
         Indicates the middle point of the edge.

        Returns
            <expression> (CNode) : the middle point of the edge.
        """  
        from math import floor
        middle_point = tuple([floor(sum(x)/2) for x in zip(self.node1.pos, self.node2.pos)])
        
        return CNode("edge_{}".format(self._id), middle_point, 2)

    def __eq__(self, p_edge):
        return type(p_edge) == type(self) and self._id == p_edge._id and self.node1 == p_edge.node1 and self.node2 == p_edge.node2
        
class CCenterline(CEdge):
    """
    Class representing a centerline.
    As a subclass of CEdge, this class can be used into a graph. In this case, it could be seen as a "parametric edge" and is usefull for skeleton representation of an object, eg. for vessel structure/graph extraction. 

    Parameters
        p_id                        : The ID of the edge
        p_node1 (CNode)             : The first end point/node
        p_node2 (CNode)             : The second end point/node
        p_skeleton_points (list)    : The list of points that form the skeleton of the centerline. None (default)
        p_curveness (float)         : The curveness of the edge. None (default)
    """
    def __init__(self, p_id, p_node1: CNode, p_node2: CNode, p_skeleton_points: list = None, p_curveness: float = None):
        super().__init__(p_id, p_node1, p_node2)

        self.skeleton_points = p_skeleton_points
        self.curveness = p_curveness

    def getDistance(self, norm_type: str = "default"):
        """
         Indicates the distance between the two nodes according to the centerline curvature (except for the default distance, see below) and the chosen norm type.

        Parameters
            norm_type (str) : The type of norm to compute the distance. Implemented norms are :
                - "default" : CEdge distance. Computes the distance between node1 and node2 without considering the curvature of the centerline.
                - "L2"      : the sum of euclidean distances between each point composing the centerline (nodes and skeleton points ; node1<->skpt1 + skpt2<->skpt3 + ... + skptn<->node2).

        Returns
            The distance between the two nodes
        """
        if norm_type == "default":
            dist = super().getDistance("L2")

        elif norm_type == "L2":
            from utils.metrics import distance

            dist = 0.0

            # Remember that node1 and node2 are not included in skeleton points on the Voreen file
            skpt1 = self.node1.pos
            for skpt2 in self.skeleton_points:
                dist = dist + distance(skpt1, skpt2, "L2")
                skpt1 = skpt2
            dist = dist + distance(skpt1, self.node2.pos, "L2")
        else:
            raise NotImplementedError

        return dist

    def getMidPoint(self) -> CNode:
        """
         Indicates the middle point of the centerline.

        Returns
            <expression> (CNode) : the middle point of the centerline.
        """ 
        middle_point = self.skeleton_points[round(len(self.skeleton_points) / 2)] # We place ourselves in the middle of the centerline

        return CNode("skvx_{}".format(self._id), middle_point, 2)

    def __eq__(self, p_centerline):
        equality = super().__eq__(p_centerline)
        equality = equality and self.curveness == p_centerline.curveness
        equality = equality and self.skeleton_points == p_centerline.skeleton_points

        # Here, `is not None` condition is not mandatory because already asserted in the base class.
        return equality

    def __repr__(self):
        variable_output = ""
        if self.curveness is not None:
            variable_output = variable_output + ", curveness: {}".format(self.curveness)
        if self.skeleton_points is not None:
            variable_output = variable_output + ", skeleton: {} pts".format(len(self.skeleton_points))
            
        return "CCenterline (id: {}, node1: {}, node2: {}{})".format(self._id, self.node1._id, self.node2._id, variable_output)

class CGraph:
    """
    Class representing a graph.
    The nodes and edges composing the graph are indexed by their respective ID (eg. graph.nodes[id]).

    Parameters
        p_nodes (list[CNodes])                  : The list of nodes composing the graph.
        p_connections (list[CEdge|CCenterline]) : The list of edges composing the graph.
    """    
    def __init__(self, p_nodes: list, p_connections: list):
        self.nodes = dict()
        self.connections = dict()
        self._nodes_sort = dict() # dict of int:list(node) where the key corresponds to degree and points on the list of nodes of that degree 

        for node in p_nodes:
            self.nodes[node._id] = node
            self._nodes_sort.setdefault(node.degree, []).append(node)
                 
        for connection in p_connections:
            connection.node1 = self.nodes[connection.node1] # Replace the ID of the node by the node itself 
            connection.node2 = self.nodes[connection.node2] # Replace the ID of the node by the node itself 
            
            self.connections[connection._id] = connection
            
    def get_nodes_of_degree(self, degree: int):
        return self._nodes_sort[degree]

    def __eq__(self, p_graph):
        # We don't check the equality of `_nodes_sort` because the variable contains only references.
        # If `nodes` are equals, `_nodes_sort` should be equal too.
        #
        # We use the default == opertor for dictionaries. It checks that both dict have the same keys and that the the values for a given key are equal. 
        return type(p_graph) == type(self) and self.nodes == p_graph.nodes and self.connections == p_graph.connections