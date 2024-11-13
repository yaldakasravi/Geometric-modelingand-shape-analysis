import argparse
import numpy as np
from trimesh import TriMesh
import meshplot as mp
import os


def get_common(orig_mesh,v1,v2):
        v1_neighbors = orig_mesh.vertex_vertex_neighbors(v1)
        v2_neighbors = orig_mesh.vertex_vertex_neighbors(v2)
        #print("neibhors for both ")
        #print(v1_neighbors) 
        #print(v2_neighbors) 
        common = list(set(v1_neighbors).intersection(v2_neighbors))
        #print(f"selected common {common}") 
        v3,v4 = common[0],common[1]

        return common

def subdivision_method(mesh):
    """
    Perform one iteration of the Loop subdivision on the given mesh.
    
    Parameters:
    mesh (TriMesh): The input triangular mesh.

    Returns:
    TriMesh: The subdivided mesh.
    """
    # Step 1: Insert new vertices at the midpoints of each edge
    new_vertices = {}
    edge_vertex_map = {}
    new_vs = []  # To store the new vertices

    orig_mesh = mesh.copy()
    #orig_mesh.vs = np.array( [0,1,2])
    #print(f" orig mesh {orig_mesh.vs},\n current mesh {mesh.vs}")
    #input()
    
    for i, face in enumerate(mesh.faces):
        for edge in [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])]:
    #for edge in mesh.get_edges():
			
            v1, v2 = sorted(edge)  # Ensure consistent ordering
            if (v1, v2) not in new_vertices:
                new_vertex = (mesh.vs[v1] + mesh.vs[v2]) / 2.0  # Midpoint
                new_vertex_index = len(mesh.vs) + len(new_vertices)
                new_vertices[(v1, v2)] = new_vertex_index
                edge_vertex_map[(v1, v2)] = new_vertex
                new_vs.append(new_vertex)  # Add the new vertex to the list

    # Add the new vertices to the mesh by stacking them to the existing vertex array
    #print(mesh.vs.shape)
    old_vert_count = mesh.vs.shape[0]
    #print("orig mesh ")
    #input()
    if new_vs:
        mesh.vs = np.vstack([mesh.vs, np.array(new_vs)])
   
    #print(f" new mesh {mesh.vs.shape}")
    #input() 

    # Step 2: Modify mesh connectivity
    new_faces = []
    for face in mesh.faces:
        v0, v1, v2 = face
        # Get the new vertices for the edges
        e01 = new_vertices[(min(v0, v1), max(v0, v1))]
        e12 = new_vertices[(min(v1, v2), max(v1, v2))]
        e20 = new_vertices[(min(v2, v0), max(v2, v0))]
        # Create the four new faces
        new_faces.append([v0, e01, e20])
        new_faces.append([v1, e12, e01])
        new_faces.append([v2, e20, e12])
        new_faces.append([e01, e12, e20])
    
    # Replace the old faces with the new faces
    old_faces = mesh.faces
    mesh.faces = new_faces
        
    
    # Step 3: Relocate old vertices using Loop's weight formula
 #   for i, vertex in enumerate(mesh.vs[:old_vert_count]):
 #       #mesh.get_halfedges()
 #       neighbors = orig_mesh.vertex_vertex_neighbors(i)
 #       n = len(neighbors)
 #       print(n)
 #       w = ((64 * n) / (40 - (3 + 2 * np.cos(2 * np.pi / n))**2)) - n
 #       new_pos = (1 / (n + w)) * (w * vertex + sum([orig_mesh.vs[neighbor] for neighbor in neighbors]))
 #       mesh.vs[i] = new_pos
    
    # Step 4: Relocate new vertices inserted at the midpoints of the edges
    for (v1, v2), new_vertex_index in new_vertices.items():
        # Get adjacent faces sharing the edge v1v2
        #adjacent_faces = old_mesh.vertex_face_neighbors(v1)
        v1_neighbors = orig_mesh.vertex_vertex_neighbors(v1)
        v2_neighbors = orig_mesh.vertex_vertex_neighbors(v2)
        
        common = list(set(v1_neighbors).intersection(v2_neighbors))
        v3,v4 = common[0],common[1]


        inner_loop = [v1,v2,v3,v4]
        outer_loop = []
        
        edges_to_be = [
                      [v1,v3],
                      [v2,v3],
                      [v1,v4],
                      [v2,v4]
                      ]


        #gettitng the outer loop
        #print("*"*20) 
        for ed in edges_to_be:
               prob_v = get_common(orig_mesh,ed[0],ed[1])
               #print(f" prob v {prob_v}")
               for pv in prob_v:
                   #print(f" pv {pv}")
                   if pv not in inner_loop:
                      #print(f" selected value {pv} to outer loop")
                      outer_loop.append(pv)
        
        v5,v6,v7,v8 = 0,0,0,0
       # print(f" fucking len {len(outer_loop)}") 
        #if len(outer_loop) == 0:
        #   continue
        #if len(outer_loop) == 1:
        #   v5 = outer_loop
        #   v6 = v5
        #   v7 = v5
        #   v8 = v6
        if len(outer_loop) == 2:
           v5,v6 = outer_loop
           v7 = v5
           v8 = v6
        elif len(outer_loop) == 3:
           v5,v6,v7 = outer_loop
           v8 = v6
        elif len(outer_loop) == 4:
           v5,v6,v7,v8 = outer_loop
        else:
          v5,v6,v7,v8 = outer_loop[:4]

        new_pos = ((8 * orig_mesh.vs[v1]) + (8 * orig_mesh.vs[v2]) + (2*orig_mesh.vs[v3]) + (2*orig_mesh.vs[v4]) \
			        -orig_mesh.vs[v5] -orig_mesh.vs[v6] -orig_mesh.vs[v7] -orig_mesh.vs[v8])/ 16
        mesh.vs[new_vertex_index] = new_pos

    return mesh

if __name__=="__main__":
	parser = argparse.ArgumentParser(description="Run subdivision")
	parser.add_argument("--input", "-i", default="../input/cube.obj", help="path to input .obj")
	parser.add_argument("-n", default=3, type=int, help="number of iterations to perform")
	args = parser.parse_args()

	inputfile = args.input
	number_of_iterations = args.n

	print(f"I parced arguments!\n input file: {inputfile}\n n_iterations: {number_of_iterations}\n")
	obj_name = inputfile.split('/')[-1]
	obj_name = obj_name.split('.')[0]

	mesh = TriMesh.FromOBJ_FileName(inputfile)

	os.makedirs("plots/", exist_ok=True)
	os.makedirs("output/", exist_ok=True)

	print("Saving a plot")
	mp.offline()
	p = mp.plot(mesh.vs, mesh.faces, c=mesh.vs[:,0], return_plot=True, filename='plots/test.html')

	for iteration in range(number_of_iterations):
		mesh = subdivision_method(mesh)

	#mesh.write_OBJ("output/test_output.obj")
    
	file_name = "output/"+ obj_name+ "_butterfly_"+str(number_of_iterations)+".obj"
	#mesh.write_OBJ("output/test_output.obj")
	mesh.write_OBJ(file_name)
	print("Done")
