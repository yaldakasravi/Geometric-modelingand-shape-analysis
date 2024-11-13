import argparse
import numpy as np
from trimesh import TriMesh
import meshplot as mp
import os


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
    #print(old_vert_count)
    #input()
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
    for i, vertex in enumerate(mesh.vs[:old_vert_count]):
        #mesh.get_halfedges()
        neighbors = orig_mesh.vertex_vertex_neighbors(i)
        n = len(neighbors)
        #print(n)
        w = ((64 * n) / (40 - (3 + 2 * np.cos(2 * np.pi / n))**2)) - n
        new_pos = (1 / (n + w)) * (w * vertex + sum([orig_mesh.vs[neighbor] for neighbor in neighbors]))
        mesh.vs[i] = new_pos
    
    # Step 4: Relocate new vertices inserted at the midpoints of the edges
    for (v1, v2), new_vertex_index in new_vertices.items():
        # Get adjacent faces sharing the edge v1v2
        #adjacent_faces = old_mesh.vertex_face_neighbors(v1)
        v1_neighbors = orig_mesh.vertex_vertex_neighbors(v1)
        v2_neighbors = orig_mesh.vertex_vertex_neighbors(v2)
        
        common = list(set(v1_neighbors).intersection(v2_neighbors))
        v3,v4 = common[0],common[1]
        #print(f" vertex {v1}, {v2} ")
        #print(f" current new index {new_vertex_index}")
        #print(f" {v1_neighbors}")
        #print(f" {v2_neighbors}")
        #print(f"common {common}")
        #print(f"4 vert {v1},{v2},{v3},{v4}")
        #input()
    #    f1, f2 = adjacent_faces[:2]  # Assume only two faces sharing the edge
    #    # Get the actual vertex sets for f1 and f2 from mesh.faces
    #    face1_vertices = set(mesh.faces[f1])
    #    face2_vertices = set(mesh.faces[f2])
    #    
    #    # Remove v1 and v2 to get the third vertices in each triangle
    #    v3 = face1_vertices.difference({v1, v2}).pop()
    #    v4 = face2_vertices.difference({v1, v2}).pop()

        new_pos = ((3 * orig_mesh.vs[v1]) + (3 * orig_mesh.vs[v2]) + orig_mesh.vs[v3] + orig_mesh.vs[v4]) / 8
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

	#file_name = "output/cube_loop_"+str(number_of_iterations)+".obj"
	file_name = "output/"+ obj_name+ "_butterfly_"+str(number_of_iterations)+".obj"
	#mesh.write_OBJ("output/test_output.obj")
	mesh.write_OBJ(file_name)
	print("Done")
