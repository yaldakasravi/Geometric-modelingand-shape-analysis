import numpy as np
import scipy
import scipy.sparse as sp
import scipy.sparse.linalg
import meshplot as mp
import igl
from matplotlib import cm

# Load the meshes
#bunny_vertices, bunny_faces = igl.read_triangle_mesh("../input/bunny_decimated.obj")
bunny_vertices, bunny_faces = igl.read_triangle_mesh("../input/bunny.obj")
#camel_mc_vertices, camel_mc_faces = igl.read_triangle_mesh("../input/camel_mc.obj")
#camel_1_vertices, camel_1_faces = igl.read_triangle_mesh("../input/camel-1.obj")
def is_diagonal_only(sparse_matrix):
    # Convert the matrix to COO format
    coo = sparse_matrix.tocoo()
    
    # Check if all non-zero elements are on the diagonal
    return (coo.row == coo.col).all()

#fucntion of compute normalized curvature for q3 as specified in the paper
def curvature_computed(vertices, faces):

    n_vertices = vertices.shape[0]
    areas = np.zeros(n_vertices)
    curves = np.zeros((n_vertices,3))
    
    for i, face in enumerate(faces):
        # Get vertex indices of the triangle
        i0, i1, i2 = face

        v0 = vertices[i0]
        v1 = vertices[i1]
        v2 = vertices[i2]

        # Calculate edges of the triangle
        e0 = v1 - v0
        e1 = v2 - v1
        e2 = v0 - v2
        
        len_e0 = np.linalg.norm(e0)
        len_e1 = np.linalg.norm(e1)
        len_e2 = np.linalg.norm(e2)
        #print(f" len of edge 0 {len_e0} len of edge 1 {len_e1}")
        #print(f" edge 0 {e0} , edge 1 {e1} ")
        
        theta_0 = np.arccos(np.dot(e0,-e2)/(len_e0*len_e2))
        theta_1 = np.arccos(np.dot(e1,-e0)/(len_e1*len_e0))
        theta_2 = np.arccos(np.dot(e2,-e1)/(len_e1*len_e2))

        #print(f" sum of anlge/pi = {(theta_0+theta_1+theta_2)/np.pi}")
        #input()
        alt_cot0 = 1.0/np.tan(theta_0) 
        alt_cot1 = 1.0/np.tan(theta_1) 
        alt_cot2 = 1.0/np.tan(theta_2) 

        #updates curves 
        for (cot, i,j,k) in [(alt_cot1 + alt_cot2, i0,i1,i2), (alt_cot0 + alt_cot2, i1,i0,i2), (alt_cot1 + alt_cot0, i2,i0,i1)]:
           curves[i] += ((cot* (vertices[i] - vertices[j])) + (cot* (vertices[i] - vertices[k]) ))/2
           areas[i] += cot

    K = curves[:,:] / areas[:,None]
    return K
# Function to calculate the cotangent Laplacian
def cotangent_laplacian(vertices, faces):
    n_vertices = vertices.shape[0]
    I = []  # row indices
    J = []  # column indices
    S = []  # values

    # Loop through each triangle and calculate contributions to the cotangent Laplacian
    vertices = np.array(vertices)
    v0 = vertices[0]
    v0 = v0.reshape(3)
    for i, face in enumerate(faces):
        # Get vertex indices of the triangle
        i0, i1, i2 = face

        # Get vertex positions
        v0 = vertices[i0]
        v1 = vertices[i1]
        v2 = vertices[i2]
        
        

        # Calculate edges of the triangle
        e0 = v1 - v0
        e1 = v2 - v1
        e2 = v0 - v2

        
        
        len_e0 = np.linalg.norm(e0)
        len_e1 = np.linalg.norm(e1)
        len_e2 = np.linalg.norm(e2)
        #print(f" len of edge 0 {len_e0} len of edge 1 {len_e1}")
        #print(f" edge 0 {e0} , edge 1 {e1} ")
        
        theta_0 = np.arccos(np.dot(e0,-e2)/(len_e0*len_e2))
        theta_1 = np.arccos(np.dot(e1,-e0)/(len_e1*len_e0))
        theta_2 = np.arccos(np.dot(e2,-e1)/(len_e1*len_e2))


        #alt_cot0 = abs(1.0/np.tan(theta_0) )
        #alt_cot1 = abs(1.0/np.tan(theta_1) )
        #alt_cot2 = abs(1.0/np.tan(theta_2) )
        alt_cot0 = (1.0/np.tan(theta_0) )
        alt_cot1 = (1.0/np.tan(theta_1) )
        alt_cot2 = (1.0/np.tan(theta_2) )
        # Calculate cotangents of angles using the formula: cot(theta) = (u . v) / ||u x v||
        #print(f" checkinmg between and old and new ")
        #print(alt_cot0)
        #print(alt_cot1)
        #print(alt_cot2)
        #print(cot01)
        #print(cot12)
        #print(cot20)
        #input()

        # Add contributions to Laplacian matrix
        for (cot, i, j) in [(alt_cot2, i0, i1), (alt_cot2, i1, i0),
                            (alt_cot0, i1, i2), (alt_cot0, i2, i1),
                            (alt_cot1, i2, i0), (alt_cot1, i0, i2)]:
            I.append(i)
            J.append(j)
            S.append(-cot / 2.0)
            #if (i0 == 0 or i1 == 0 or i2 == 0):
            #    print(f" puting val {cot} in indexes {i,j}")
            #    input()
        # Add diagonal entries
        for (cot, i) in [(alt_cot1 + alt_cot2, i0), (alt_cot0 + alt_cot2, i1), (alt_cot1 + alt_cot0, i2)]:
            I.append(i)
            J.append(i)
            S.append(cot / 2.0)

    # Create sparse matrix
    L = sp.coo_matrix((S, (I, J)), shape=(n_vertices, n_vertices)).tocsc()
    return L

def mass_matrix_D(vertices, faces):
    n_vertices = vertices.shape[0]
    M = sp.lil_matrix((n_vertices, n_vertices))

    # Loop through each triangle and calculate contributions to the mass matrix
    for face in faces:
        # Get vertex indices of the triangle
        i0, i1, i2 = face

        # Get vertex positions
        v0 = vertices[i0]
        v1 = vertices[i1]
        v2 = vertices[i2]

        # Calculate area of the triangle
        area = np.linalg.norm(np.cross(v1 - v0, v2 - v0)) / 2.0

        # Distribute area to the vertices (using the barycentric approach)
        for i,j,k in [(i0,i1,i2), (i1,i2,i0), (i2,i0,i1)]:
            M[i, j] += area / 12.0
            M[i, k] += area / 12.0
    
    M = M.todense() 

    diag = np.array(np.sum(M,axis=1))
    diag = diag.reshape((-1,))
    diag = np.diag(diag)

    M = M[:,:] + diag[:,:]
    #print(f"Is the M matrix sparse?", sp.issparse(M))
    #print(f"Is the M matrix symmetric?", np.allclose(M.todense(), M.T.todense()))
    #print(f"Sum of M each row :\n", M.sum(axis=1)) 

    return M

# Function to calculate the mass matrix
def mass_matrix(vertices, faces):
    n_vertices = vertices.shape[0]
    M = sp.lil_matrix((n_vertices, n_vertices))
    M = np.zeros((n_vertices,n_vertices))

    # Loop through each triangle and calculate contributions to the mass matrix
    for face in faces:
        # Get vertex indices of the triangle
        i0, i1, i2 = face

        # Get vertex positions
        v0 = vertices[i0]
        v1 = vertices[i1]
        v2 = vertices[i2]

        # Calculate area of the triangle
        area = np.linalg.norm(np.cross(v1 - v0, v2 - v0)) / 2.0

        # Distribute area to the vertices (using the barycentric approach)
        for i in [i0, i1, i2]:
            M[i,i] += abs(area) / 3.0
     
    #print(f"Is the M matrix sparse?", sp.issparse(M))
    print(f"Is the M matrix symmetric?", np.allclose(M, M.T))
    print(f"Sum of M each row :\n", M.sum(axis=1)) 

    return sp.csr_matrix(M)

# Function to compute and visualize eigenvectors
def visualize_eigenvectors(vertices, faces, mesh_name, num_eigenvectors=10):
    # Calculate the Laplacian
    L = cotangent_laplacian(vertices, faces)
    M = mass_matrix(vertices,faces)
    #L_dense = np.array(L.todense())
    #M_dense = np.array(M.todense())
    #print(f" L {L_dense}, M{M_dense}")
    #input()
    len_vert = len(vertices)-1
    len_vert = 10
    print(f" len of vertices {len_vert}")

    # Check basic properties
    print(f"{mesh_name} - Is the matrix sparse?", sp.issparse(L))
    print(f"{mesh_name} - Is the matrix symmetric?", np.allclose(L.todense(), L.T.todense()))
    print(f"{mesh_name} - Sum of each row (should be close to zero):\n", L.sum(axis=1))

    # Compute eigenfunctions
    #eigenvalues_d, eigenvectors_d = scipy.linalg.eig(a=L_dense, b=M_dense)
    eigenvalues, eigenvectors = sp.linalg.eigsh(L, k=len_vert, which='SM')
    print(f" {mesh_name} - min eigenvalue {min(eigenvalues)}")

    # Sort eigenvectors by eigenvalues_d
#    sorted_indices = np.argsort(eigenvalues_d)
#    eigenvalues_d = eigenvalues_d[sorted_indices]
#    eigenvectors_d = eigenvectors_d[:, sorted_indices]

    sorted_indices = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Show smallest and largest eigenvalues
    print(f"{mesh_name} - Smallest eigenvalues:", eigenvalues[:3])
    print(f"{mesh_name} - Largest eigenvalues:", eigenvalues[-3:])
    print(f"{mesh_name} - first and last eigenvalue {eigenvalues[0],eigenvalues[-1]}")
    #print(f"{mesh_name} - Smallest eigenvalues_d:", eigenvalues_d[:3])
    #print(f"{mesh_name} - Largest eigenvalues_d:", eigenvalues_d[-3:])
    #print(f"{mesh_name} - first and last eigenvalue {eigenvalues_d[0],eigenvalues_d[-1]}")
    #input()

    # Visualize eigenvectors corresponding to smallest and largest eigenvalues
    mp.offline()
    for i in range(4):
        p = mp.plot(vertices, faces, c=eigenvectors[:, i], return_plot=True, filename=f'plots/{mesh_name}_smallest_eigenvector_{i}.html')
    for i in range(1, 5):
        p = mp.plot(vertices, faces, c=eigenvectors[:, -i], return_plot=True, filename=f'plots/{mesh_name}_largest_eigenvector_{5-i}.html')

    print(f"Visualization saved to plots/ directory for {mesh_name}")

# Visualize eigenvectors on bunny, camel-mc, and camel-1 meshes
visualize_eigenvectors(bunny_vertices, bunny_faces, "bunny")
#visualize_eigenvectors(camel_mc_vertices, camel_mc_faces, "camel_mc")
#visualize_eigenvectors(camel_1_vertices, camel_1_faces, "camel-1")

# Calculate and verify the mass matrix for the bunny mesh
M = mass_matrix(bunny_vertices, bunny_faces)
print("Bunny - Mass matrix basic properties:")
print("Is the matrix sparse?", sp.issparse(M))
print("Mass matrix diagonal elements:", M.diagonal())

# Problem 2: Solve the heat equation on the mesh

#def solve_heat_equation(vertices, faces, num_timesteps=10, delta_t=0.0001, conductivity=1):
def solve_heat_equation(vertices, faces, num_timesteps=10, delta_t=0.1, conductivity=1):
    # Calculate Laplacian and Mass matrix
    L = cotangent_laplacian(vertices, faces)
    M = mass_matrix(vertices, faces)
    print(f" Is the matrix sparse?", sp.issparse(L))
    print(f" Is the matrix symmetric?", np.allclose(L.todense(), L.T.todense()))
    print(f" Sum of each row (should be close to zero):\n", L.sum(axis=1))
    L_dense = np.array(L.todense())
    M_dense = np.array(M.todense())

    #eig_vals, eigenvectors_d = scipy.linalg.eigh(a=L_dense, b=M_dense)
    eig_vals, eigenvectors_d = scipy.linalg.eigh(a=L_dense )
    norms = np.linalg.norm(eigenvectors_d,axis=1)
    eig_f = eigenvectors_d
    #eig_f = L_dense
    eig_f_inv = np.linalg.inv(eig_f)
   # print(f"max of eig_vals {max(eig_vals)}")
   # print(f" shape fo func {eig_f.shape}")
    # Initial condition: random temperature distribution
    #u_init = eigenvectors_d[:, 1]
    #u_init = 100 * (u_init - np.min(u_init)) / (np.max(u_init) - np.min(u_init)) 
    u_random = np.random.rand(vertices.shape[0]) * 100
    #u_random = np.ones(vertices.shape[0]) * 100
    u_random[:] = 1
    u_random[0] = 10
    u_init = u_random
    
    #A_0 = np.dot(u_init, eig_f_inv)
    #u_pred = np.dot(A_0,eig_f)
    #A_up_0 = A_0[:] * eig_vals[:]
    #print(f"U pred {u_pred[:2]}, u_init {u_init[:2]} ")
    #print(f" shape of A0 {A_0.shape}, shape of eig_vals {eig_vals.shape} shape of updated A {A_up_0.shape}")
    #print(f"A_0 {A_0[:2]}")
    #print(f"eig_vals {eig_vals[:2]}")
    #print(f"A_up_0 {A_up_0[:2]}")
    # Convert to sparse matrix format for computation
    mp.offline()
    # Time-stepping loop
    u = u_init
    time = 0
    for t in range(num_timesteps):

        A_0 = np.dot(u, eig_f)
        #A_up_0 = A_0[:] * eig_vals[:]
        mp.plot(vertices, faces, c=u, return_plot=True, filename=f'plots/git_heat_equation_step_{t}.html')
        time = delta_t 
        exp_term = np.exp(-eig_vals[:]*time)
        coeffs = A_0*exp_term
        alt_u = eig_f.dot(coeffs)
        u = alt_u
    #for t in range(num_timesteps):
    #    Q_sq = np.dot(eig_f,eig_f.T)
    #    #A_up_0 = A_0[:] * eig_vals[:]
    #    mp.plot(vertices, faces, c=u, return_plot=True, filename=f'plots/alt_heat_equation_step_{t}.html')
    #    time = delta_t 
    #    exp_term = np.exp(-eig_vals[:]*time)
    #    delta_u = np.dot(Q_sq,exp_term)
    #    print(f" shape of u {u.shape}, delta_u {delta_u.shape}")
    #    print(f"delta u  {delta_u[:2],delta_u[1070]}")
    #    print(f" u {u[0],u[1]}")
    #    u -= (delta_u *u[:])*delta_t
    #    print(f" updated u u {u[0],u[1],u[1070]}")
    #    input()
    #u = u_init
    #for t in range(num_timesteps):

    #    A_0 = np.dot(u, eig_f_inv)
    #    print(f" A 0 {A_0[:2]}")
    #    u_pred = np.dot(A_0,eig_f)
    #    print(f" upred {u_pred[:2]}")
    #    #A_up_0 = A_0[:] * eig_vals[:]
    #    mp.plot(vertices, faces, c=u, return_plot=True, filename=f'plots/new_heat_equation_step_{t}.html')
    #    time = delta_t 
    #    exp_term = np.exp(-eig_vals[:]*time)
    #    delta_u = np.zeros(u.shape)
    #    for i,du in enumerate(delta_u):
    #        delta_u[i] = np.sum( A_0[:] * eig_vals[:] * eig_f[:,i]*exp_term[:])

    #    alt_delta_u = A_0*eig_vals[:]*exp_term[:]
    #    print(f"delta u  {delta_u[:2],delta_u[1070]}")
    #    print(f"alt delta u  {alt_delta_u[:2],alt_delta_u[1070]}")
    #    print(f" u {u[0],u[1]}")
    #    print(f" indexes where temp is increasing {delta_u[delta_u < -0.001]}")
    #    u -= (delta_u * delta_t)
    #    print(f" updated u u {u[0],u[1],u[1070]}")
    #    input()
    #u = u_init
    #for t in range(num_timesteps):
    #    # Solve the linear system A * u_{j+1} = M * u_j
    #    #u = sp.linalg.spsolve(A, M @ u)
    #    
    #    A_0 = np.dot(u, eig_f_inv)
    #    print(f" A 0 {A_0[:2]}")
    #    u_pred = np.dot(A_0,eig_f)
    #    
    #    print(f" upred {u_pred[:2]}")
    #    #A_up_0 = A_0[:] * eig_vals[:]
    #    A_up_0 = A_0[:] 
    #    mp.plot(vertices, faces, c=u, return_plot=True, filename=f'plots/heat_equation_step_{t}.html')
    #    time = delta_t 
    #    exp_term = np.exp(-eig_vals[:]*time)
    #      
    #    comb = (A_up_0[:]*exp_term[:])
    #    delta_u = np.dot(comb, eig_f)
    #    print(f"delta u  {delta_u[:2]}")
    #    print(f" u {u[0],u[1]}")
    #    print(f" indexes where temp is increasing {delta_u[delta_u < -0.001]}")
    #    u -= (delta_u * delta_t)
    #    
    #    print(f" updated u u {u[0],u[1]}")
    #    #input()
        # Visualize temperature at each step

    # Visualize initial random temperature distribution
    #mp.plot(vertices, faces, c=u_random, return_plot=True, filename='plots/heat_equation_initial_random.html')
    mp.plot(vertices, faces, c=u, return_plot=True, filename='plots/heat_equation_initial_eigenvectors.html')
    # Visualize temperature after one large time step
    #u_large_step = sp.linalg.spsolve(A, M @ u_random)
    #mp.plot(vertices, faces, c=u_large_step, return_plot=True, filename='plots/heat_equation_one_large_step.html')

    print("Heat equation visualizations saved to plots/ directory")

# Solve the heat equation for the bunny mesh
#NOTE Q2
#solve_heat_equation(bunny_vertices, bunny_faces)

# Problem 3: Implicit fairing using diffusion

# Function to solve the diffusion equation using implicit integration
def solve_diffusion_equation(vertices, faces, num_timesteps=100, delta_t=5, lambda_=1e-6):
#def solve_diffusion_equation(vertices, faces, num_timesteps=10, delta_t=5, lambda_=1e-7):
    # Calculate Laplacian and Mass matrix
    L = cotangent_laplacian(vertices, faces)
    M = mass_matrix(vertices, faces)
    M_D = mass_matrix_D(vertices, faces)
    M = igl.massmatrix(bunny_vertices, bunny_faces)
    L_dense = L.todense()
    M_dense = M.todense() 
    #print(M_diag.shape)
    M_dense_inv = np.linalg.inv(6 * M_dense)
    M_D_inv = np.linalg.inv(M_D)
    #print((6*M_dense[0,0]),1/M_dense_inv[0,0])
    #input()
    #L_dense = np.dot(M_dense_inv,L_dense )
    #L_dense = L_dense[:,:]/M_diag[None,:]
    #L_dense = M_dense_inv @ L_dense
    #L_dense = np.linalg.inv(L_dense)
    eig_vals, eigenvectors_d = scipy.linalg.eigh(a=L_dense )
    L_dense = M_dense_inv @ L_dense
    #NOTE use below for Q3 bonus
    #L_dense = M_D_inv @ L_dense
    #L_dense = np.linalg.inv(L_dense)
    #L_dense = np.diag( L_diag[:]/M_diag[:])
    #print(f"non diagnal elements {np.count_nonzero(L_dense - np.diag(np.diagonal(L_dense)))}")

    sorted_indices = np.argsort(eig_vals)
    eig_vals = eig_vals[sorted_indices]
    eigenvectors_d = eigenvectors_d[:, sorted_indices]
    n_vertices = vertices.shape[0]

    I = sp.eye(n_vertices)
    #K = curvature_computed(vertices, faces)
    #print(f"shape of K {K.shape}")
    
    #K_alt = np.dot(K,K.T)
    #K_norm = np.linalg.norm(K,axis=1)
    # Construct the matrix A = (I - lambda * delta_t * L)
    #A = I - lambda_ * delta_t * K_alt
    #A = I - lambda_ * delta_t * np.diag(K_norm)
    #for scale in np.linspace(0,1e-7,10):
    #    print(f" scale {scale}")
    #    sanity_matrix = scale * L_dense
    #    print(f" norm of sanity {np.linalg.norm(sanity_matrix)}")
    #    A_alt = np.linalg.inv( I - sanity_matrix)
    #    correction = np.dot(sanity_matrix,vertices)
    #    corrected = np.dot(A_alt,vertices)
    #    print(f"correction max and min {np.max(correction),np.min(correction)}")
    #    print(f"inverse A max and min {np.max(A_alt),np.min(A_alt)}")

    A = I - lambda_ * delta_t * L_dense
    #print(f"non diagnal elements for A {np.count_nonzero(A - np.diag(np.diagonal(A)))}")
    #input()


    # Initial condition: random temperature distribution
    #u = np.random.rand(n_vertices)
    A_inv = np.linalg.inv(A)
    A_inv_diag = np.diag(A_inv)

    # Pre-factorize the matrix A using an efficient solver (PBCG or equivalent)
    #factor = sp.linalg.factorized(A)
    print(f"A_inv max and min {np.max(A_inv_diag),np.min(A_inv_diag)}, ")
    print(f"vert {vertices.shape}, ")
    # Time-stepping loop
    for t in range(num_timesteps):
        # Solve the linear system: A * u_{t+1} = u_t
        #vertices = np.dot(A_inv,vertices)
        vertices = np.dot(A,vertices)
        # Visualize temperature at each step
        mp.plot(vertices, faces, c=eigenvectors_d[:,0], return_plot=True, filename=f'plots/diffusion_step_{t}.html')
        print(f"Step {t}: Visualization saved to plots/diffusion_step_{t}.html")

    print("Diffusion process completed and visualizations saved.")

def solve_diffusion_equation_bonus(vertices, faces, num_timesteps=100, delta_t=5, lambda_=1e-6):
#def solve_diffusion_equation(vertices, faces, num_timesteps=10, delta_t=5, lambda_=1e-7):
    # Calculate Laplacian and Mass matrix
    for t in range(num_timesteps):
    	M_D = mass_matrix_D(np.array(vertices), faces)
    	M_D = igl.massmatrix(bunny_vertices, bunny_faces)
    	M_dense = M.todense() 
    	#print(M_diag.shape)
    	M_dense_inv = np.linalg.inv(6 * M_dense)

    	M_D_inv = np.linalg.inv(M_D)
    	L = cotangent_laplacian(vertices, faces)
    	L_dense = L.todense()
    	#L_dense = M_D_inv @ L_dense
    	eig_vals, eigenvectors_d = scipy.linalg.eigh(a=L_dense )
    	L_dense = M_dense_inv @ L_dense
    	#L_dense = np.linalg.inv(L_dense)
    	#L_dense = np.diag( L_diag[:]/M_diag[:])
    	#print(f"non diagnal elements {np.count_nonzero(L_dense - np.diag(np.diagonal(L_dense)))}")

    	sorted_indices = np.argsort(eig_vals)
    	eig_vals = eig_vals[sorted_indices]
    	eigenvectors_d = eigenvectors_d[:, sorted_indices]
    	n_vertices = vertices.shape[0]

    	I = sp.eye(n_vertices)
    	A = I - lambda_ * delta_t * L_dense

    # Pre-factorize the matrix A using an efficient solver (PBCG or equivalent)
    #factor = sp.linalg.factorized(A)
        # Solve the linear system: A * u_{t+1} = u_t
        #vertices = np.dot(A_inv,vertices)
    	vertices = np.dot(A,vertices)
        # Visualize temperature at each step
    	mp.plot(vertices, faces, c=eigenvectors_d[:,0], return_plot=True, filename=f'plots/diffusion_bonus_step_{t}.html')
    	print(f"Step {t}: bonus stuff Visualization saved to plots/diffusion_step_{t}.html")

    print("Diffusion process completed and visualizations saved.")
# Load the mesh (replace with the appropriate file path)
bunny_vertices, bunny_faces = igl.read_triangle_mesh("../input/bunny.obj")

# Solve the diffusion equation for the bunny mesh
#NOTE Q3 part 1
solve_diffusion_equation(bunny_vertices, bunny_faces)
#solve_diffusion_equation_bonus(bunny_vertices, bunny_faces)

