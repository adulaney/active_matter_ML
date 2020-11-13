"""
Functions pertaining to feature generation, computation, or augmentation.
"""
# General
import numpy as np
import pandas as pd

# Compute / HOOMD Specific
import gsd.hoomd
import freud
import rowan
import networkx as nx
from scipy.interpolate import Rbf

##########################
### Feature Generation ###
##########################


def compute_trajectory(gsd_traj, full_traj=False, graphs=False):
    list_frames = []
    if full_traj == True:
        for frame in gsd_traj:
            frame_features = compute_features(frame, graphs=graphs)
            list_frames.append((frame.configuration.step, frame_features))
    else:
        frame = gsd_traj[len(gsd_traj) - 1]
        frame_features = compute_features(frame, graphs=graphs)
        list_frames.append((frame.configuration.step, frame_features))

    trajectory_frame = pd.DataFrame(
        list_frames, columns=["Time Step", "Features"])

    return trajectory_frame


def compute_features(traj, graphs=True):
    box_data = traj.configuration.box
    pos_data = traj.particles.position
    quat_data = traj.particles.orientation

    fbox = freud.box.Box(box_data[0], box_data[1], is2D=True)

    return compute_features_frame(traj, fbox, pos_data, quat_data, graphs=graphs)


def compute_features_frame(traj, fbox, pos_data, quat_data, graphs=True):
    # Initialize Freud components

    voro = freud.locality.Voronoi(fbox, np.max(fbox.L) / 2, is2D=True)
    voro.compute(traj)
    nlist = voro.nlist

    # Get 1st neighbor shell of each particle
    neigh1 = first_neighbors(nlist)

    # Get 2nd neighbor shell of each particle
    neigh2 = second_neighbors(neigh1)

    # Get 3rd neighbor shell of each particle
    neigh3 = third_neighbors(neigh1, neigh2)

    # Number of Neighbors for shells
    nn1 = [len(neigh) for neigh in neigh1]
    nn2 = [len(neigh) for neigh in neigh2]
    nn3 = [len(neigh) for neigh in neigh3]

    # Disk volume
    vp1 = np.pi * (0.5 * 2 ** (1.0 / 6.0)) ** 2

    # Compute Volumes
    volumes = voro.volumes
    vf = vp1 / volumes
    vf1 = [
        (np.sum(vf[neigh1[i]]) + vf[i]) / (len(neigh1[i]) + 1.0)
        for i in range(len(neigh1))
    ]
    vf2 = [
        (np.sum(vf[neigh2[i]]) + vf[i]) / (len(neigh2[i]) + 1.0)
        for i in range(len(neigh2))
    ]
    vf3 = [
        (np.sum(vf[neigh3[i]]) + vf[i]) / (len(neigh3[i]) + 1.0)
        for i in range(len(neigh3))
    ]

    # Compute hexatic order parameter
    psi6 = hexatic_features(fbox, pos_data)

    # Compute translational order parameter
    dr = trans_order_features(fbox, pos_data)

    # Compute orientations
    q = orientation(quat_data)

    # Compute net force
    net_force = net_force_calc(fbox, pos_data, q)

    # Compute particle speed
    speed = particle_speed(net_force)

    # Compute Force - Orientation correlation function
    F0 = force_orientation_correlation(net_force, q, neigh1)

    # dict for panda frame
    frame = {
        "Volume Fraction": vf,
        "1st Volume Fraction": vf1,
        "2nd Volume Fraction": vf2,
        "3rd Volume Fraction": vf3,
        "1st Neighbors": nn1,
        "2nd Neighbors": nn2,
        "3rd Neighbors": nn3,
        "Hexatic Order (R)": psi6.real,
        "Hexatic Order (I)": psi6.imag,
        "Hexatic Order (ABS)": np.abs(psi6),
        "Hexatic Order (ANG)": np.angle(psi6),
        "Translational Order (R)": dr.real,
        "Translational Order (I)": dr.imag,
        "Translational Order (ABS)": np.abs(dr),
        "Translational Order (ANG)": np.angle(dr),
        "Speed": speed,
        "Force Orientation Correlation": F0,
    }

    if graphs == True:
        G = generate_graphs(nlist)
        frame['Graph'] = G

    return pd.DataFrame(frame)

############################
### Feature Calculations ###
############################

### Neighbors ###


def first_neighbors(nlist):
    """ Computes 1st neighbor shell of each particle.
    Args:
        nlist (freud.nlist): List of voronoi neighbors
    Returns:
        list [np.array]: List of neighbor indices for each particle
    """
    neigh1 = []
    for i in np.unique(nlist[:, 0]):
        neigh_inds = np.argwhere(nlist[:, 0] == i)[:, 0]
        neigh = nlist[neigh_inds, 1]
        neigh1.append(neigh)

    return neigh1


def second_neighbors(neigh1):
    """ Computes 1st and 2nd shell neighbors of each particle.
    Args:
        neigh1 (list[np.array]): List of voronoi neighbors
    Returns:
        list[np.array]: List of 1st and 2nd shell neighbor indices for each particle
    """
    neigh2 = []
    for i, neigh in enumerate(neigh1):
        neighs = neigh1[i]
        for n in neigh:
            neighs = np.append(neighs, neigh1[n])
        neighs = np.unique(neighs)
        neighs = np.delete(neighs, np.argwhere(neighs == i))
        neigh2.append(neighs)

    return neigh2


def third_neighbors(neigh1, neigh2):
    """ Finds first 3 shell neighbors of each particle.
    Args:
        neigh1 (list[np.array]): List of voronoi neighbors
        neigh2 (list[np.array]): List of first 2 shell voronoi neighbors
    Returns:
        list[np.array]: List of first 3 shell neighbor indices for each particle
    """
    neigh3 = []
    for i, neigh in enumerate(neigh2):
        neighs = neigh2[i]
        for n in neigh:
            neighs = np.append(neighs, neigh1[n])
        neighs = np.unique(neighs)
        neighs = np.delete(neighs, np.argwhere(neighs == i))
        neigh3.append(neighs)

    return neigh3


### Structural Order Parameters ###
# Compute hexatic order parameter features
def hexatic_features(fbox, pos_data):
    hex_order = freud.order.Hexatic(k=6)
    hex_order.compute(system=(fbox, pos_data), neighbors={'r_max': 1.3})
    return hex_order.particle_order


# Compute translational order parameter features
def trans_order_features(fbox, pos_data):
    trans_order = freud.order.Translational(k=6)
    trans_order.compute(system=(fbox, pos_data), neighbors={'r_max': 1.3})
    return trans_order.particle_order

### Speed Feature ###
# Compute Force - Orientation correlation function


def force_orientation_correlation(net_force, q, neighbor):
    return [np.dot(q[i], net_force[i]) for i in range(len(neighbor))]

# Calculate orientations


def orientation(quat_data):
    return rowan.rotate(quat_data, np.array([1, 0, 0]))

# Calculates particle velocity


def particle_speed(net_force, drag=1):
    velocity = net_force / drag
    return [np.sqrt(np.dot(velocity[i], velocity[i])) for i in range(len(velocity))]

# Compute net force


def net_force_calc(fbox, pos_data, q):
    buffers, nlist_contact = contact_neighbors(fbox, pos_data)
    lj_f = net_ljf(pos_data, nlist_contact, fbox, buffers)
    act_f = act_force(q)
    return lj_f + act_f

# Calculates net force from all neighboring LJ (WCA) particle interactions


def net_ljf(pos_data, nlist, fbox, buffer):
    n_ljf = np.zeros(pos_data.shape)
    for i in range(pos_data.shape[0]):
        # neighs = nlist.index_j[nlist.index_i == i]
        neigh_inds = np.argwhere(nlist[:, 0] == i)[:, 0]
        neighs = nlist[neigh_inds, 1]
        if len(neighs) != 0:
            n_ljf[i][:] = np.sum(
                [ljf(pos_data[i], pos_data[j], j, fbox, buffer) for j in neighs], axis=0
            )
    return n_ljf

# Calculates LJ (WCA) force between two particles


def ljf(pos, pos1, id1, fbox, buffer, eps=100):
    # Handle wrapped particles
    if (np.abs(pos - pos1) > fbox.L / 2).any():
        pos1 = buffer.buffer_points[buffer.buffer_ids == id1]
        dists = pos - pos1
        dist = dists[np.where(dists[:, 0] ** 2 + dists[:, 1] ** 2 < 4)[0]][0]
    else:
        dist = pos - pos1

    lj = (
        48
        * eps
        / np.dot(dist, dist)
        * (np.dot(dist, dist) ** (-6) - np.dot(dist, dist) ** (-3) / 2)
        * dist
    )
    return lj

# Compute number of contact neighbors


def contact_neighbors(fbox, pos_data, r_max=2.0 ** (1.0 / 6.0)):
    lc = freud.locality.LinkCell(fbox, pos_data, r_max)
    nlist_contact = lc.query(
        pos_data, {'r_max': r_max, 'exclude_ii': True}).toNeighborList()
    buffers = freud.locality.PeriodicBuffer()
    buffers.compute(system=(fbox, pos_data), buffer=r_max)
    return buffers, nlist_contact

# Calculates active force on particles


def act_force(q, U=1, drag=1):
    return U * drag * q


### Orientation - Density Gradient Correlation ###
# Calculate density gradient
def density_gradient(pos_data, dens, neigh, buffer, fbox, del_x=3):
    dn = np.zeros((len(pos_data), 3))

    for i, p in enumerate(pos_data):
        inds = [i] + neigh[i]

        pos = np.zeros((1, 3), dtype=np.float32)
        pos[0] = p

        for j in neigh[i]:
            if (np.abs(p - pos_data[j]) > fbox.L / 2).any():
                pos1 = buffer.buffer_particles[buffer.buffer_ids == j]
                dists = p - pos1
                pos1 = pos1[
                    np.where(np.sqrt(dists[:, 0] ** 2 +
                                     dists[:, 1] ** 2) < 13)[0]
                ]
                if pos1.shape[0] == 0:
                    pos1 = pos_data[j]
            else:
                pos1 = pos_data[j]
            #             print(pos1)
            pos = np.concatenate((pos, pos1.reshape(1, 3)), axis=0)

        rbf = Rbf(pos[:, 0], pos[:, 1], dens[inds])
        dn[i][0] = (rbf(p[0] + del_x, p[1]) -
                    rbf(p[0] - del_x, p[1])) / (2 * del_x)
        dn[i][1] = (rbf(p[0], p[1] + del_x) -
                    rbf(p[0], p[1] - del_x)) / (2 * del_x)
        del rbf

    return dn


### Graph Structure ###
def generate_graphs(nlist):
    G = nx.Graph()

    for k, (i, j) in enumerate(nlist):
        G.add_edge(i, j, weight=nlist.distance[k])

    return G

############################
### Feature Augmentation ###
############################


def aggregate_features(df, G, feat_list=None):
    if feat_list is None:
        return df

    df_agg = pd.DataFrame()
    for node in list(G.adjacency()):
        cur_ind = node[0]
        n_list = list(node[1].keys()) + [cur_ind]

        df_new = df[feat_list].iloc[n_list].mean(axis=0).to_frame().T
        df_new.columns = ['agg ' + col for col in df_new.columns]
        df_new.index = [cur_ind]
        df_agg = df_agg.append(df_new)

    return df[feat_list].merge(df_agg, left_index=True, right_index=True)


def construct_graphs(sim_file, save_path=None, save=False):
    if save == True:
        # Check if file exists and verify save path
        if save_path == None:
            try:
                save_path = sim_file.replace(
                    '.gsd', '.pkl').replace('simulations', 'graphs')
            except:
                try:
                    base_file = './'+sim_file.split('/')[-1]
                except:
                    save_path = './'+sim_file
        if os.path.isfile(save_path) is True:
            print('This file exists.')
            return
    # Open Sim file
    traj_file = gsd.hoomd.open(sim_file, mode='rb')
    list_frames = []

    # Iterate through Frames
    for frame in traj_file:
        pos_data = frame.particles.position
        box_data = frame.configuration.box
        fbox = freud.box.Box(box_data[0], box_data[1], is2D=True)

        # Compute Voronoi cells and neighbors #
        voro = freud.locality.Voronoi(fbox, np.max(fbox.L)/2)
        voro.compute(frame)
        nlist = voro.nlist

        # Construct graph
        G = nx.Graph()
        for k, (i, j) in enumerate(nlist[:]):
            G.add_edge(i, j, weight=nlist.distances[k])
        # Add to frame list
        list_frames.append((frame.configuration.step, G))

    # Construct and save df
    df = pd.DataFrame(list_frames, columns=['Time Step', "Features"])
    if save == True:
        df.to_pickle(save_path)

    return df
