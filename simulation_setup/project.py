import pickle
import signac
from flow import FlowProject, directives
import environment
import os, sys
import numpy as np
import math
import importlib
import itertools
import gsd
import gsd.hoomd
import random
import hoobas


try:
    here = os.path.dirname(__file__)
except NameError:
    here = os.getcwd()
sys.path.append(os.path.join(here, '..'))

project = signac.get_project()
project_path = os.getcwd()
with open(os.path.join(project_path, 'project_metadata.pickle'), 'rb') as file:
    project_metadata = pickle.load(file)

amino_threeName = ['Ile', 'Val', 'Leu', 'Phe', 'Cys', 'Met', 'Ala', 'Gly',
                   'Thr', 'Ser', 'Trp', 'Tyr', 'Pro', 'His', 'Asn', 'Asp',
                   'Gln', 'Glu', 'Lys', 'Arg']
amino_classes = ['I', 'V', 'L', 'F', 'C', 'M', 'A', 'G', 'T', 'S', 'W', 'Y',
                 'P', 'H', 'N', 'D', 'Q', 'E', 'K', 'R']
plum_epsilon = [.84, .65, 1.0, .97, .54, .67, .26, .17, .16, .11, .64, .49, .14,
                .25, .10, .06, .13, .05, 0.0, .13]
eps_types = {
    'hp': 4.5,
    'hb': 6.0,
    'bb': 0.02
}

num_chains = 64

class Project(FlowProject):
    pass


file_names = project_metadata['filenames']

@Project.post.isfile('job_metadata.pickle')
@Project.post.isfile(file_names['init_single'])
@Project.operation(directives={"tasks_per_node": 1, "ngpu": 1, "nodes": 1, "cpu_per_task": 24})
#@Project.operation
def generate_single_job_metadata(job):
    import hoobas
    import hoomd
    import numpy as np
    import pickle
    import copy

    metadata = {}

    # Retrieve the sequence information based on job metadata
    sequence_name = project_metadata['fragment_names'][job.sp.fragmentID]
    chain_sequence = project_metadata['fragment_sequences'][sequence_name]

    # Create the base chain object for the simulation
    base_chain = hoobas.LinearChain.PlumProteinSequence([x for x in chain_sequence], kuhn_length=1.45)
    base_chain.randomize_dirs()
    
    # Set the box size to contain all chains
    box_size = 32000.0
     

    # Initialize an empty simulation box
    hoomdBox = hoobas.SimulationDomain.EmptyBox(box_size)
    baseInit = hoobas.Build.HOOMDBuilder(hoomdBox)
    
    # Calculate the number of chains per side for a 3D grid
    grid_side = int(np.ceil(np.cbrt(num_chains)))  # Cube root to determine the 3D grid layout
    spacing_x = box_size / grid_side
    spacing_y = box_size / grid_side
    spacing_z = box_size / grid_side
    # Generate positions for chains in a 3D grid layout centered in the box
    positions = []
    for i in range(grid_side):
    	for j in range(grid_side):
    		for k in range(grid_side):
    			if len(positions) < num_chains:  # Only add positions up to num_chains
    				x = (i + 0.5) * spacing_x
    				y = (j + 0.5) * spacing_y
    				z = (k + 0.5) * spacing_z
    				positions.append(np.array([x, y, z]))

    
    # Calculate the number of rows and columns for a square grid layout
#    grid_side = int(np.ceil(np.sqrt(num_chains)))  # Determines the grid layout based on chain count
#    spacing_x = box_size / grid_side
#    spacing_y = box_size / grid_side

    # Generate positions for chains in a 2D grid layout centered in the box
#    positions = []
#    for i in range(grid_side):
#        for j in range(grid_side):
#            if len(positions) < num_chains:  # Only add positions up to num_chains
#                x = (i + 0.5) * spacing_x
#                y = (j + 0.5) * spacing_y
#                z = box_size / 2  # Position in the center of the box along z-axis
#                positions.append(np.array([x, y, z]))

    # Modified function to add objects at specified positions without random rotation
    def add_N_ext_obj(builder, ext_obj, N, positions=None):
        """
        Adds N composite objects to the system at specified positions, without random rotation.

        :param builder: builder to add objects to
        :param ext_obj: external object to be copied
        :param N: number of objects
        :param positions: Optional list of specific positions for each object (length should match N)
        :return: None
        """
        L = builder.current_box()  # Retrieve the current box dimensions
        for n in range(N):
            composite_copy = copy.deepcopy(ext_obj)  # Deep copy of the object to add
            composite_copy.beads.relink()  # Restore bead linkage for integrity

            if hasattr(composite_copy, 'do_on_copy') and callable(getattr(composite_copy, 'do_on_copy')):
                composite_copy.do_on_copy()  # Call any copy-specific behavior

            # Assign a specified position if provided
            if positions and n < len(positions):
                composite_copy.center_position = positions[n]
            else:
                # Default to the center if no positions are specified
                composite_copy.center_position = np.array([0.0, 0.0, L[2] / 2.0])

            # Merge the object into the simulation without rotation
            builder.merge(composite_copy)

    # Use add_N_ext_obj function with calculated positions to add chains
    for i in range(num_chains):
        add_N_ext_obj(baseInit, base_chain, 1, positions=[positions[i]])

    baseInit.build_finalize()

    # Create an initial snapshot from the chain configuration
    sn = hoomd.snapshot.Snapshot()
    baseInit.set_snapshot(sn)

    # Initialize the HOOMD simulation on CPU
    dev = hoomd.device.CPU()
    sim = hoomd.simulation.Simulation(device=dev)
    sim.create_state_from_snapshot(sn)

    # Write the initial configuration to a GSD file
    gsd_writer = hoomd.write.GSD(filename=job.fn(file_names['init_single']), 
                                 trigger=hoomd.trigger.Periodic(1),
                                 filter=hoomd.filter.All())
    sim.operations.writers.append(gsd_writer)

    # Thermalize particle momenta for the initial state
    sim.state.thermalize_particle_momenta(hoomd.filter.All(), kT=1.0)
    sim.run(1)

    # Collect metadata including chain IDs and store positions
    metadata.update({
        'name': sequence_name,
        'raw_sequence': chain_sequence,
        'bead_types': baseInit.bead_types,
        'chain_positions': positions  # Store chain positions in metadata
    })
    
    # Save metadata to a pickle file for later retrieval
    with open(job.fn('job_metadata.pickle'), 'wb') as file:
        pickle.dump(metadata, file)

def get_pair_params(A, B):
    params = {'LJ': {}, 'WCA': {}}

    def getEpsilonLJ(_A):
        return plum_epsilon[amino_classes.index(_A)]

    def getSigma(_A):
        if _A in amino_classes:
            return 5.0
        if _A == 'Nbb' or _A == 'Nbbp':
            return 2.9
        if _A == 'Ca':
            return 3.7
        if _A == 'Cp':
            return 3.5

    if A in amino_classes and B in amino_classes:
        params['LJ'] = {
            'epsilon': math.sqrt(getEpsilonLJ(A) * getEpsilonLJ(B)) * eps_types[
                'hp'],
            'sigma': 5.0
        }
        params.update({'LJ_rc': 2.5 * 5.0})

        params['WCA'] = {
            'epsilon': (1 - math.sqrt(getEpsilonLJ(A) * getEpsilonLJ(B))) *
                       eps_types['hp'],
            'sigma': 5.0}
        params.update({'WCA_rc': 2 ** (1 / 6) * 5.0})
    else:
        mean_sigma = 0.5 * (getSigma(A) + getSigma(B))
        params['LJ'] = {'epsilon': 0.0, 'sigma': 0.0}
        params.update({'LJ_rc': 0.0})
        params['WCA'] = {'epsilon': eps_types['bb'], 'sigma': mean_sigma}
        params.update({'WCA_rc': 2 ** (1 / 6) * mean_sigma})
    return params


def apply_bonded_ff(forces):
    bonded = forces['bond']
    bonded.params['N-Ca'] = dict(k=300., r0=1.455)
    bonded.params['Ca-Cp'] = dict(k=300., r0=1.51)
    bonded.params['Cp-N'] = dict(k=300., r0=1.325)
    bonded.params['Ca-Cb'] = dict(k=300., r0=1.530)

    angle = forces['angle']
    angle.params['N-Ca-Cb'] = dict(k=300, t0=math.radians(108.0))
    angle.params['Cb-Ca-Cp'] = dict(k=300, t0=math.radians(113.0))
    angle.params['N-Ca-Cp'] = dict(k=300, t0=math.radians(111.0))
    angle.params['Ca-Cp-N'] = dict(k=300, t0=math.radians(116.0))
    angle.params['Cp-N-Ca'] = dict(k=300, t0=math.radians(122.0))

    dihedral = forces['dihedral']
    dihedral.params['phi'] = dict(k=-0.3 * 2.0, d=-1, n=1, phi0=0)
    dihedral.params['psi'] = dict(k=-0.3 * 2.0, d=-1, n=1, phi0=0)
    dihedral.params['omega'] = dict(k=67.0 * 2.0, d=-1, n=1,
                                    phi0=math.radians(180.0))
    dihedral.params['omegaP'] = dict(k=3.0 * 2.0, d=-1, n=2, phi0=0)
    dihedral.params['improper'] = dict(k=17.0 * 2.0, d=-1, n=1,
                                       phi0=math.radians(-120.0))




@Project.pre.isfile('job_metadata.pickle')
@Project.pre.isfile(file_names['init_single'])
@Project.operation(directives={"tasks_per_node": 2, "ngpu": 2, "nodes": 6, "cpu_per_task": 24})
def run_simulation(job):
    # Imports
    import hoomd
    import hoomd.md
    import hoomd.plugin_plum
    from mpi4py import MPI
    import numpy as np
    import math
    import random
    import pickle
    import itertools
    import os

    def metropolis_criterion(delta_energy):
        if delta_energy < 0:
            return True, 1.0
        else:
            boltzmann_factor = math.exp(-delta_energy)
            acceptance_prob = min(1, boltzmann_factor)
            return random.random() < acceptance_prob, acceptance_prob

    # MPI setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    bias_array = np.linspace(0.75, 1.0, size)
    bias = bias_array[rank]
    all_bias = comm.gather(bias, root=0)

    # Trajectory and restart setup
    trajectory_name = job.fn('trajectory_bias_' + str(bias) + '.gsd')
    is_restart = os.path.exists(trajectory_name)

    # HOOMD setup
    hoomd_comm = hoomd.communicator.Communicator()
    dev = hoomd.device.GPU(communicator=hoomd_comm)
    sim = hoomd.simulation.Simulation(device=dev)

    if is_restart:
        sim.create_state_from_gsd(trajectory_name)
    else:
        sim.create_state_from_gsd(job.fn(file_names['init_single']))

    # Particle filtering
    particles_length = sim.state.get_snapshot().particles.N
    irange = [x for x in range(particles_length) if x % (particles_length // num_chains) != 0]
    iselector = hoomd.filter.Tags(irange)

    # Zero velocities of particles excluded by `iselector`
    snapshot = sim.state.get_snapshot()
    if snapshot.particles.position.shape[0] > 0:
        snapshot.particles.velocity[~np.isin(range(snapshot.particles.N), irange)] = 0.0
    sim.state.set_snapshot(snapshot)

    # Neighbor list setup
    neighbor_list = hoomd.md.nlist.Tree(buffer=10.0, exclusions=('bond', '1-3'))

    # Bonded forces
    bonded_forces = {
        'bond': hoomd.md.bond.Harmonic(),
        'angle': hoomd.md.angle.Harmonic(),
        'dihedral': hoomd.md.dihedral.Periodic()
    }
    apply_bonded_ff(bonded_forces)

    # Nonbonded forces
    nonbonded_forces = {
        'LJ': hoomd.md.pair.LJ(neighbor_list, mode='shift'),
        'WCA': hoomd.md.pair.LJ(neighbor_list, mode='shift'),
        'H-Bond': hoomd.plugin_plum.plum.PlumHBonds(neighbor_list)
    }

    # Load metadata
    with open(job.fn('job_metadata.pickle'), 'rb') as file:
        job_metadata = pickle.load(file)
    bead_types = job_metadata['bead_types']

    # Pair force parameters
    for A in bead_types:
        for B in bead_types:
            params = get_pair_params(A, B)
            nonbonded_forces['LJ'].params[(A, B)] = params['LJ']
            nonbonded_forces['LJ'].r_cut[(A, B)] = params['LJ_rc']
            nonbonded_forces['WCA'].params[(A, B)] = params['WCA']
            nonbonded_forces['WCA'].r_cut[(A, B)] = params['WCA_rc']
            if A == 'Nbb' and B == 'Cp':
                nonbonded_forces['H-Bond'].r_cut = 2.5 * 4.11
            else:
                nonbonded_forces['H-Bond'].r_cut = 0.0

    nonbonded_forces['H-Bond'].epsilon = eps_types['hb'] * bias
    nonbonded_forces['H-Bond'].sigma = 4.11
    nonbonded_forces['H-Bond'].r_cut = 2.5 * 4.11
    nonbonded_forces['H-Bond'].N_name = 'Nbb'
    nonbonded_forces['H-Bond'].Ca_name = 'Ca'
    nonbonded_forces['H-Bond'].Cp_name = 'Cp'

    # Integration setup
    idt = 3.4 / 340.0
    integrator = hoomd.md.Integrator(dt=idt)
    sim.operations.integrator = integrator
    for _, f in itertools.chain(bonded_forces.items(), nonbonded_forces.items()):
        integrator.forces.append(f)

    nve = hoomd.md.methods.DisplacementCapped(filter=iselector, maximum_displacement=0.02)
    langevin = hoomd.md.methods.Langevin(filter=iselector, kT=1.0, default_gamma= 0.01, default_gamma_r=(0.01, 0.01, 0.01))

    trajectory_writer = hoomd.write.GSD(filename=trajectory_name,
                                        trigger=hoomd.trigger.Periodic(100000),
                                        filter=hoomd.filter.All())
    sim.operations.writers.append(trajectory_writer)

    thermo = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())

    logger = hoomd.logging.Logger(categories=['scalar'])
    logger.add(sim, ['timestep', 'tps'])
    logger[('Temperature', '')] = (lambda: thermo.kinetic_temperature, 'scalar')
    logger[('Kinetic_Energy', '')] = (lambda: thermo.kinetic_energy, 'scalar')
    logger[('Total_Energy', '')] = (lambda: thermo.potential_energy + thermo.kinetic_energy, 'scalar')
    for name, force in bonded_forces.items():
        logger[(f'{name}_Potential', '')] = (lambda force=force: force.energy, 'scalar')
    for name, force in nonbonded_forces.items():
        logger[(f'{name}_Potential', '')] = (lambda force=force: force.energy, 'scalar')

    log_io = open(job.fn('log_bias_' + str(bias) + '.log'), 'at')
    table_writer = hoomd.write.Table(trigger=hoomd.trigger.Periodic(250000),
                                     logger=logger,
                                     output=log_io,
                                     pretty=True)

    sim.operations.computes.append(thermo)
    sim.operations.writers.append(table_writer)

    # ADDED: Exchange rate log file
    exchange_log_file = job.fn('replica_exchange_rate.log')
    if rank == 0:
        with open(exchange_log_file, 'w') as f:
            f.write("Timestep\tReplica_i\tReplica_i+1\tSwap_Attempts\tAccepted_Swaps\tAcceptance_Rate\n")

    # Restart logic
    if is_restart:
        integrator.methods = [nve]
        for dt in [0.001, 0.01, idt]:
            integrator.dt = dt
            sim.run(10000)
    else:
        integrator.methods = [nve]
        sim.run(80000)

        langevin = hoomd.md.methods.Langevin(filter=iselector, kT=1.0, default_gamma=10.0, default_gamma_r=(10.0, 10.0, 10.0))
        integrator.methods = [langevin]
        sim.run(90000)

        langevin = hoomd.md.methods.Langevin(filter=iselector, kT=1.0, default_gamma=0.1, default_gamma_r=(0.1, 0.1, 0.1))
        integrator.methods = [langevin]
        sim.run(100000)


    move = 0
    hb_energy = nonbonded_forces['H-Bond'].energy
    all_hb_energies = comm.gather(hb_energy, root=0)
    all_positions = comm.gather(sim.state.get_snapshot().particles.position, root=0)
    all_momenta = comm.gather(sim.state.get_snapshot().particles.velocity, root=0)

    integrator.methods = [langevin]
    while sim.timestep < 2.1e9:
        sim.run(1e5)

        move = (move + 1) & 1
        comm.Barrier()
        hb_energy = nonbonded_forces['H-Bond'].energies

        particles_per_chain = len(hb_energy) // num_chains
        chain_hb_energies = np.array([np.sum(hb_energy[i * particles_per_chain:(i + 1) * particles_per_chain]) for i in range(num_chains)])
        all_chain_hb_energies = comm.gather(chain_hb_energies, root=0)
        all_positions = comm.gather(sim.state.get_snapshot().particles.position, root=0)
        all_momenta = comm.gather(sim.state.get_snapshot().particles.velocity, root=0)

        if rank == 0 and size > 1:
            it = range(0, size, 2) if move else range(1, size, 2)
            for i in it:
                if i < size - 1:
                    swaps = []
                    idx_j = np.arange(num_chains)
                    np.random.shuffle(idx_j)

                    # ADDED
                    swap_attempts = 0
                    swap_accepted = 0

                    for chain_index in range(num_chains):
                        next_chain_index = idx_j[chain_index]
                        chain_energy_current = all_chain_hb_energies[i][chain_index]
                        chain_energy_next = all_chain_hb_energies[i + 1][next_chain_index]
                        delta_energy = (all_bias[i + 1] - all_bias[i]) * (chain_energy_current - chain_energy_next)
                        accept, acceptance_prob = metropolis_criterion(delta_energy)

                        swap_attempts += 1  # ADDED
                        if accept:
                            swap_accepted += 1  # ADDED
                            swaps.append((chain_index, next_chain_index, i, i + 1))

                    # ADDED: write swap stats to file
                    with open(exchange_log_file, 'a') as f:
                        f.write(f"{sim.timestep}\t{i}\t{i+1}\t{swap_attempts}\t{swap_accepted}\t{swap_accepted / swap_attempts if swap_attempts > 0 else 0.0:.4f}\n")

                    for (chain_index, next_chain_index, index1, index2) in swaps:
                        start_i = chain_index * particles_per_chain
                        end_i = start_i + particles_per_chain
                        start_next = next_chain_index * particles_per_chain
                        end_next = start_next + particles_per_chain

                        chains_left = np.copy(all_positions[index1][start_i:end_i]) - all_positions[index1][start_i]
                        chains_right = np.copy(all_positions[index2][start_next:end_next]) - all_positions[index2][start_next]

                        all_positions[index1][start_i:end_i] = all_positions[index1][start_i] + chains_right
                        all_positions[index2][start_next:end_next] = all_positions[index2][start_next] + chains_left

                        temp_momenta = np.copy(all_momenta[index1][start_i:end_i])
                        all_momenta[index1][start_i:end_i] = np.copy(all_momenta[index2][start_next:end_next])
                        all_momenta[index2][start_next:end_next] = temp_momenta

            updated_positions = all_positions
            updated_momenta = all_momenta
        else:
            updated_positions = None
            updated_momenta = None

        updated_positions = comm.bcast(updated_positions, root=0)
        updated_momenta = comm.bcast(updated_momenta, root=0)
        comm.Barrier()

        restored_snapshot = sim.state.get_snapshot()
        restored_snapshot.particles.position[:] = updated_positions[rank]
        restored_snapshot.particles.velocity[:] = updated_momenta[rank]
        sim.state.set_snapshot(restored_snapshot)


if __name__ == "__main__":
    Project().main()



