from threadpoolctl import threadpool_limits
_ = threadpool_limits(limits=1) # set threads to 1

from mpi4py import MPI
import dill as pickle
import os
import numpy as np
from tqdm import tqdm
from scipy import interpolate
import h5py

def get_inputs(gridvals):
    tmp = np.meshgrid(*gridvals, indexing='ij')
    inputs = np.empty((tmp[0].size, len(tmp)))
    for i, t in enumerate(tmp):
        inputs[:, i] = t.flatten()
    return inputs

def initialize_hdf5(filename):
    """Initialize the HDF5 file."""
    with h5py.File(filename, 'w') as f:
        pass

def save_result_hdf5(filename, index, x, res, grid_shape):
    """Save a single result to the preallocated HDF5 file."""
    
    unraveled_idx = np.unravel_index(index, grid_shape)
    
    with h5py.File(filename, 'a') as f:
        
        # Save input parameters
        if 'inputs' not in f:
            dset = f.create_dataset('inputs', shape=(np.prod(grid_shape),len(x),), dtype=x.dtype)
            f['inputs'][:] = np.nan
        f['inputs'][index] = x

        # Create 'results' group if it doesn't exist
        if 'results' not in f:
            f.create_group('results')
        
        for key, val in res.items():
            data_shape = grid_shape + val.shape  # accommodate vector outputs
            #print(f"This is the grid_shape: {grid_shape} and type: {type(grid_shape)}") # allows for a longer array to be saved
            max_shape_tuple = grid_shape + (None,)
            length_val = len(val)
            #print(f"This is the tuple for max_shapes: {max_shape_tuple}")
            
            if key not in f['results']:
                f['results'].create_dataset(key, shape=data_shape, maxshape=max_shape_tuple, dtype=val.dtype)

            if len(list(f['results'][key][unraveled_idx])) < length_val:
                new_shape = grid_shape + (length_val,)
                dataset = f['results'][key]
                dataset.resize(new_shape)
                #print(f"New shape (results were > initialized results): {dataset.shape}")
                f['results'][key][unraveled_idx] = val

            elif len(list(f['results'][key][unraveled_idx])) > length_val:
                padded_val = np.pad(val, (0, len(list(f['results'][key][unraveled_idx])) - length_val), 'constant', constant_values=0)
                f['results'][key][unraveled_idx] = padded_val
                #print(f"Results were < initialized results; added zeros to match grid length")

            else:
                f['results'][key][unraveled_idx] = val
            
        if 'completed' not in f:
            f.create_dataset('completed', shape=(np.prod(grid_shape),), dtype='bool')
            f['completed'][:] = np.zeros(np.prod(grid_shape),dtype='bool')
        f['completed'][index] = True

def load_completed_mask(filename):
    if os.path.isfile(filename):
        with h5py.File(filename, 'r') as f:
            if 'completed' not in f:
                return np.array([], dtype=int)
            return np.where(f['completed'])[0]
    return np.array([], dtype=int)

def assign_job(comm, rank, serialized_model, job_iter, inputs):
    try:
        job_index = next(job_iter)
        comm.send((serialized_model, job_index, inputs[job_index]), dest=rank, tag=1)
        return True
    except StopIteration:
        comm.send(None, dest=rank, tag=0)
        return False

def master(model_func, gridvals, filename, progress_filename):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    inputs = get_inputs(gridvals)
    gridshape = tuple(len(v) for v in gridvals)

    serialized_model = pickle.dumps(model_func)

    # Initialize HDF5 if needed
    if not os.path.exists(filename):
        print("Initializing HDF5 output...")
        initialize_hdf5(filename)

    completed_inds = load_completed_mask(filename)
    if len(completed_inds) > 0:
        print(f'Calculations completed/total: {len(completed_inds)}/{inputs.shape[0]}.')
        if len(completed_inds) == inputs.shape[0]:
            print('All calculations completed.')
        else:
            print('Restarting calculations...')

    # Get inputs that have not yet been computed
    job_indices = [i for i in range(len(inputs)) if i not in completed_inds]
    job_iter = iter(job_indices)
    
    # Open progress log file for writing
    with open(progress_filename, 'w') as log_file:
        pbar = tqdm(total=len(job_indices), file=log_file, dynamic_ncols=True)
        status = MPI.Status()

        # Assign initial workers
        active_workers = 0
        for rank in range(1, size):
            if assign_job(comm, rank, serialized_model, job_iter, inputs):
                active_workers += 1

        # Continue until all workers are terminated
        while active_workers > 0:

            # Get result form worker
            index, x, res = comm.recv(source=MPI.ANY_SOURCE, tag=2, status=status)
            #print(f'Here is one place where results were calculated: {res}')
            worker_rank = status.Get_source()

            # Save the result
            save_result_hdf5(filename, index, x, res, gridshape)
            
            pbar.update(1)
            log_file.flush()

            # Assign a new job to the worker.
            if not assign_job(comm, worker_rank, serialized_model, job_iter, inputs):
                active_workers -= 1

        pbar.close()

def worker():
    comm = MPI.COMM_WORLD
    status = MPI.Status()

    while True:
        # Get inputs from master process
        data = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        if status.Get_tag() == 0:
            break # Shutdown signal

        # Call the function on the inputs
        serialized_model, index, x = data
        model_func = pickle.loads(serialized_model)
        res = model_func(x)

        # Send the results to the master process
        comm.send((index, x, res), dest=0, tag=2)

def make_grid(model_func, gridvals, filename, progress_filename):
    """
    Run a parallel grid computation using MPI, saving results to an HDF5 file.

    This function distributes computations across available MPI ranks. The master
    process assigns jobs to worker processes, collects results, and writes them to
    an HDF5 file. A separate progress log file tracks computation progress.

    Parameters
    ----------
    model_func : callable
        A function that takes a 1D numpy array of input parameters and returns
        a dictionary of results, where each key corresponds to a quantity (numpy array)
        to be saved.
    
    gridvals : tuple of 1D numpy arrays
        Defines the parameter grid. Each array in the tuple represents the discrete 
        values for one dimension of the parameter space.

    filename : str
        Path to the HDF5 file where computed results will be stored. The file will contain
        groups for each grid point index, each with datasets for the input parameters 
        and the model output.

    progress_filename : str
        Path to the text file where progress updates (from the master process) will be logged.

    Notes
    -----
    - This function must be run with an MPI launcher (e.g., `mpiexec -n N python script.py`).
    - The results are saved incrementally, so the computation can be resumed if interrupted.
    - Only rank 0 (master) writes to the output files.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        # Master process
        master(model_func, gridvals, filename, progress_filename)
    else:
        # Worker process
        worker()

class GridInterpolator():
    """
    A class for interpolating data saved from an HDF5 grid of simulation outputs.

    This class reads an HDF5 file containing simulation outputs stored on a parameter grid.
    It provides a method to generate interpolators that can predict values or arrays of 
    results at arbitrary points within the grid using `scipy.interpolate.RegularGridInterpolator`.

    Parameters
    ----------
    filename : str
        Path to the HDF5 file containing the simulation results.

    gridvals : tuple of np.ndarray
        The parameter grid values, used to define the interpolation space.

    Attributes
    ----------
    gridvals : tuple of np.ndarray
        The parameter values for each grid dimension.

    gridshape : tuple of int
        The shape of the parameter grid, inferred from the lengths of `gridvals`.

    data : dict of np.ndarray
        The results from the HDF5 file.
    """

    def __init__(self, filename, gridvals):
        """
        Initialize the GridInterpolator by loading data from an HDF5 file.

        Parameters
        ----------
        filename : str
            Path to the HDF5 file containing the simulation results.

        gridvals : tuple of np.ndarray
            The parameter grid values, used to define the interpolation space.
        """
        self.gridvals = gridvals
        self.gridshape = tuple(len(a) for a in gridvals)
        print(f"This is the gridshape: {self.gridshape}")

        with h5py.File(filename, 'r') as f:
            self.data = {}
            for key in f['results'].keys():
                self.data[key] = f['results'][key][...]

    def make_interpolator(self, key, logspace=False):
        """
        Create an interpolator for a grid parameter.

        Parameters
        ----------
        key : str
            The key in the `self.data` dictionary for which to create the interpolator.

        logspace : bool, optional
            If True, interpolation is performed in log10-space. This is useful for 
            quantities that span many orders of magnitude.

        Returns
        -------
        interp : function
            Interpolator function, which is called with a tuple of arguments: `interp((2,3,4))`.
        """
    
        data = self.data[key]
    
        # Apply log-space transformation if needed
        if logspace:
            data = np.log10(np.maximum(data, 2e-38))

        # Create the interpolator
        rgi = interpolate.RegularGridInterpolator(self.gridvals, data)

        def interp(vals):
            out = rgi(vals)
            if logspace:
                out = 10.0 ** out
            return out

        return interp
    