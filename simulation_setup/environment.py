import signac
from flow import environment


class RavenCluster(environment.DefaultSlurmEnvironment):
    hostname_pattern = r"vip"
    template="viper.sh"
    
    _partition_config = environment._PartitionConfig(cpus_per_node={'CPU':48, 'GPU':48}, 
                                                     gpus_per_node={'GPU':2}, 
                                                     node_types={'CPU': environment._NodeTypes.SHARED,
                                                                 'GPU':environment._NodeTypes.SHARED})
    
    @classmethod
    def _get_mpi_prefix(cls, operation, parallel):
        """Get the jsrun options based on directives.

        Parameters
        ----------
        operation : :class:`flow.project._JobOperation`
            The operation to be prefixed.
        parallel : bool
            If True, operations are assumed to be executed in parallel, which
            means that the number of total tasks is the sum of all tasks
            instead of the maximum number of tasks. Default is set to False.

        Returns
        -------
        str
            The prefix to be added to the operation's command.

        """
        #nranks = operation.directives.get("nranks")
        #tpn = operation.directives.get("tasks_per_node")
        #nnodes = operation.directives.get("nodes")
        
        #all_none = nranks is None and tpn is None and nnodes is None
        
        requires_prefix = True #all_none or ((nranks > 1) | (tpn > 1) | (nnodes > 1))
        if requires_prefix:
            return 'srun'
        else:
            return ''
