import signac

project = signac.init_project()

# Define the number of fragments
num_fragments = 3

# Initialize jobs for each fragment
for fid in range(num_fragments):
    initial_values = {
        'fragmentID': fid
    }
    project.open_job(initial_values).init()

