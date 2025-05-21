import numpy as np
from scipy.integrate import solve_ivp
import random
import csv
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Constants
G = 6.67430e-11  # Gravitational constant

def two_body_equations(t, y, m1, m2):
    """Differential equations for the two-body problem."""
    x1, y1, x2, y2, vx1, vy1, vx2, vy2 = y
    dx = x2 - x1
    dy = y2 - y1
    r = np.sqrt(dx**2 + dy**2)
    f = G * m1 * m2 / (r**3)
    return [vx1, vy1, vx2, vy2, f*dx/m1, f*dy/m1, -f*dx/m2, -f*dy/m2]

def generate_random_inputs(n):
    """Generate random initial conditions for the two-body system."""
    inputs = np.zeros((11, n))
    
    # Vectorized generation of parameters
    inputs[0] = 10**np.random.uniform(20, 30, n)  # m1
    inputs[1] = 10**np.random.uniform(20, 30, n)  # m2
    
    # Positions with safety distance
    for i in range(n):
        x1, y1 = random.uniform(-1e12, 1e12), random.uniform(-1e12, 1e12)
        x2, y2 = random.uniform(-1e12, 1e12), random.uniform(-1e12, 1e12)
        while np.sqrt((x2-x1)**2 + (y2-y1)**2) < 1e10:
            x2, y2 = random.uniform(-1e12, 1e12), random.uniform(-1e12, 1e12)
        inputs[2:6, i] = [x1, y1, x2, y2]
    
    # Velocities
    inputs[6:10] = np.random.uniform(-1e4, 1e4, (4, n))
    
    # Time
    inputs[10] = np.random.uniform(1e5, 1e7, n)
    
    return inputs

def simulate_single(args):
    """Wrapper function for parallel simulation of a single system"""
    i, m1, m2, initial_state, t_max = args
    try:
        sol = solve_ivp(two_body_equations, [0, t_max], initial_state,
                       args=(m1, m2), method='DOP853',
                       rtol=1e-6, atol=1e-6)  # Slightly relaxed tolerances for speed
        return sol.y[:4, -1]
    except:
        return np.full(4, np.nan)  # Handle rare integration failures

def simulate_two_body_system(inputs):
    """Parallel simulation of multiple systems"""
    n = inputs.shape[1]
    args_list = [(i, inputs[0,i], inputs[1,i], inputs[2:10,i], inputs[10,i]) 
                 for i in range(n)]
    
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(simulate_single, args_list), total=n))
    
    return np.array(results)

def save_to_csv(inputs, outputs, filename='two_body_data_large.csv'):
    """Save input parameters and final positions to CSV file"""
    header = [
        'mass1', 'mass2',
        'x1_initial', 'y1_initial', 'x2_initial', 'y2_initial',
        'vx1_initial', 'vy1_initial', 'vx2_initial', 'vy2_initial',
        'time',
        'x1_final', 'y1_final', 'x2_final', 'y2_final'
    ]
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        
        for i in tqdm(range(inputs.shape[1]), desc='Saving data'):
            row = list(inputs[:, i]) + list(outputs[i])
            writer.writerow(row)

if __name__ == "__main__":
    # Generate 15,000 data points
    n_simulations = 500
    
    # Generate all inputs first
    print("Generating initial conditions...")
    inputs = generate_random_inputs(n_simulations)
    
    # Run simulations in parallel
    print("Running simulations...")
    final_positions = simulate_two_body_system(inputs)
    
    # Filter out any failed simulations (should be very few)
    valid_mask = ~np.isnan(final_positions).any(axis=1)
    inputs = inputs[:, valid_mask]
    final_positions = final_positions[valid_mask]
    
    # Save to CSV
    print("Saving results...")
    save_to_csv(inputs, final_positions)
    
    print(f"CSV file generated successfully with {len(final_positions)} valid data points.")