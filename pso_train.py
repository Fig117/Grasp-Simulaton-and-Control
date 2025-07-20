# pso_train.py

from pyswarm import pso
from evaluate_pid_params import evaluate_pid_params

# Define PID parameter bounds: [kp, ki, kd]
lower_bounds = [0.01, 0.0, 0.001]
upper_bounds = [2.0, 0.5, 0.2]

# Run PSO optimization
best_params, best_score = pso(
    evaluate_pid_params,
    lower_bounds,
    upper_bounds,
    swarmsize=50,      
    maxiter=50,       
    debug=True        
)

print("\n=== Optimization Complete ===")
print("Best PID Parameters:")
print(f"Kp: {best_params[0]:.4f}, Ki: {best_params[1]:.4f}, Kd: {best_params[2]:.4f}")
print(f"Final Score (total error): {best_score:.6f}")
