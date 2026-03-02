import model_data_loader as mdl
import sys
from datetime import datetime as dt
from multiprocessing import Pool
import math


def save_wva_dataset(args):
    """ Load the dataset for a specific experiment and month, calculate the precipitation age, and save the modified dataset."""
    experiment_name, month, logger = args
    t0_ = dt.now()
    try:
        monthly_dataset = mdl.MonthlyDataset(experiment_name, month)
        monthly_dataset.add_new_fields_to_ds()
        monthly_dataset.save_dataset()
        monthly_dataset.save_monthly_average()
        t_final = dt.now()
        logger.log(f"✓ Completed month {month}: {t_final-t0_}")
        return True
    except Exception as e:
        logger.log(f"✗ Failed to process month {month}: {str(e)}")
        return False


class Logger:
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name

    def log(self, message):
        print(f"[{dt.now()}] {message}", file=sys.stdout, flush=True)


if __name__ == "__main__":
    experiment_name = str(sys.argv[1])
    month_start = int(sys.argv[2])
    month_end = int(sys.argv[3])
    num_cpus = 10
    
    logger = Logger(experiment_name)
    
    # Calculate months to process
    total_months = month_end - month_start + 1
    num_batches = math.ceil(total_months / num_cpus)
    
    logger.log(f"Processing experiment: {experiment_name}")
    logger.log(f"Months to process: {total_months} (from {month_start} to {month_end})")
    logger.log(f"Using {num_cpus} CPUs in {num_batches} batch(es)")
    
    t0 = dt.now()
    
    # Create list of months to process
    months = list(range(month_start, month_end + 1))
    
    # Process in batches
    completed = 0
    failed = 0
    
    for batch_num in range(num_batches):
        batch_start_idx = batch_num * num_cpus
        batch_end_idx = min(batch_start_idx + num_cpus, total_months)
        batch_months = months[batch_start_idx:batch_end_idx]
        batch_size = len(batch_months)
        
        logger.log(f"Starting batch {batch_num + 1}/{num_batches} with {batch_size} month(s): {batch_months}")
        
        # Create arguments for each month
        task_args = [(experiment_name, month, logger) for month in batch_months]
        
        # Run processes in parallel
        with Pool(processes=batch_size) as pool:
            results = pool.map(save_wva_dataset, task_args)
        
        batch_completed = sum(results)
        batch_failed = batch_size - batch_completed
        completed += batch_completed
        failed += batch_failed
        
        logger.log(f"Batch {batch_num + 1}/{num_batches} complete: {batch_completed} succeeded, {batch_failed} failed")
    
    t1 = dt.now()
    logger.log(f"All processing complete for experiment: {experiment_name}")
    logger.log(f"Results: {completed} succeeded, {failed} failed in {t1-t0}")