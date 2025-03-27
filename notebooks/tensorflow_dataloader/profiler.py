import time
import psutil
import numpy as np
import tensorflow as tf
from ray.data import Dataset
import matplotlib.pyplot as plt
from tensorflow.python.profiler import profiler_v2 as profiler

class DataPipelineProfiler:
    def __init__(self, files, batch_size=32, img_shape=(256, 256)):
        self.files = files
        self.batch_size = batch_size
        self.img_shape = img_shape
        self.results = {}
        self.labels_dict = {}  # Should be populated with actual labels

    def track_resources(self):
        """Track system resources during execution"""
        self.cpu_usage = []
        self.memory_usage = []
        
        def monitor():
            while getattr(self, 'monitoring', True):
                self.cpu_usage.append(psutil.cpu_percent())
                self.memory_usage.append(psutil.virtual_memory().percent)
                time.sleep(0.1)
                
        from threading import Thread
        self.monitor_thread = Thread(target=monitor)
        self.monitor_thread.start()

    def stop_tracking(self):
        self.monitoring = False
        self.monitor_thread.join()

    def profile_original(self):
        """Profile the original implementation"""
        self.track_resources()
        start_time = time.time()
        
        # Original data loading
        data = []
        for i, file in enumerate(self.files):
            ret = self.load_data(file, ids=i)
            if ret is not None:
                data.append(ret)
        
        # Original dataset creation
        dataset = self.create_dataset_original(data)
        
        # Warm-up and iteration
        train_time = self.time_dataset_iteration(dataset)
        
        total_time = time.time() - start_time
        self.stop_tracking()
        
        return {
            'total_time': total_time,
            'train_time_per_epoch': train_time,
            'cpu_usage': np.mean(self.cpu_usage),
            'memory_usage': np.max(self.memory_usage)
        }

    def profile_tf_optimized(self):
        """Profile TensorFlow-optimized implementation"""
        self.track_resources()
        start_time = time.time()
        
        # Optimized dataset
        dataset = self.create_dataset_optimized()
        
        # Warm-up and iteration
        train_time = self.time_dataset_iteration(dataset)
        
        total_time = time.time() - start_time
        self.stop_tracking()
        
        return {
            'total_time': total_time,
            'train_time_per_epoch': train_time,
            'cpu_usage': np.mean(self.cpu_usage),
            'memory_usage': np.max(self.memory_usage)
        }

    def profile_ray_optimized(self):
        """Profile Ray-optimized implementation"""
        import ray
        ray.init(ignore_reinit_error=True)
        
        self.track_resources()
        start_time = time.time()
        
        # Ray dataset pipeline
        dataset = self.create_ray_dataset()
        tf_dataset = dataset.to_tf(...)  # Implement proper conversion
        
        # Warm-up and iteration
        train_time = self.time_dataset_iteration(tf_dataset)
        
        total_time = time.time() - start_time
        self.stop_tracking()
        ray.shutdown()
        
        return {
            'total_time': total_time,
            'train_time_per_epoch': train_time,
            'cpu_usage': np.mean(self.cpu_usage),
            'memory_usage': np.max(self.memory_usage)
        }

    def time_dataset_iteration(self, dataset):
        # Warm-up
        for _ in dataset.take(1): pass
        
        # TF Profiler
        profiler.start()
        start = time.time()
        for _ in dataset: pass
        train_time = time.time() - start
        profiler.stop()
        
        return train_time

    def create_dataset_optimized(self):
        """Optimized TensorFlow pipeline implementation"""
        def tf_decode_image(file_path):
            img = tf.io.read_file(file_path)
            img = tf.io.decode_image(img, channels=3, expand_animations=False)
            img = tf.image.resize(img, self.img_shape)
            img = tf.transpose(img, [2, 0, 1])
            return img / 255.0

        return tf.data.Dataset.from_tensor_slices(self.files)\
            .shuffle(len(self.files))\
            .map(tf_decode_image, num_parallel_calls=tf.data.AUTOTUNE)\
            .batch(self.batch_size)\
            .prefetch(tf.data.AUTOTUNE)

    def create_ray_dataset(self):
        """Ray-optimized implementation"""
        import ray
        return ray.data.read_images(self.files)\
            .map(lambda x: {'image': x})\
            .random_shuffle()\
            .iter_tf_batches(
                batch_size=self.batch_size,
                prefetch_blocks=10
            )

    def generate_report(self, results):
        """Generate comparative performance report"""
        print("\n=== Performance Comparison Report ===")
        print(f"{'Metric':<25} | {'Original':<10} | {'TF Optimized':<12} | {'Ray Optimized':<14}")
        print("-"*65)
        
        for metric in ['total_time', 'train_time_per_epoch', 'cpu_usage', 'memory_usage']:
            row = f"{metric:<25} | "
            for version in ['original', 'tf_optimized', 'ray_optimized']:
                value = results[version][metric]
                if isinstance(value, float):
                    row += f"{value:>8.2f} | "
                else:
                    row += f"{value:>8} | "
            print(row)

        # Plotting
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        metrics = ['total_time', 'train_time_per_epoch', 'cpu_usage', 'memory_usage']
        titles = ['Total Processing Time', 'Training Time per Epoch', 
                 'Average CPU Usage (%)', 'Peak Memory Usage (%)']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axs[idx//2, idx%2]
            values = [results[v][metric] for v in ['original', 'tf_optimized', 'ray_optimized']]
            ax.bar(['Original', 'TF Optimized', 'Ray Optimized'], values)
            ax.set_title(title)
            if '%' in title:
                ax.set_ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig('performance_comparison.png')
        print("\nSaved performance plot to performance_comparison.png")

# Usage Example
if __name__ == "__main__":
    files = glob.glob('/path/to/data/*')  # Replace with actual data path
    
    profiler = DataPipelineProfiler(files)
    
    results = {
        'original': profiler.profile_original(),
        'tf_optimized': profiler.profile_tf_optimized(),
        'ray_optimized': profiler.profile_ray_optimized()
    }
    
    profiler.generate_report(results)
