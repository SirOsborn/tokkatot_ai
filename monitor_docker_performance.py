"""
Docker Performance Monitoring Script for Ensemble Model
Monitors CPU and RAM usage during model inference to assess cloud deployment viability.
"""

import docker
import time
import requests
import json
from datetime import datetime
from pathlib import Path
import statistics
import psutil
from typing import Dict, List, Tuple


class DockerPerformanceMonitor:
    """Monitor Docker container performance for ML model deployment."""
    
    def __init__(self, container_name: str = "tokkatot-ai-service"):
        """
        Initialize the performance monitor.
        
        Args:
            container_name: Name of the Docker container to monitor
        """
        self.container_name = container_name
        self.client = docker.from_env()
        self.container = None
        self.metrics = {
            'cpu_percent': [],
            'memory_mb': [],
            'memory_percent': [],
            'network_io': [],
            'timestamps': []
        }
        self.inference_results = []
        
    def get_container_stats(self) -> Dict:
        """Get current container statistics."""
        if not self.container:
            return {}
        
        try:
            stats = self.container.stats(stream=False)
            
            # Calculate CPU percentage
            cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                       stats['precpu_stats']['cpu_usage']['total_usage']
            system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                          stats['precpu_stats']['system_cpu_usage']
            cpu_count = len(stats['cpu_stats']['cpu_usage'].get('percpu_usage', [1]))
            
            cpu_percent = 0.0
            if system_delta > 0:
                cpu_percent = (cpu_delta / system_delta) * cpu_count * 100.0
            
            # Calculate memory usage
            memory_usage = stats['memory_stats']['usage']
            memory_limit = stats['memory_stats']['limit']
            memory_mb = memory_usage / (1024 * 1024)
            memory_percent = (memory_usage / memory_limit) * 100
            
            # Network I/O
            network_io = stats['networks']
            
            return {
                'cpu_percent': cpu_percent,
                'memory_mb': memory_mb,
                'memory_percent': memory_percent,
                'network_io': network_io,
                'timestamp': datetime.now()
            }
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {}
    
    def wait_for_container_ready(self, timeout: int = 120, check_interval: int = 2) -> bool:
        """
        Wait for container to be ready by checking health endpoint.
        
        Args:
            timeout: Maximum time to wait in seconds
            check_interval: Time between checks in seconds
            
        Returns:
            True if container is ready, False if timeout
        """
        print(f"Waiting for container to be ready (timeout: {timeout}s)...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get("http://localhost:8000/health", timeout=5)
                if response.status_code == 200:
                    print("✓ Container is ready!")
                    return True
            except requests.exceptions.RequestException:
                pass
            
            time.sleep(check_interval)
            elapsed = int(time.time() - start_time)
            print(f"  Waiting... ({elapsed}s elapsed)", end='\r')
        
        print(f"\n✗ Container did not become ready within {timeout}s")
        return False
    
    def run_inference_test(self, image_path: str, description: str = "") -> Dict:
        """
        Run a single inference test and measure performance.
        
        Args:
            image_path: Path to test image
            description: Description of the test
            
        Returns:
            Dictionary with test results
        """
        print(f"\nRunning inference test: {description or image_path}")
        
        # Record baseline stats
        baseline_stats = self.get_container_stats()
        
        # Run inference
        start_time = time.time()
        try:
            with open(image_path, 'rb') as f:
                files = {'file': (Path(image_path).name, f, 'image/jpeg')}
                response = requests.post(
                    "http://localhost:8000/predict",
                    files=files,
                    timeout=30
                )
            
            inference_time = time.time() - start_time
            
            # Record post-inference stats
            post_stats = self.get_container_stats()
            
            result = {
                'description': description or image_path,
                'image_path': image_path,
                'inference_time_ms': inference_time * 1000,
                'status_code': response.status_code,
                'response': response.json() if response.status_code == 200 else None,
                'baseline_cpu': baseline_stats.get('cpu_percent', 0),
                'peak_cpu': post_stats.get('cpu_percent', 0),
                'baseline_memory_mb': baseline_stats.get('memory_mb', 0),
                'peak_memory_mb': post_stats.get('memory_mb', 0),
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"  ✓ Inference completed in {inference_time*1000:.2f}ms")
            print(f"  CPU: {baseline_stats.get('cpu_percent', 0):.1f}% -> {post_stats.get('cpu_percent', 0):.1f}%")
            print(f"  Memory: {baseline_stats.get('memory_mb', 0):.1f}MB -> {post_stats.get('memory_mb', 0):.1f}MB")
            
            if result['response']:
                pred = result['response']
                print(f"  Prediction: {pred.get('predicted_class', 'N/A')} ({pred.get('confidence', 0)*100:.1f}%)")
            
            return result
            
        except Exception as e:
            print(f"  ✗ Error during inference: {e}")
            return {
                'description': description or image_path,
                'image_path': image_path,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def monitor_container(self, duration: int = 60, interval: int = 1):
        """
        Monitor container resource usage over time.
        
        Args:
            duration: How long to monitor in seconds
            interval: Time between measurements in seconds
        """
        print(f"\nMonitoring container for {duration}s (sampling every {interval}s)...")
        start_time = time.time()
        
        while time.time() - start_time < duration:
            stats = self.get_container_stats()
            if stats:
                self.metrics['cpu_percent'].append(stats['cpu_percent'])
                self.metrics['memory_mb'].append(stats['memory_mb'])
                self.metrics['memory_percent'].append(stats['memory_percent'])
                self.metrics['timestamps'].append(stats['timestamp'])
            
            time.sleep(interval)
        
        print("✓ Monitoring complete")
    
    def run_comprehensive_test(self, test_images: List[Tuple[str, str]], 
                              monitoring_duration: int = 30):
        """
        Run comprehensive performance testing.
        
        Args:
            test_images: List of (image_path, description) tuples
            monitoring_duration: How long to monitor between tests
        """
        print("\n" + "="*70)
        print("STARTING COMPREHENSIVE DOCKER PERFORMANCE TEST")
        print("="*70)
        
        try:
            # Get container
            self.container = self.client.containers.get(self.container_name)
            print(f"✓ Found container: {self.container_name}")
            
            # Wait for container to be ready
            if not self.wait_for_container_ready():
                print("Cannot proceed - container not ready")
                return
            
            # Initial monitoring period (idle state)
            print("\n--- Phase 1: Baseline Monitoring (Idle State) ---")
            self.monitor_container(duration=monitoring_duration, interval=1)
            
            # Run inference tests
            print("\n--- Phase 2: Inference Testing ---")
            for image_path, description in test_images:
                if Path(image_path).exists():
                    result = self.run_inference_test(image_path, description)
                    self.inference_results.append(result)
                    time.sleep(2)  # Brief pause between tests
                else:
                    print(f"  ✗ Image not found: {image_path}")
            
            # Post-inference monitoring
            print("\n--- Phase 3: Post-Inference Monitoring ---")
            self.monitor_container(duration=monitoring_duration, interval=1)
            
            # Generate report
            self.generate_report()
            
        except docker.errors.NotFound:
            print(f"✗ Container '{self.container_name}' not found")
            print("  Make sure the container is running")
        except Exception as e:
            print(f"✗ Error during testing: {e}")
    
    def generate_report(self, output_dir: str = "outputs/performance"):
        """Generate comprehensive performance report."""
        print("\n" + "="*70)
        print("GENERATING PERFORMANCE REPORT")
        print("="*70)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = output_path / f"docker_performance_report_{timestamp}.txt"
        json_file = output_path / f"docker_performance_data_{timestamp}.json"
        
        # Calculate statistics
        cpu_stats = {
            'min': min(self.metrics['cpu_percent']) if self.metrics['cpu_percent'] else 0,
            'max': max(self.metrics['cpu_percent']) if self.metrics['cpu_percent'] else 0,
            'mean': statistics.mean(self.metrics['cpu_percent']) if self.metrics['cpu_percent'] else 0,
            'median': statistics.median(self.metrics['cpu_percent']) if self.metrics['cpu_percent'] else 0,
            'stdev': statistics.stdev(self.metrics['cpu_percent']) if len(self.metrics['cpu_percent']) > 1 else 0
        }
        
        mem_stats = {
            'min': min(self.metrics['memory_mb']) if self.metrics['memory_mb'] else 0,
            'max': max(self.metrics['memory_mb']) if self.metrics['memory_mb'] else 0,
            'mean': statistics.mean(self.metrics['memory_mb']) if self.metrics['memory_mb'] else 0,
            'median': statistics.median(self.metrics['memory_mb']) if self.metrics['memory_mb'] else 0,
            'stdev': statistics.stdev(self.metrics['memory_mb']) if len(self.metrics['memory_mb']) > 1 else 0
        }
        
        # Calculate inference statistics
        successful_inferences = [r for r in self.inference_results if 'error' not in r]
        if successful_inferences:
            inference_times = [r['inference_time_ms'] for r in successful_inferences]
            inference_stats = {
                'count': len(successful_inferences),
                'min_ms': min(inference_times),
                'max_ms': max(inference_times),
                'mean_ms': statistics.mean(inference_times),
                'median_ms': statistics.median(inference_times),
                'stdev_ms': statistics.stdev(inference_times) if len(inference_times) > 1 else 0
            }
        else:
            inference_stats = {'count': 0}
        
        # Generate text report
        report = []
        report.append("="*70)
        report.append("DOCKER PERFORMANCE MONITORING REPORT")
        report.append("Ensemble Model - CPU-Only Deployment Assessment")
        report.append("="*70)
        report.append(f"\nReport Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Container: {self.container_name}")
        report.append(f"Monitoring Duration: {len(self.metrics['cpu_percent'])}s")
        
        report.append("\n" + "-"*70)
        report.append("CPU USAGE STATISTICS")
        report.append("-"*70)
        report.append(f"Minimum:      {cpu_stats['min']:.2f}%")
        report.append(f"Maximum:      {cpu_stats['max']:.2f}%")
        report.append(f"Mean:         {cpu_stats['mean']:.2f}%")
        report.append(f"Median:       {cpu_stats['median']:.2f}%")
        report.append(f"Std Dev:      {cpu_stats['stdev']:.2f}%")
        
        report.append("\n" + "-"*70)
        report.append("MEMORY USAGE STATISTICS")
        report.append("-"*70)
        report.append(f"Minimum:      {mem_stats['min']:.2f} MB")
        report.append(f"Maximum:      {mem_stats['max']:.2f} MB")
        report.append(f"Mean:         {mem_stats['mean']:.2f} MB")
        report.append(f"Median:       {mem_stats['median']:.2f} MB")
        report.append(f"Std Dev:      {mem_stats['stdev']:.2f} MB")
        
        report.append("\n" + "-"*70)
        report.append("INFERENCE PERFORMANCE")
        report.append("-"*70)
        report.append(f"Total Tests:  {len(self.inference_results)}")
        report.append(f"Successful:   {inference_stats['count']}")
        report.append(f"Failed:       {len(self.inference_results) - inference_stats['count']}")
        
        if inference_stats['count'] > 0:
            report.append(f"\nInference Time Statistics:")
            report.append(f"  Minimum:    {inference_stats['min_ms']:.2f} ms")
            report.append(f"  Maximum:    {inference_stats['max_ms']:.2f} ms")
            report.append(f"  Mean:       {inference_stats['mean_ms']:.2f} ms")
            report.append(f"  Median:     {inference_stats['median_ms']:.2f} ms")
            report.append(f"  Std Dev:    {inference_stats['stdev_ms']:.2f} ms")
        
        report.append("\n" + "-"*70)
        report.append("DETAILED INFERENCE RESULTS")
        report.append("-"*70)
        for i, result in enumerate(self.inference_results, 1):
            report.append(f"\nTest {i}: {result.get('description', 'N/A')}")
            if 'error' in result:
                report.append(f"  Status: FAILED - {result['error']}")
            else:
                report.append(f"  Inference Time: {result['inference_time_ms']:.2f} ms")
                report.append(f"  CPU Usage: {result['baseline_cpu']:.1f}% -> {result['peak_cpu']:.1f}%")
                report.append(f"  Memory: {result['baseline_memory_mb']:.1f}MB -> {result['peak_memory_mb']:.1f}MB")
                if result.get('response'):
                    pred = result['response']
                    report.append(f"  Prediction: {pred.get('predicted_class', 'N/A')}")
                    report.append(f"  Confidence: {pred.get('confidence', 0)*100:.1f}%")
                    report.append(f"  Safe For Processing: {'Yes' if pred.get('safe_for_processing') else 'No'}")
        
        report.append("\n" + "="*70)
        report.append("CLOUD CPU DEPLOYMENT ASSESSMENT")
        report.append("="*70)
        
        # Assessment logic
        max_cpu = cpu_stats['max']
        mean_cpu = cpu_stats['mean']
        max_mem_mb = mem_stats['max']
        mean_inference_ms = inference_stats.get('mean_ms', 0)
        
        report.append(f"\nResource Requirements:")
        report.append(f"  Recommended CPU Cores: {max(1, int(max_cpu / 100) + 1)}")
        report.append(f"  Recommended RAM: {max(2, int(max_mem_mb / 1024) + 1)} GB")
        
        report.append(f"\nPerformance Assessment:")
        if mean_inference_ms < 1000:
            report.append(f"  ✓ EXCELLENT: Avg inference time {mean_inference_ms:.0f}ms (<1s)")
        elif mean_inference_ms < 3000:
            report.append(f"  ✓ GOOD: Avg inference time {mean_inference_ms:.0f}ms (<3s)")
        elif mean_inference_ms < 5000:
            report.append(f"  ⚠ ACCEPTABLE: Avg inference time {mean_inference_ms:.0f}ms (<5s)")
        else:
            report.append(f"  ✗ SLOW: Avg inference time {mean_inference_ms:.0f}ms (>5s)")
        
        if mean_cpu < 50:
            report.append(f"  ✓ LOW CPU: Avg {mean_cpu:.1f}% CPU usage")
        elif mean_cpu < 80:
            report.append(f"  ⚠ MODERATE CPU: Avg {mean_cpu:.1f}% CPU usage")
        else:
            report.append(f"  ✗ HIGH CPU: Avg {mean_cpu:.1f}% CPU usage")
        
        if max_mem_mb < 1024:
            report.append(f"  ✓ LOW MEMORY: Peak {max_mem_mb:.0f}MB (<1GB)")
        elif max_mem_mb < 2048:
            report.append(f"  ✓ MODERATE MEMORY: Peak {max_mem_mb:.0f}MB (<2GB)")
        elif max_mem_mb < 4096:
            report.append(f"  ⚠ HIGH MEMORY: Peak {max_mem_mb:.0f}MB (<4GB)")
        else:
            report.append(f"  ✗ VERY HIGH MEMORY: Peak {max_mem_mb:.0f}MB (>4GB)")
        
        report.append(f"\nCloud CPU Deployment Recommendation:")
        if mean_inference_ms < 3000 and mean_cpu < 80 and max_mem_mb < 2048:
            report.append("  ✓ STRONGLY RECOMMENDED")
            report.append("  This model is well-suited for cloud CPU deployment.")
            report.append("  Suggested instance: 2 vCPU, 2-4GB RAM")
        elif mean_inference_ms < 5000 and max_mem_mb < 4096:
            report.append("  ✓ RECOMMENDED with considerations")
            report.append("  This model can run on cloud CPU but may benefit from optimization.")
            report.append("  Suggested instance: 2-4 vCPU, 4GB RAM")
        else:
            report.append("  ⚠ NOT RECOMMENDED for basic CPU instances")
            report.append("  Consider GPU instances or model optimization for production use.")
        
        report.append("\n" + "="*70)
        report.append("END OF REPORT")
        report.append("="*70)
        
        # Write text report
        report_text = "\n".join(report)
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(report_text)
        
        # Write JSON data
        json_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'container': self.container_name,
                'monitoring_duration_s': len(self.metrics['cpu_percent']),
                'test_count': len(self.inference_results)
            },
            'cpu_statistics': cpu_stats,
            'memory_statistics_mb': mem_stats,
            'inference_statistics': inference_stats,
            'detailed_metrics': {
                'cpu_percent': self.metrics['cpu_percent'],
                'memory_mb': self.metrics['memory_mb'],
                'memory_percent': self.metrics['memory_percent'],
                'timestamps': [ts.isoformat() for ts in self.metrics['timestamps']]
            },
            'inference_results': self.inference_results
        }
        
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"\n✓ Reports saved:")
        print(f"  Text Report: {report_file}")
        print(f"  JSON Data: {json_file}")


def main():
    """Main execution function."""
    print("Docker Performance Monitoring Tool")
    print("="*70)
    
    # Find test images from the archive
    test_images = []
    archive_path = Path("archive/data/test")
    
    if archive_path.exists():
        # Get one sample from each class
        for class_name in ['Healthy', 'Coccidiosis', 'Salmonella', 'New Castle Disease']:
            class_path = archive_path / class_name
            if class_path.exists():
                images = list(class_path.glob("*.jpg"))[:2]  # Get 2 images per class
                for img in images:
                    test_images.append((str(img), f"{class_name} Sample"))
    
    if not test_images:
        print("⚠ No test images found in archive/data/test/")
        print("Will run monitoring without inference tests")
        test_images = []
    else:
        print(f"✓ Found {len(test_images)} test images")
    
    # Create monitor and run tests
    monitor = DockerPerformanceMonitor("tokkatot-ai-service")
    monitor.run_comprehensive_test(
        test_images=test_images,
        monitoring_duration=30  # 30 seconds of monitoring for idle/post states
    )


if __name__ == "__main__":
    main()
