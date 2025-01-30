import asyncio
import aiohttp
import time
import numpy as np
from collections import defaultdict
from datetime import datetime

# Configuration
TEST_URL = "http://71.138.22.131:8000/generate"
PROMPT = "Write a Python function to calculate fibonacci numbers using recursion"
TIMEOUT = 300  # 5 minutes timeout for each request
WARMUP_REQUESTS = 5  # Initial requests to warm up the model

def create_stats_dict():
    """Create a properly initialized stats dictionary"""
    stats = defaultdict(int)
    stats['latencies'] = []
    stats['error_types'] = []
    return stats

async def send_request(session, semaphore, stats):
    """Send a single request with concurrency control"""
    start_time = time.time()
    try:
        async with semaphore, session.post(
            TEST_URL,
            json={
                "prompt": PROMPT,
                "max_tokens": 512,
                "temperature": 0.7
            },
            timeout=aiohttp.ClientTimeout(total=TIMEOUT)
        ) as response:
            json_response = await response.json()
            latency = time.time() - start_time
            
            stats['success'] += 1
            stats['latencies'].append(latency)
            
            if "error" in json_response:
                stats['errors'] += 1
                stats['error_types'].append(json_response.get("error", "unknown"))
            
            return latency
            
    except Exception as e:
        stats['failures'] += 1
        stats['error_types'].append(str(e))
        return None

async def run_concurrent_test(concurrent_workers, total_requests, stats):
    """Run test with specified concurrency level"""
    semaphore = asyncio.Semaphore(concurrent_workers)
    connector = aiohttp.TCPConnector(limit=0)  # No limit on parallel connections
    
    async with aiohttp.ClientSession(connector=connector) as session:
        # Create all tasks
        tasks = [send_request(session, semaphore, stats) for _ in range(total_requests)]
        # Run tasks and gather results
        await asyncio.gather(*tasks)

def print_stats(concurrent_workers, stats, duration):
    """Print formatted statistics"""
    print(f"\n{' Concurrency Test Results ':-^50}")
    print(f"Concurrent Workers: {concurrent_workers}")
    print(f"Total Requests: {stats['total']}")
    print(f"Successful: {stats['success']} ({stats['success']/stats['total']:.1%})")
    print(f"Failures: {stats['failures']}")
    print(f"Errors: {stats['errors']}")
    
    if stats['latencies']:
        print("\nLatency Metrics:")
        print(f"Average: {np.mean(stats['latencies']):.2f}s")
        print(f"Median: {np.median(stats['latencies']):.2f}s")
        print(f"90th %ile: {np.percentile(stats['latencies'], 90):.2f}s")
        print(f"95th %ile: {np.percentile(stats['latencies'], 95):.2f}s")
        print(f"Max: {np.max(stats['latencies']):.2f}s")
        print(f"\nThroughput: {stats['success']/duration:.2f} requests/sec")

    if stats['error_types']:
        print("\nError Breakdown:")
        error_counts = defaultdict(int)
        for error in stats['error_types']:
            error_counts[error] += 1
        for error, count in error_counts.items():
            print(f"- {error}: {count}x")

async def warmup(session):
    """Warm up the model with initial requests"""
    print("Running warmup requests...")
    warmup_stats = create_stats_dict()  # Use properly initialized stats dictionary
    warmup_tasks = [send_request(session, asyncio.Semaphore(1), warmup_stats) 
                   for _ in range(WARMUP_REQUESTS)]
    await asyncio.gather(*warmup_tasks)
    print("Warmup complete!")

async def main():
    # Extended concurrency levels
    concurrency_levels = [90]  # Testing higher loads
    requests_per_level = 100  # Increased sample size
    
    async with aiohttp.ClientSession() as session:
        await warmup(session)
    
    for concurrency in concurrency_levels:
        stats = create_stats_dict()
        stats['total'] = requests_per_level
        
        start_time = time.time()
        print(f"\n{' Starting Test: '+str(concurrency)+' workers ':=^50}")
        await run_concurrent_test(concurrency, requests_per_level, stats)
        
        duration = time.time() - start_time
        print_stats(concurrency, stats, duration)
        
        print("\nCooling down...")
        await asyncio.sleep(15)  # Increased cooldown for higher loads




if __name__ == "__main__":
    start = datetime.now()
    print(f"Starting load test at {start.strftime('%Y-%m-%d %H:%M:%S')}")
    asyncio.run(main())
    print(f"\nTotal test duration: {datetime.now()-start}")