"""
Main script to run all NLP tasks
"""

import sys
import os
import time

# Add tasks directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'tasks'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

# Import all task modules
from task1_sentiment_analysis import sentiment_analysis
from task2_news_classification import news_classification
from task3_fake_news_detection import fake_news_detection
from task4_named_entity_recognition import named_entity_recognition
from task5_topic_modeling import topic_modeling

def run_all_tasks():
    """
    Run all NLP tasks sequentially.
    """
    print("=" * 80)
    print("NATURAL LANGUAGE PROCESSING TASKS - COMPLETE SUITE")
    print("=" * 80)
    print("This script will run all 5 NLP tasks with sample data.")
    print("For production use, please download the recommended datasets.")
    print("=" * 80)
    
    tasks = [
        ("Task 1: Sentiment Analysis", sentiment_analysis),
        ("Task 2: News Category Classification", news_classification),
        ("Task 3: Fake News Detection", fake_news_detection),
        ("Task 4: Named Entity Recognition", named_entity_recognition),
        ("Task 5: Topic Modeling", topic_modeling)
    ]
    
    results = {}
    total_start_time = time.time()
    
    for task_name, task_function in tasks:
        print(f"\n{'='*20} STARTING {task_name.upper()} {'='*20}")
        
        start_time = time.time()
        
        try:
            task_results = task_function()
            results[task_name] = {
                'status': 'completed',
                'results': task_results,
                'duration': time.time() - start_time
            }
            
            print(f"\n✓ {task_name} completed successfully!")
            print(f"  Duration: {results[task_name]['duration']:.2f} seconds")
            
        except Exception as e:
            print(f"\n✗ Error in {task_name}: {str(e)}")
            results[task_name] = {
                'status': 'failed',
                'error': str(e),
                'duration': time.time() - start_time
            }
        
        print(f"{'='*60}")
    
    total_duration = time.time() - total_start_time
    
    # Print summary
    print("\n" + "="*80)
    print("EXECUTION SUMMARY")
    print("="*80)
    
    completed_tasks = 0
    failed_tasks = 0
    
    for task_name, task_info in results.items():
        status_symbol = "✓" if task_info['status'] == 'completed' else "✗"
        print(f"{status_symbol} {task_name}: {task_info['status'].upper()} "
              f"({task_info['duration']:.2f}s)")
        
        if task_info['status'] == 'completed':
            completed_tasks += 1
        else:
            failed_tasks += 1
    
    print(f"\nTotal Tasks: {len(tasks)}")
    print(f"Completed: {completed_tasks}")
    print(f"Failed: {failed_tasks}")
    print(f"Total Duration: {total_duration:.2f} seconds")
    
    print(f"\nAll results and visualizations have been saved to the 'results' folder.")
    print("="*80)
    
    return results

def run_individual_task(task_number):
    """
    Run a specific task by number.
    
    Args:
        task_number (int): Task number (1-5)
    """
    tasks = {
        1: ("Sentiment Analysis", sentiment_analysis),
        2: ("News Category Classification", news_classification),
        3: ("Fake News Detection", fake_news_detection),
        4: ("Named Entity Recognition", named_entity_recognition),
        5: ("Topic Modeling", topic_modeling)
    }
    
    if task_number not in tasks:
        print(f"Invalid task number: {task_number}")
        print("Please choose a task number between 1 and 5.")
        return
    
    task_name, task_function = tasks[task_number]
    
    print(f"Running Task {task_number}: {task_name}")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        results = task_function()
        duration = time.time() - start_time
        
        print(f"\n✓ Task {task_number} completed successfully!")
        print(f"Duration: {duration:.2f} seconds")
        
        return results
        
    except Exception as e:
        print(f"\n✗ Error in Task {task_number}: {str(e)}")
        return None

def main():
    """
    Main function to handle command line arguments.
    """
    if len(sys.argv) > 1:
        try:
            task_num = int(sys.argv[1])
            run_individual_task(task_num)
        except ValueError:
            print("Please provide a valid task number (1-5) or run without arguments for all tasks.")
    else:
        run_all_tasks()

if __name__ == "__main__":
    main()