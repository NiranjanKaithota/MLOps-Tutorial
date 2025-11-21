from clearml import Task

# Connect to ClearML immediately
task = Task.init(project_name="MetroPT Maintenance", task_name="Training Experiment")

# Your training code will go here later...
print("ClearML Task Initialized!")